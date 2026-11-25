"""Offscreen robosuite context backed by a persistent MuJoCo 4 process."""

import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys

import mujoco
import numpy as np


# The physics environment owns these poses, including fixtures moved on reset.
_MODEL_FIELDS = ("body_pos", "body_quat", "cam_pos", "cam_quat", "cam_fovy",
                 "cam_resolution", "cam_sensorsize", "cam_intrinsic", "site_rgba")
_DATA_FIELDS = ("qpos", "mocap_pos", "mocap_quat")
_RGB_ONLY = (
    "Filament evaluation supports RGB only. "
    "Use renderer='mujoco' for depth or segmentation."
)


def filament_python():
    executable = shutil.which(os.environ.get("LIBERO_FILAMENT_PYTHON", ""))
    if executable is None:
        raise RuntimeError(
            "Set LIBERO_FILAMENT_PYTHON to the Python executable in a separate "
            "environment with mujoco==4.0.1 installed. See the LIBERO README."
        )
    return os.path.abspath(executable)


class FilamentRenderContext:
    """Implements the offscreen context interface used by robosuite's MjSim."""

    offscreen = True

    def __init__(self):
        self.vopt = mujoco.MjvOption()
        self._python = filament_python()
        self._process = None
        self._xml = None
        self._model_loaded = False

    def attach(self, sim):
        self._model = sim.model._model
        self._data = sim.data._data
        xml = sim.model.get_xml()
        if xml != self._xml:
            self._xml = xml
            self._model_loaded = False
        sim.add_render_context(self)

    def _request(self, message):
        try:
            pickle.dump(message, self._process.stdin, protocol=4)
            self._process.stdin.flush()
            success, response = pickle.load(self._process.stdout)
        except (OSError, EOFError, pickle.UnpicklingError) as error:
            self.close()
            raise RuntimeError(
                "The Filament renderer process stopped unexpectedly. Check its logs."
            ) from error
        if not success:
            self.close()
            raise RuntimeError(f"Filament renderer: {response}")
        return response

    def render(self, width, height, camera_id=None, segmentation=False):
        if segmentation:
            raise ValueError(_RGB_ONLY)
        if self._process is None:
            self._process = subprocess.Popen(
                [self._python, "-I", str(Path(__file__).resolve())],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        if not self._model_loaded:
            self._request(("load", self._xml))
            self._model_loaded = True

        model_state = tuple(getattr(self._model, name).tobytes() for name in _MODEL_FIELDS)
        data_state = tuple(getattr(self._data, name).tobytes() for name in _DATA_FIELDS)
        pixels = self._request((
            "render",
            width,
            height,
            camera_id,
            model_state,
            data_state,
            self.vopt.geomgroup.tobytes(),
            self.vopt.sitegroup.tobytes(),
        ))
        # MjSim exposes bottom-up pixels. robosuite's sensors apply the selected
        # OpenGL/OpenCV convention after calling read_pixels.
        pixels = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
        self._pixels = pixels[::-1].copy()

    def read_pixels(self, width, height, depth=False, segmentation=False):
        if depth or segmentation:
            raise ValueError(_RGB_ONLY)
        return self._pixels

    def close(self):
        process, self._process = self._process, None
        self._model_loaded = False
        if process is None:
            return
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
        finally:
            process.stdout.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


class _FilamentScene:
    def __init__(self, context, xml):
        from mujoco import mjrf
        from mujoco.rendering.filament import renderer

        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.model.light_castshadow[:] = True
        self.options = mujoco.MjvOption()
        self.engine = renderer.Renderer(context)
        self.scene = self.engine.scene("eval")
        self.scene.configure_from_model(self.model)
        self.resources = mjrf.ModelResources(context, self.model)
        self.lights = mjrf.ModelLights(self.scene, self.resources)
        self.renderables = mjrf.ModelRenderables(self.scene, self.resources)
        # robosuite hides sites with nonpositive alpha. Manage site membership
        # explicitly because Filament can still draw those translucent sites.
        self.options.sitegroup[:] = False
        self.renderables.set_options(self.options)
        self.visible_sites = np.zeros(self.model.nsite, dtype=bool)
        self.Camera = mjrf.Camera
        self.View = renderer.View
        self.targets = {}

    def render(
        self, width, height, camera_id, model_state, data_state, geomgroup, sitegroup
    ):
        for obj, fields, values in (
            (self.model, _MODEL_FIELDS, model_state),
            (self.data, _DATA_FIELDS, data_state),
        ):
            for name, value in zip(fields, values):
                array = getattr(obj, name)
                array[...] = np.frombuffer(value, dtype=array.dtype).reshape(array.shape)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.options.geomgroup[:] = np.frombuffer(geomgroup, dtype=np.uint8)
        self.renderables.set_options(self.options)
        self.lights.update(self.data)
        self.renderables.update(self.data)
        site_groups = np.frombuffer(sitegroup, dtype=np.uint8)
        visible_sites = (self.model.site_rgba[:, 3] > 0) & (
            site_groups[self.model.site_group] != 0
        )
        for index in np.flatnonzero(self.visible_sites != visible_sites):
            site = self.renderables.get_renderable(mujoco.mjtObj.mjOBJ_SITE, int(index))
            if visible_sites[index]:
                self.scene.add_renderable(site)
            else:
                self.scene.remove_renderable(site)
        self.visible_sites = visible_sites

        camera = mujoco.MjvCamera()
        if camera_id is None or camera_id == -1:
            mujoco.mjv_defaultFreeCamera(self.model, camera)
        else:
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            camera.fixedcamid = camera_id
        name = f"{width}x{height}"
        if name not in self.targets:
            self.targets[name] = np.empty((height, width, 3), dtype=np.uint8)
            self.engine.target(name, self.targets[name])
        gl_camera = mujoco.mjv_camera2GLCamera(self.model, self.data, camera)
        # Retain the horizontal frustum and account for Filament's opposite
        # vertical lens-shift convention for off-center principal points.
        projection = self.Camera(
            pos=gl_camera.pos, forward=gl_camera.forward, up=gl_camera.up,
            frustum_center=gl_camera.frustum_center, frustum_width=gl_camera.frustum_width,
            frustum_bottom=-gl_camera.frustum_top, frustum_top=-gl_camera.frustum_bottom,
            frustum_near=gl_camera.frustum_near, frustum_far=gl_camera.frustum_far,
            orthographic=gl_camera.orthographic,
        )
        view = self.View(
            scene="eval",
            target=name,
            camera=projection,
            shadows=True,
        )
        self.engine.render([view])
        return self.targets[name].tobytes()


def _serve(replies):
    from mujoco import mjrf

    context = None
    scene = None
    while True:
        try:
            command, *args = pickle.load(sys.stdin.buffer)
        except EOFError:
            return
        try:
            if command == "load":
                if context is None:
                    context = mjrf.Context(mjrf.ContextConfig())
                scene = None
                scene = _FilamentScene(context, *args)
                response = None
            else:
                response = scene.render(*args)
            pickle.dump((True, response), replies, protocol=4)
            replies.flush()
        except Exception as error:
            pickle.dump((False, str(error)), replies, protocol=4)
            replies.flush()
            return


if __name__ == "__main__":
    # Filament writes native logs to stdout. Send those to stderr and reserve
    # the original pipe for requests and replies between the two interpreters.
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as replies:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        _serve(replies)
