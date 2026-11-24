"""Render snapshots in the separate MuJoCo 4 environment."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from mujoco import mjrf
from mujoco.rendering.filament import renderer
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("pose", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("camera")
    parser.add_argument("size", type=int)
    parser.add_argument("--no-shadows", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    with np.load(args.pose) as pose:
        data.qpos[:] = pose["qpos"]
        data.mocap_pos[:] = pose["mocap_pos"]
        data.mocap_quat[:] = pose["mocap_quat"]
    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera)
    if camera_id < 0:
        parser.error(f"camera {args.camera!r} is absent from the task model")
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = camera_id

    shadows = not args.no_shadows
    model.light_castshadow[:] = shadows
    context = mjrf.Context(mjrf.ContextConfig())
    engine = renderer.Renderer(context)
    scene = engine.scene("task")
    scene.configure_from_model(model)
    resources = mjrf.ModelResources(context, model)
    lights = mjrf.ModelLights(scene, resources)
    renderables = mjrf.ModelRenderables(scene, resources)
    options = mujoco.MjvOption()
    options.geomgroup[0] = False  # Collision meshes
    options.sitegroup[:] = False  # Task regions and robot debug markers
    renderables.set_options(options)
    lights.update(data)
    renderables.update(data)

    pixels = np.empty((args.size, args.size, 3), dtype=np.uint8)
    engine.target("image", pixels)
    engine.render([
        renderer.View(
            scene="task",
            target="image",
            camera=mujoco.mjv_camera2GLCamera(model, data, camera),
            shadows=shadows,
        )
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output)


if __name__ == "__main__":
    main()
