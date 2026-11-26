"""Batched MuJoCo Warp physics and RGB cameras for DROID evaluation."""

from contextlib import contextmanager
from types import SimpleNamespace

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from .droid import (
    CONTROL_PERIOD, DROID_KQ, DROID_KQD, DROID_KX, DROID_KXD, DroidEnv,
    EXTERIOR_CAMERA, IMAGE_SIZE, WRIST_CAMERA, droid_control_action,
)


@contextmanager
def robosuite_mujoco_compatibility():
    """Adapt robosuite 1.4's bindings to the MuJoCo 3.17 API."""
    from robosuite.controllers import base_controller
    from robosuite.utils.binding_utils import MjData, MjModel

    # MuJoCo 3.17 takes the MjData directly and no longer exposes its qM array.
    original = base_controller.mujoco
    joint_types = MjModel.jnt_type
    MjData.qM = property(lambda data: data._data)
    base_controller.mujoco = SimpleNamespace(
        mj_fullM=lambda model, dst, data: mujoco.mj_fullM(model, data, dst)
    )
    # robosuite checks NumPy joint types with tuple membership, which invokes
    # the enum's comparison first. MuJoCo 3.17 expects Python integers there.
    MjModel.jnt_type = property(lambda model: joint_types.fget(model).astype(object))
    try:
        yield
    finally:
        MjModel.jnt_type = joint_types
        base_controller.mujoco = original
        del MjData.qM


@wp.kernel
def _apply_hybrid_torque(
    qpos: wp.array2d[float], qvel: wp.array2d[float], bias: wp.array2d[float],
    jacp: wp.array3d[float], jacr: wp.array3d[float],
    control: wp.array2d[float],
    qpos_ids: wp.array[int], dof_ids: wp.array[int], actuator_ids: wp.array[int],
    kp: wp.array[float], kd: wp.array[float], kx: wp.vec3, kr: wp.vec3,
    dx: wp.vec3, dr: wp.vec3, limits: wp.array[wp.vec2],
    ctrl: wp.array2d[float],
):
    world, joint = wp.tid()
    if joint == 7:
        low, high = limits[joint][0], limits[joint][1]
        ctrl[world, actuator_ids[joint]] = low + 0.5 * (1.0 + control[world, joint]) * (high - low)
        return

    dof = dof_ids[joint]
    error = control[world, joint] - qpos[world, qpos_ids[joint]]
    torque = kp[joint] * error - kd[joint] * qvel[world, dof] + bias[world, dof]
    # Project the Cartesian impedance back to this joint with J.T.
    for axis in range(3):
        p_error = float(0.0)
        r_error = float(0.0)
        p_vel = float(0.0)
        r_vel = float(0.0)
        for j in range(7):
            other_dof = dof_ids[j]
            other_error = control[world, j] - qpos[world, qpos_ids[j]]
            p_error += jacp[world, axis, other_dof] * other_error
            r_error += jacr[world, axis, other_dof] * other_error
            p_vel += jacp[world, axis, other_dof] * qvel[world, other_dof]
            r_vel += jacr[world, axis, other_dof] * qvel[world, other_dof]
        torque += jacp[world, axis, dof] * (kx[axis] * p_error - dx[axis] * p_vel)
        torque += jacr[world, axis, dof] * (kr[axis] * r_error - dr[axis] * r_vel)
    ctrl[world, actuator_ids[joint]] = wp.clamp(torque, limits[joint][0], limits[joint][1])


# The reset controller runs on CPU, so copy the solver warm start and actuator
# state as well as the positions and velocities of each settled scene.
_RESET_FIELDS = ("time", "qpos", "qvel", "act", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat")
# The closed Robotiq and manipulable objects can exceed Warp's reset-based
# contact heuristic once the arm starts moving.
CONTACTS_PER_WORLD = 256
CONSTRAINTS_PER_WORLD = 1024


class DroidWarpEnv:
    """Evaluate several initial states of one Spatial task in one Warp model."""

    def __init__(self, bddl_file_name, *, robotiq_asset, device="cuda:0"):
        if device.startswith("cuda") and not wp.is_cuda_available():
            raise RuntimeError("MuJoCo Warp evaluation requires an NVIDIA CUDA device.")
        self.device = wp.get_device(device)
        self._host = DroidEnv(bddl_file_name, renderer="warp", robotiq_asset=robotiq_asset, camera_obs=False)
        self.gripper_type = self._host.gripper_type

    def describe(self):
        return {**self._host.describe(), "physics": "mujoco-warp", "warp_device": str(self.device),
                "mujoco_version": mujoco.__version__, "mujoco_warp_version": mjw.__version__}

    def reset_batch(self, initial_states):
        resets = []
        transferred_states = []
        fixture_poses = None
        for initial_state in initial_states:
            self._host.reset(initial_state)
            model = self._host.sim.model
            poses = np.column_stack((model.body_pos, model.body_quat))
            if fixture_poses is not None and not np.array_equal(fixture_poses, poses):
                raise ValueError("Warp trials must share fixture poses in the compiled model")
            fixture_poses = poses
            data = self._host.sim.data._data
            resets.append({name: np.array(getattr(data, name), copy=True) for name in _RESET_FIELDS})
            transferred_states.append(self._host.initial_state.copy())

        with wp.ScopedDevice(self.device):
            self._initialize_batch(resets)
            observations, _ = self._observe_batch(range(self._data.nworld))
        return observations, transferred_states

    def _initialize_batch(self, resets):
        robot = self._host.robot
        model = self._host.sim.model._model
        nworld = len(resets)
        if model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError("Warp DROID control requires a split-step MuJoCo integrator")

        self._graph = None
        self._model = mjw.put_model(model)
        self._data = mjw.put_data(
            model, self._host.sim.data._data, nworld=nworld,
            nconmax=CONTACTS_PER_WORLD, njmax=CONSTRAINTS_PER_WORLD,
        )
        self._model.opt.warn_overflow = mjw.OverflowType.ALL & ~(
            mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
        )
        for name in _RESET_FIELDS:
            dst = getattr(self._data, name)
            if dst.size:
                wp.copy(dst, wp.array(np.stack([state[name] for state in resets]), dtype=dst.dtype))

        body_id = model.body(robot.robot_model.eef_name).id
        # Slicing a Warp array by body ID returns a strided view across worlds.
        self._flange_positions = self._data.xpos[:, body_id]
        self._flange_ids = wp.full(nworld, body_id, dtype=int)
        self._jacp = wp.empty((nworld, 3, model.nv), dtype=float)
        self._jacr = wp.empty_like(self._jacp)
        self._control = wp.empty((nworld, 8), dtype=float)
        gripper_id, = [model.actuator(name).id for name in robot.gripper.actuators]
        self._torque_inputs = [
            self._data.qpos, self._data.qvel, self._data.qfrc_bias,
            self._jacp, self._jacr, self._control,
            wp.array(robot._ref_joint_pos_indexes, dtype=int),
            wp.array(robot._ref_joint_vel_indexes, dtype=int),
            wp.array([*robot._ref_joint_actuator_indexes, gripper_id], dtype=int),
            wp.array(DROID_KQ, dtype=float), wp.array(DROID_KQD, dtype=float),
            wp.vec3(*DROID_KX[:3]), wp.vec3(*DROID_KX[3:]),
            wp.vec3(*DROID_KXD[:3]), wp.vec3(*DROID_KXD[3:]),
            wp.array(np.vstack((np.asarray(robot.torque_limits).T, model.actuator_ctrlrange[gripper_id])),
                     dtype=wp.vec2),
        ]
        self._substeps = int(CONTROL_PERIOD / model.opt.timestep)
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self._substep()
            self._graph = capture.graph

        height, width = IMAGE_SIZE
        self._render_context = mjw.create_render_context(
            model, nworld=nworld, cam_res=(width, height), render_rgb=True,
            cam_active=[EXTERIOR_CAMERA, WRIST_CAMERA], enabled_geom_groups=[1],
            use_shadows=True,
        )
        self._rgb = wp.empty((nworld, height, width), dtype=wp.vec3)
        self._last_actions = np.zeros((nworld, 8))

    def _substep(self):
        mjw.step1(self._model, self._data)
        mjw.jac(self._model, self._data, self._jacp, self._jacr, self._flange_positions, self._flange_ids)
        wp.launch(_apply_hybrid_torque, (self._data.nworld, 8), inputs=self._torque_inputs,
                  outputs=[self._data.ctrl])
        mjw.step2(self._model, self._data)

    def step_batch(self, actions):
        """Advance one policy tick. Actions are keyed by active world index."""
        with wp.ScopedDevice(self.device):
            self._last_actions[list(actions)] = list(actions.values())
            qpos = self._states[:, 1:1 + self._host.sim.model.nq]
            controls = droid_control_action(
                self._last_actions, qpos[:, self._host.robot._ref_joint_pos_indexes], self._host.joint_limits,
            )
            wp.copy(self._control, wp.array(controls, dtype=float))
            for _ in range(self._substeps):
                if self._graph is None:
                    self._substep()
                else:
                    wp.capture_launch(self._graph)
            return self._observe_batch(actions)

    def _observe_batch(self, worlds):
        d = self._data
        overflow = d.overflow.numpy() & int(self._model.opt.warn_overflow)
        if np.any(overflow):
            raise RuntimeError(f"MuJoCo Warp exceeded physics capacity, flags by world: {overflow.tolist()}")
        self._states = np.column_stack((d.time.numpy(), d.qpos.numpy(), d.qvel.numpy()))
        if not np.isfinite(self._states).all():
            raise RuntimeError("MuJoCo Warp produced a nonfinite simulation state")

        mjw.fwd_kinematics(self._model, d)
        mjw.refit_bvh(self._model, d, self._render_context)
        mjw.render(self._model, d, self._render_context)
        images = {}
        for index, key in enumerate(("exterior_image", "wrist_image")):
            mjw.get_rgb(self._render_context, index, self._rgb)
            images[key] = np.rint(self._rgb.numpy() * 255).astype(np.uint8)

        observations, successes = {}, {}
        self._object_positions = {}
        for world in worlds:
            self._load_state(world)
            self._object_positions[world] = self._host.object_positions()
            observations[world] = {
                **self._host.proprioception(),
                **{key: rgb[world] for key, rgb in images.items()},
            }
            successes[world] = bool(self._host.check_success())
        return observations, successes

    def _load_state(self, world):
        # Reconstruct CPU collision geometry for LIBERO predicates and the
        # measured distance between finger pads. Integration stays on Warp.
        sim = self._host.sim
        sim.set_state_from_flattened(self._states[world])
        mujoco.mj_fwdPosition(sim.model._model, sim.data._data)

    def get_sim_state(self, world):
        return self._states[world].copy()

    def object_positions(self, world):
        return self._object_positions[world]

    def close(self):
        self._host.close()
