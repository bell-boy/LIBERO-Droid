"""LIBERO tasks in the DROID robot setup.

DroidEnv wraps a LIBERO task with the hardware conventions of DROID: the
MolmoSpaces Robotiq 2F-85, the DROID reset pose, DROID's hybrid joint
impedance controller at 15 Hz, a DROID exterior camera placed relative to the
robot base, and the calibrated DROID wrist camera. Any policy trained on DROID
can use its observations and actions directly.

Observations:
    exterior_image    (180, 320, 3) uint8, upright
    wrist_image       (180, 320, 3) uint8, upright
    joint_position    (7,) measured arm joints in radians
    gripper_position  (1,) measured closure, 0 open and 1 closed

Actions: seven absolute joint targets in radians and a gripper closure in [0, 1].
"""
from math import pi
from pathlib import Path
import types

import mujoco
import numpy as np

from . import robotiq
from .env_wrapper import OffScreenRenderEnv

DEFAULT_ROBOTIQ_ASSET = Path(__file__).resolve().parents[3] / ".libero/robotiq_2f85_v4/2f85.xml"
EXTERIOR_CAMERA = "agentview"
WRIST_CAMERA = "robot0_eye_in_hand"
IMAGE_SIZE = (180, 320)

# droid/robot_env.py reset joints.
DROID_RESET_POSE = (0., -pi / 5, 0., -4 * pi / 5, 0., 3 * pi / 5, 0.)
DROID_RESET_SOURCE = "https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/robot_env.py#L26-L29"
CONTROL_FREQ = 15
CONTROL_PERIOD = 0.066
SETTLE_STEPS = 10
MAX_JOINT_DELTA = 0.2
OPEN_WIDTH = 0.085

# DROID config/panda/franka_hardware.yaml, angular components last.
DROID_KQ = np.array([40, 30, 50, 25, 35, 25, 10])
DROID_KQD = np.array([4, 6, 5, 5, 3, 2, 1])
DROID_KX = np.array([400, 400, 400, 15, 15, 15])
DROID_KXD = np.array([37, 37, 37, 2, 2, 2])

DROID_EXTRINSICS = "https://huggingface.co/KarlP/droid/blob/bcb840c3b496533e0adf548a54b51f2f00057837/cam2base_extrinsics.json"
DROID_INTRINSICS = "https://huggingface.co/KarlP/droid/blob/bcb840c3b496533e0adf548a54b51f2f00057837/intrinsics.json"
DROID_WRIST_EXTRINSICS = "https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/calibration/calibration_info.json"
# Median ZED 2 exterior calibration across DROID episodes: fy = 525 px at 1280×720.
EXTERIOR_FOVY = 68.9
# Median pose, in the robot base frame with level roll, of the left-side DROID
# exterior cameras whose view holds the shoulder, the reset gripper and the
# table center (23% of calibrated left cameras).
EXTERIOR_POSITION = (0.233, 0.566, 0.454)
EXTERIOR_YAW_PITCH_DEG = (-64.3, 18.7)
# Left rectified lens of ZED Mini 19824535, scaled from the 1280×720 calibration.
WRIST_INTRINSICS_PX = tuple(v / 4 for v in (733.6874389648438, 733.6874389648438, 650.0894775390625, 354.6379699707031))
WRIST_POSITION = (-0.07528310648512061, 0.02872136764193708, 0.023316565592628957)
WRIST_EULER_XYZ = (-0.3255773557130956, 0.006247432431090383, -1.562874530416053)


def look_at_rotation(position, target):
    """MuJoCo camera local axes: +x right, +y up, -z viewing direction."""
    backward = np.asarray(position, float) - np.asarray(target, float)
    backward /= np.linalg.norm(backward)
    right = np.cross([0., 0., 1.], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    return np.column_stack([right, up, backward])


def _configure_exterior(env):
    model, data = env.sim.model, env.sim.data
    base_id = model.body_name2id(f"{env.robots[0].robot_model.naming_prefix}link0")
    base_rotation = data.body_xmat[base_id].reshape(3, 3)
    cid = model.camera_name2id(EXTERIOR_CAMERA)
    if model.cam_bodyid[cid] != 0:
        raise ValueError("The exterior camera must be world-fixed")
    yaw, pitch = np.deg2rad(EXTERIOR_YAW_PITCH_DEG)
    forward = base_rotation @ [np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), -np.sin(pitch)]
    position = data.body_xpos[base_id] + base_rotation @ EXTERIOR_POSITION
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, look_at_rotation(position, position + forward).ravel())
    model.cam_pos[cid] = position
    model.cam_quat[cid] = quat
    model.cam_fovy[cid] = EXTERIOR_FOVY


def _configure_wrist(env):
    model = env.sim.model
    prefix = env.robots[0].robot_model.naming_prefix
    wrist_id = model.camera_name2id(WRIST_CAMERA)
    hand_id = model.cam_bodyid[wrist_id]
    if model.body_parentid[hand_id] != model.body_name2id(f"{prefix}link7"):
        raise ValueError("Expected a Panda hand-mounted wrist camera")
    wrist_quat, link7_quat, inverse_hand, local_quat = (np.zeros(4) for _ in range(4))
    # DROID uses fixed-axis XYZ in the flange frame, 107 mm along link 7 Z.
    # The OpenCV lens axes (+X right, +Y down, +Z forward) need a 180° X rotation.
    mujoco.mju_euler2Quat(wrist_quat, np.asarray(WRIST_EULER_XYZ), "XYZ")
    mujoco.mju_mulQuat(link7_quat, wrist_quat, np.array([0., 1., 0., 0.]))
    link7_position = np.asarray(WRIST_POSITION) + np.array([0., 0., .107])
    height, width = IMAGE_SIZE
    fx, fy, cx, cy = WRIST_INTRINSICS_PX
    # MuJoCo uses offsets about the sensor center, with the opposite horizontal sign.
    model.cam_resolution[wrist_id] = [width, height]
    model.cam_sensorsize[wrist_id] = [width, height]
    model.cam_intrinsic[wrist_id] = [fx, fy, width / 2 - cx, cy - height / 2]
    mujoco.mju_negQuat(inverse_hand, model.body_quat[hand_id])
    mujoco.mju_mulQuat(local_quat, inverse_hand, link7_quat)
    local_position = np.zeros(3)
    mujoco.mju_rotVecQuat(local_position, link7_position - model.body_pos[hand_id], inverse_hand)
    model.cam_pos[wrist_id] = local_position
    model.cam_quat[wrist_id] = local_quat
    model.cam_fovy[wrist_id] = np.rad2deg(np.arctan(cy / fy) + np.arctan((height - cy) / fy))


def _install_absolute_targets(controller):
    """Treat JOINT_POSITION actions as absolute targets instead of scaled deltas."""
    original = controller.set_goal

    def absolute_goal(self, action, set_qpos=None):
        return original(np.zeros(7), set_qpos=np.asarray(action) if set_qpos is None else set_qpos)

    controller.set_goal = types.MethodType(absolute_goal, controller)


def _install_hybrid_feedback(env):
    """Replace the simulated torque law with DROID's hybrid joint impedance."""
    controller = env.robots[0].controller
    if controller.interpolator is not None:
        raise ValueError("DROID control expects zero-order-held joint targets")
    from robosuite.controllers.base_controller import Controller

    flange = env.robots[0].robot_model.eef_name

    def hybrid_torque(self):
        self.update()
        if self.goal_qpos is None:
            self.reset_goal()
        data = self.sim.data
        # MuJoCo's right_hand origin approximates panda_link8 to 0.5 mm.
        J = np.vstack((data.get_body_jacp(flange), data.get_body_jacr(flange)))[:, self.qvel_index]
        error = self.goal_qpos - self.joint_pos
        dq = self.joint_vel
        cartesian = DROID_KX * (J @ error) - DROID_KXD * (J @ dq)
        self.torques = DROID_KQ * error - DROID_KQD * dq + J.T @ cartesian + self.torque_compensation
        Controller.run_controller(self)
        return self.torques

    controller.run_controller = types.MethodType(hybrid_torque, controller)


class DroidEnv:
    """A LIBERO task with the DROID robot, cameras and controller."""

    def __init__(self, bddl_file_name, renderer="filament", robotiq_asset=DEFAULT_ROBOTIQ_ASSET, seed=7):
        self.gripper_type = robotiq.register_gripper(robotiq_asset)
        height, width = IMAGE_SIZE
        self.env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_file_name), renderer=renderer, gripper_types=self.gripper_type,
            controller="JOINT_POSITION", control_freq=CONTROL_FREQ, ignore_done=True,
            camera_names=[EXTERIOR_CAMERA, WRIST_CAMERA], camera_heights=height, camera_widths=width,
        )
        self.renderer = renderer
        self.seed = seed

    @property
    def sim(self):
        return self.env.sim

    @property
    def robot(self):
        return self.env.robots[0]

    def reset(self, init_state):
        """Start from a LIBERO initial state with the Robotiq open and the arm at the DROID reset pose."""
        env, robot = self.env, self.robot
        env.seed(self.seed)
        env.reset()
        # Hard resets replace the controller, so patch it every episode.
        controller = robot.controller
        if controller.name != "JOINT_POSITION" or controller.control_dim != 7:
            raise ValueError("Expected the seven-joint JOINT_POSITION controller")
        _install_absolute_targets(controller)
        state = robotiq.transfer_initial_state(env, init_state)
        state[1 + np.asarray(robot._ref_joint_pos_indexes)] = DROID_RESET_POSE
        state[1 + env.sim.model.nq + np.asarray(robot._ref_joint_vel_indexes)] = 0
        self.initial_state = state
        env.set_init_state(state)
        _configure_exterior(env)
        _configure_wrist(env)
        env.sim.forward()
        obs = env.env._get_observations(force_update=True)
        self.joint_limits = env.sim.model.jnt_range[robot._ref_joint_indexes]
        env.env.control_timestep = 1 / CONTROL_FREQ
        hold = np.r_[obs["robot0_joint_pos"], -1.0]
        for _ in range(SETTLE_STEPS):
            obs, *_ = env.step(hold)
        # Switch to DROID control after the arm settles under the same target.
        _install_hybrid_feedback(env)
        env.env.control_timestep = CONTROL_PERIOD
        return self._observation(obs)

    def step(self, action):
        """Apply absolute joint targets and a gripper closure. Returns the observation and LIBERO success."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError(f"Expected 7 joint targets and a gripper closure, got {action}")
        measured = self.sim.data.qpos[self.robot._ref_joint_pos_indexes]
        # Scale the whole arm displacement so no joint moves more than MAX_JOINT_DELTA per step.
        displacement = action[:7] - measured
        scale = max(1.0, np.max(np.abs(displacement)) / MAX_JOINT_DELTA)
        target = np.clip(measured + displacement / scale, self.joint_limits[:, 0], self.joint_limits[:, 1])
        grip = 2 * np.clip(action[7], 0, 1) - 1
        obs, *_ = self.env.step(np.r_[target, grip])
        return self._observation(obs), self.env.check_success()

    def _observation(self, obs):
        robot = self.robot
        width = robotiq.finger_gap(robot)
        return {
            # MuJoCo renders bottom-up. Both cameras look world-up, so a vertical flip restores them.
            "exterior_image": np.ascontiguousarray(obs[f"{EXTERIOR_CAMERA}_image"][::-1]),
            "wrist_image": np.ascontiguousarray(obs[f"{WRIST_CAMERA}_image"][::-1]),
            "joint_position": self.sim.data.qpos[robot._ref_joint_pos_indexes].copy(),
            "gripper_position": np.array([1.0 - np.clip(width / OPEN_WIDTH, 0.0, 1.0)]),
        }

    def get_sim_state(self):
        return self.env.get_sim_state()

    def check_success(self):
        return self.env.check_success()

    def object_positions(self):
        return {name: self.sim.data.body_xpos[body].copy() for name, body in self.env.env.obj_body_id.items()}

    def describe(self):
        """Setup details for run manifests."""
        return {
            "renderer": self.renderer,
            "gripper": f"MolmoSpaces DROID Robotiq 2F-85 ({robotiq.ASSET_URL})",
            "initial_arm_pose": {"joint_positions_rad": DROID_RESET_POSE, "source": DROID_RESET_SOURCE},
            "controller": {"type": "DROID hybrid joint impedance", "period_s": CONTROL_PERIOD,
                           "max_joint_delta_rad": MAX_JOINT_DELTA, "settle_steps": SETTLE_STEPS},
            "exterior_camera": {"position_in_base_m": EXTERIOR_POSITION, "yaw_pitch_deg": EXTERIOR_YAW_PITCH_DEG,
                                "vertical_fov_deg": EXTERIOR_FOVY, "size": IMAGE_SIZE,
                                "references": [DROID_EXTRINSICS, DROID_INTRINSICS]},
            "wrist_camera": {"serial": "19824535", "intrinsics_px": WRIST_INTRINSICS_PX, "size": IMAGE_SIZE,
                             "references": [DROID_WRIST_EXTRINSICS, DROID_INTRINSICS]},
        }

    def close(self):
        self.env.close()
