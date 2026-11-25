"""Adapt the MolmoSpaces DROID Robotiq 2F-85 MJCF to robosuite's gripper interface."""

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.robots.manipulators.panda_robot import Panda

GRIPPER_TYPE = "MolmoBotRobotiq85"
ASSET_URL = "https://huggingface.co/datasets/allenai/molmospaces/tree/5f802a46816ffc3520e59b5ef49fa8dafee8eeca/mujoco/robots/franka_droid/20260127"


def _inline_defaults(root):
    # robosuite's XML loader handles explicit flat defaults. The Ai2 asset
    # also inherits nested classes through body childclass attributes.
    defaults = {}

    def collect(node, inherited):
        attrs = {tag: attrib.copy() for tag, attrib in inherited.items()}
        for child in node:
            if child.tag != "default":
                attrs.setdefault(child.tag, {}).update(child.attrib)
        defaults[node.get("class")] = attrs
        for child in node.findall("default"):
            collect(child, attrs)

    def inline(node, inherited):
        cls = node.attrib.pop("class", inherited)
        for attr, value in defaults[cls].get(node.tag, {}).items():
            node.attrib.setdefault(attr, value)
        childclass = node.attrib.pop("childclass", inherited)
        for child in node:
            inline(child, childclass)

    root_default = root.find("default")
    collect(root_default, {})
    root.remove(root_default)
    inline(root, None)


def _adapt(source):
    tree = ET.parse(source)
    root = tree.getroot()
    _inline_defaults(root)
    for mesh in root.findall("./asset/mesh"):
        asset = source.parent / "assets" / mesh.attrib["file"]
        mesh.attrib.setdefault("name", asset.stem)
        mesh.set("file", str(asset))
    for geom in root.findall("./worldbody//geom"):
        geom.set("group", "1" if geom.get("group") == "2" else "0")
    base = root.find("./worldbody/body")
    # The Ai2 model attaches the Robotiq at z=.107 and +90° in link 7.
    # Robosuite attaches grippers under right_hand, whose XML uses a rounded quaternion.
    panda = Panda()
    hand = panda.worldbody.find(f".//body[@name='{panda.eef_name}']")
    hand_pos = np.fromstring(hand.attrib["pos"], sep=" ")
    hand_quat = np.fromstring(hand.attrib["quat"], sep=" ")
    hand_quat /= np.linalg.norm(hand_quat)
    inverse, position, quat = np.empty(4), np.empty(3), np.empty(4)
    mujoco.mju_negQuat(inverse, hand_quat)
    mujoco.mju_rotVecQuat(position, np.array([0., 0., .107]) - hand_pos, inverse)
    mujoco.mju_mulQuat(quat, inverse, np.array([np.sqrt(.5), 0., 0., np.sqrt(.5)]))
    base.set("pos", " ".join(map(str, position)))
    base.set("quat", " ".join(map(str, quat)))
    base.remove(base.find("camera"))  # The shared camera rig supplies the same link-7 pose.
    grasp = base.find("site[@name='grasp_site']")
    eef = ET.SubElement(base, "body", name="eef", pos=grasp.attrib["pos"],
                        quat="0.707106781187 0 0 -0.707106781187")
    for name in ("grip_site", "grip_site_cylinder", "ee_x", "ee_y", "ee_z"):
        ET.SubElement(eef, "site", name=name, size="0.001", rgba="0 0 0 0")
    ET.SubElement(base, "site", name="ft_frame", size="0.001", rgba="0 0 0 0")
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "force", name="force_ee", site="ft_frame")
    ET.SubElement(sensor, "torque", name="torque_ee", site="ft_frame")
    return tree


def register_gripper(source):
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError("Ai2 Robotiq asset missing. Run bash benchmark_scripts/setup_molmobot_gripper.sh")
    tree = _adapt(source)

    class MolmoBotRobotiq85(GripperModel):
        def __init__(self, idn=0):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "robotiq.xml"
                tree.write(path)
                super().__init__(str(path), idn)

        @property
        def init_qpos(self):
            return np.zeros(len(self.joints))

        def format_action(self, action):
            # robosuite scales [-1,1] into the source actuator's [0,255].
            return np.asarray(action)

        @property
        def _important_geoms(self):
            return {f"{side}_{key}": [f"{side}_pad1", f"{side}_pad2"]
                    for side in ("left", "right") for key in ("finger", "fingerpad")}

    GRIPPER_MAPPING[GRIPPER_TYPE] = MolmoBotRobotiq85
    return GRIPPER_TYPE


def transfer_initial_state(env, panda_state):
    """Keep the Panda arm, scene, and time while using the new open gripper state."""
    robot = env.robots[0]
    model = env.sim.model
    current = env.get_sim_state()
    qpos_grip = robot._ref_gripper_joint_pos_indexes
    qvel_grip = robot._ref_gripper_joint_vel_indexes
    nq_old = model.nq - len(qpos_grip) + 2
    nv_old = model.nv - len(qvel_grip) + 2
    panda_state = np.asarray(panda_state)
    if panda_state.shape != (1 + nq_old + nv_old,):
        raise ValueError("Unexpected LIBERO Panda initial-state layout")

    def replace_gripper(saved, default, indices):
        start = indices[0]
        if list(indices) != list(range(start, start + len(indices))):
            raise ValueError("Expected a contiguous gripper state")
        return np.r_[saved[:start], default[indices], saved[start + 2:]]

    return np.r_[panda_state[0],
                 replace_gripper(panda_state[1:1 + nq_old], current[1:1 + model.nq], qpos_grip),
                 replace_gripper(panda_state[1 + nq_old:], current[1 + model.nq:], qvel_grip)]


def finger_gap(robot):
    model, data = robot.sim.model._model, robot.sim.data._data
    ids = [model.geom(robot.gripper.correct_naming(f"{side}_pad2")).id for side in ("left", "right")]
    return max(0.0, mujoco.mj_geomDistance(model, data, *ids, 0.1, None))
