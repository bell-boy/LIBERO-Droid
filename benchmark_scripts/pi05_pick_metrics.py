"""Score pickup of the target bowl in LIBERO Spatial tasks from saved MuJoCo states."""

from itertools import chain
import json
from pathlib import Path

import numpy as np


# Every LIBERO Spatial goal places this bowl on the plate.
TARGET = "akita_black_bowl_1"
MIN_LIFT_M = 0.02
MIN_FRAMES = 10


def measure_pick(env, trajectory):
    """A held bowl must be lifted, touching both jaws, and clear of other geometry."""
    model = env.sim.model
    data = env.sim.data
    gripper = env.robots[0].gripper
    body = env.env.obj_body_id[TARGET]
    target = set(np.flatnonzero(model.geom_bodyid == body))
    pads = [{model.geom_name2id(name) for name in gripper.important_geoms[side + "_fingerpad"]}
            for side in ("left", "right")]
    hand = {i for i in range(model.ngeom) if (model.geom_id2name(i) or "").startswith(gripper.naming_prefix)}
    first_frame = None
    longest = run = 0
    peak_lift = 0.0
    with np.load(trajectory) as archive:
        z0 = archive[TARGET + "_xyz"][0, 2]
        saved = archive["sim_states"]
        if not len(saved) or "terminal_sim_state" not in archive:
            raise ValueError(f"{trajectory} lacks the full simulation states needed to score a pickup.")
        states = chain(saved, [archive["terminal_sim_state"]])
        for frame, state in enumerate(states):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            touched = set()
            for contact in data.contact[:data.ncon]:
                if contact.geom1 in target:
                    touched.add(contact.geom2)
                if contact.geom2 in target:
                    touched.add(contact.geom1)
            lift = float(data.body_xpos[body, 2] - z0)
            held = all(touched & pad for pad in pads) and not (touched - hand - target)
            if held:
                peak_lift = max(peak_lift, lift)
            run = run + 1 if held and lift > MIN_LIFT_M else 0
            if run == MIN_FRAMES and first_frame is None:
                first_frame = frame - MIN_FRAMES + 1
            longest = max(longest, run)
    return dict(picked=first_frame is not None, target=TARGET, first_pick_frame=first_frame,
                longest_hold_frames=longest, peak_unsupported_grasp_lift_m=peak_lift,
                minimum_lift_m=MIN_LIFT_M, minimum_consecutive_frames=MIN_FRAMES)


def score_pick_episodes(output, bddl_file, gripper_type, task_id=0):
    from libero.libero.envs.env_wrapper import ControlEnv

    env = ControlEnv(bddl_file_name=str(bddl_file), gripper_types=gripper_type, controller="JOINT_POSITION",
                     control_freq=15, use_camera_obs=False, has_offscreen_renderer=False)
    try:
        trajectories = sorted(Path(output).glob(f"libero_spatial/task_{task_id:03d}/episode_*/trajectory.npz"))
        for trajectory in trajectories:
            env.reset()
            metric = measure_pick(env, trajectory)
            trajectory.with_name("pick.json").write_text(json.dumps(metric, indent=2) + "\n")
            print(f"{trajectory.parent.name}: picked={metric['picked']}, "
                  f"longest unsupported grasp={metric['longest_hold_frames']} frames", flush=True)
    finally:
        env.close()
