"""One paired-benchmark rollout. The Modal worker brokers private inference."""
import json
import sys
from multiprocessing.connection import Client
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from openpi_client import msgpack_numpy

from pi05_droid_eval import configure_libero, policy_request
from compare_progress import TaskProgress


class Classic:
    def __init__(self, bddl):
        from libero.libero.envs.env_wrapper import OffScreenRenderEnv
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, gripper_types="PandaGripper",
                                      controller="JOINT_POSITION", camera_heights=256, camera_widths=256,
                                      control_freq=15, ignore_done=True)

    def observation(self, obs):
        robot = self.env.robots[0]
        width = np.abs(self.env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes]).sum()
        return dict(exterior_image=obs["agentview_image"][::-1].copy(),
                    wrist_image=obs["robot0_eye_in_hand_image"][::-1].copy(),
                    joint_position=obs["robot0_joint_pos"].copy(),
                    gripper_position=np.array([1 - np.clip(width / .08, 0, 1)]))

    def reset(self, state):
        from libero.libero.envs.droid import _install_absolute_targets
        self.env.seed(7)
        self.env.reset()
        _install_absolute_targets(self.env.robots[0].controller)
        obs = self.env.set_init_state(state)
        hold = np.r_[obs["robot0_joint_pos"], -1.]
        for _ in range(10):
            obs, *_ = self.env.step(hold)
        return self.observation(obs)

    def step(self, action):
        obs, *_ = self.env.step(np.r_[action[:7], 2 * action[7] - 1])
        return self.observation(obs), self.env.check_success()


def main(variant, suite_name, task_id, trial, directory, socket):
    from libero.libero import benchmark
    from libero.libero.envs.droid import DroidEnv
    suite = benchmark.get_benchmark_dict()[suite_name]()
    bddl = suite.get_task_bddl_file_path(task_id)
    prompt = suite.get_task(task_id).language
    renderer = {"classic": "mujoco", "droid": "mujoco", "droid_filament": "filament"}[variant]
    env = Classic(bddl) if variant == "classic" else DroidEnv(bddl, renderer=renderer)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        obs = env.reset(np.asarray(suite.get_task_init_states(task_id)[trial]))
        if renderer == "filament":
            context = env.env.env._filament_context
            if context is None or not context._model_loaded:
                raise RuntimeError("Filament did not render the initial policy observation")
        score = TaskProgress(env.env, Path(bddl).stem)
        with Client(socket, family="AF_UNIX") as conn, imageio.get_writer(
            directory / "trajectory.mp4", fps=15, codec="libx264", macro_block_size=1,
            output_params=["-threads", "1"]) as video:
            def frame():
                video.append_data(np.concatenate([obs["exterior_image"], obs["wrist_image"]], axis=1))
            frame()
            for step in range(530):
                if step % 15 == 0:
                    request = policy_request(obs, prompt)
                    request["sampling_seed"] = int(np.random.SeedSequence([7, task_id, trial, step]).generate_state(1)[0])
                    conn.send_bytes(msgpack_numpy.packb(request))
                    chunk = np.asarray(msgpack_numpy.unpackb(conn.recv_bytes())["actions"])
                    if chunk.ndim != 2 or chunk.shape[1] != 8 or len(chunk) < 15 or not np.isfinite(chunk).all():
                        raise ValueError(f"Invalid policy chunk {chunk.shape}")
                action = chunk[step % 15].copy()
                action[7] = float(action[7] > .5)
                obs, success = env.step(action)
                tp = score.update(success)
                frame()
                if success:
                    break
        row = dict(variant=variant, renderer=renderer, suite=suite_name, task_id=task_id, task=prompt, trial=trial,
                   steps=step+1, success=bool(success), tp=tp, milestones=score.values)
        (directory / "episode.json").write_text(json.dumps(row, indent=2))
    finally:
        env.env.close()


if __name__ == "__main__":
    configure_libero(Path("/opt/LIBERO-Droid"))
    variant, suite, task, trial, out, sock = sys.argv[1:]
    main(variant, suite, int(task), int(trial), Path(out), sock)
