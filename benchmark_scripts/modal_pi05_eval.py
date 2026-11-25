"""Adapt the companion π0.5 evaluator to LIBERO-Droid's renderer and Robotiq."""

import argparse
from contextlib import ExitStack
from functools import partial
import json
import os
from pathlib import Path
from unittest.mock import patch

import yaml


def evaluation_environment(base_env, renderer, robotiq_asset=None):
    if robotiq_asset is None:
        gripper = "Robotiq85Gripper"
    else:
        from pi05_libero import robotiq
        from robosuite.utils.observables import Observable, sensor

        gripper = robotiq.register_gripper(robotiq_asset)

    class DroidRenderEnv(base_env):
        def __init__(self, **kwargs):
            kwargs.update(gripper_types=gripper, renderer=renderer)
            super().__init__(**kwargs)
            if robotiq_asset is not None:
                @sensor(modality="robot0_proprio")
                def finger_gap_m(obs_cache):
                    return robotiq.finger_gap(self.robots[0])

                self.env.add_observable(Observable(
                    "robot0_finger_gap_m", finger_gap_m, sampling_rate=self.env.control_freq
                ))

        def set_init_state(self, init_state):
            if robotiq_asset is not None and len(init_state) != 1 + self.sim.model.nq + self.sim.model.nv:
                init_state = robotiq.transfer_initial_state(self, init_state)
            return super().set_init_state(init_state)

    return DroidRenderEnv


def run(libero_root, output, libero_revision, openpi_revision, port, robotiq_asset=None, trials=3,
        sampling_seed=None, droid_pick=False, binary_gripper=False, full_chunks=False, renderer=None, task_id=0):
    from pi05_libero import adapters, evaluate

    if (binary_gripper or full_chunks) and not droid_pick:
        raise ValueError("Binary gripper control and full policy chunks require --droid-pick.")

    base = libero_root / "libero/libero"
    config_dir = Path(os.environ["LIBERO_CONFIG_PATH"])
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(base),
        "bddl_files": str(base / "bddl_files"),
        "init_states": str(base / "init_files"),
        "assets": str(base / "assets"),
        "datasets": str(libero_root / "datasets"),
    }))
    from libero.libero import envs

    renderer = renderer or "filament"

    policy_observation = adapters.policy_observation

    def droid_observation(obs, *args, **kwargs):
        width_key = "robot0_finger_gap_m" if robotiq_asset is not None else "robot0_panda_gripper_qpos"
        policy_input = dict(obs, robot0_gripper_qpos=obs[width_key])
        return policy_observation(policy_input, *args, **kwargs)

    revisions = {libero_root: libero_revision, config_dir.parent / "openpi": openpi_revision}
    argv = [
        "pi05-libero", "--host", "127.0.0.1", "--port", str(port),
        "--suites", "libero_spatial", "--task-ids", str(task_id), "--trials", str(trials),
        "--libero-root", str(libero_root), "--output", str(output), "--fail-fast",
    ]
    if droid_pick:
        if robotiq_asset is None:
            raise ValueError("The DROID pickup preset requires the MolmoSpaces Robotiq asset.")
        argv.extend([
            "--ai2-control-loop", "--gripper", "robotiq_2f85", "--initial-arm-pose", "droid_default_reset",
            "--camera-rig", "droid_oblique_calibrated_wrist",
            "--max-steps", "530", "--save-states",
        ])
        if not full_chunks:
            argv.extend(["--temporal-ensemble", "--replan-steps", "1"])
    else:
        argv.extend(["--camera-rig", "droid_oblique", "--replan-steps", "15"])
    if robotiq_asset is not None:
        argv.extend(["--gripper-open-width", "0.085"])
    if sampling_seed is not None:
        argv.extend(["--sampling-seed", str(sampling_seed)])
    try:
        with ExitStack() as patches:
            patches.enter_context(patch("sys.argv", argv))
            patches.enter_context(patch.object(evaluate, "ROOT", config_dir.parent))
            patches.enter_context(patch.object(evaluate, "revision", revisions.__getitem__))
            if droid_pick:
                from pi05_libero import robotiq

                render_env = partial(envs.OffScreenRenderEnv, renderer=renderer)
                patches.enter_context(patch.object(envs, "OffScreenRenderEnv", render_env))
                # The companion looks for this asset in its own checkout. Use the mounted copy.
                gripper = robotiq.register_gripper(robotiq_asset)
                patches.enter_context(patch.object(robotiq, "register_gripper", return_value=gripper))
            else:
                env_class = evaluation_environment(envs.OffScreenRenderEnv, renderer, robotiq_asset)
                patches.enter_context(patch.object(envs, "OffScreenRenderEnv", env_class))
                patches.enter_context(patch.object(adapters, "policy_observation", droid_observation))
            if binary_gripper:
                environment_action = adapters.environment_action

                def binary_action(target, joint_limits):
                    action, clipped = environment_action(target, joint_limits)
                    action[-1] = 1.0 if target[-1] > 0.5 else -1.0
                    return action, clipped

                patches.enter_context(patch.object(adapters, "environment_action", binary_action))
            evaluate.main()
        if droid_pick:
            from benchmark_scripts.pi05_pick_metrics import score_pick_episodes
            from libero.libero import benchmark

            suite = benchmark.get_benchmark_dict()["libero_spatial"]()
            score_pick_episodes(output, suite.get_task_bddl_file_path(task_id), gripper, task_id)
    finally:
        manifest = output / "manifest.json"
        if manifest.is_file():
            metadata = json.loads(manifest.read_text())
            metadata["simulation_renderer"] = renderer
            if droid_pick:
                metadata["gripper_command"] = "binary, close above 0.5" if binary_gripper else "continuous"
            metadata["args"]["gripper"] = "MolmoBotRobotiq85" if robotiq_asset else "Robotiq85Gripper"
            if robotiq_asset:
                from pi05_libero.robotiq import ASSET_URL
                metadata["gripper_asset"] = ASSET_URL
                metadata["gripper_open_width_m"] = 0.085
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-revision", required=True)
    parser.add_argument("--openpi-revision", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robotiq-asset", type=Path)
    parser.add_argument("--renderer", choices=("mujoco", "filament"))
    parser.add_argument("--binary-gripper", action="store_true")
    parser.add_argument("--full-chunks", action="store_true")
    parser.add_argument("--droid-pick", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--task-id", type=int, default=0)
    args = parser.parse_args()
    run(args.libero_root.resolve(), args.output, args.libero_revision, args.openpi_revision, args.port,
        args.robotiq_asset, args.trials, args.sampling_seed, args.droid_pick, args.binary_gripper, args.full_chunks, args.renderer,
        args.task_id)
