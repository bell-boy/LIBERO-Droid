"""Render a LIBERO benchmark initial state with MuJoCo's Filament renderer."""

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state", type=int, default=0)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--size", type=int, default=960, help="Square image size in pixels")
    parser.add_argument("--no-shadows", action="store_true")
    parser.add_argument("--filament-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    if not args.filament_python.is_file():
        parser.error("--filament-python must point to a Python executable")

    import numpy as np

    from libero.libero import benchmark
    from libero.libero.envs.env_wrapper import ControlEnv

    benchmarks = benchmark.get_benchmark_dict()
    if args.suite not in benchmarks:
        parser.error(f"unknown suite {args.suite!r}. Choose from {', '.join(benchmarks)}")
    suite = benchmarks[args.suite]()
    if not 0 <= args.task_id < suite.get_num_tasks():
        parser.error(f"--task-id is out of range for {args.suite}")
    states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.init_state < len(states):
        parser.error("--init-state is out of range for this task")

    with TemporaryDirectory(prefix="libero-filament-") as directory:
        model_path = Path(directory) / "model.xml"
        pose_path = Path(directory) / "pose.npz"
        env = ControlEnv(
            bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
            gripper_types="Robotiq85Gripper",
            use_camera_obs=False,
            has_offscreen_renderer=False,
            hard_reset=False,
        )
        try:
            env.seed(0)
            env.env.reset()
            env.set_init_state(states[args.init_state])
            model_path.write_text(env.env.model.get_xml(), encoding="utf-8")
            data = env.sim.data
            np.savez(
                pose_path,
                qpos=data.qpos,
                mocap_pos=data.mocap_pos,
                mocap_quat=data.mocap_quat,
            )
        finally:
            env.close()

        command = [
            str(args.filament_python.absolute()),
            "-I",
            str(Path(__file__).with_name("_filament_renderer.py").resolve()),
            str(model_path),
            str(pose_path),
            str(args.output),
            args.camera,
            str(args.size),
        ]
        if args.no_shadows:
            command.append("--no-shadows")
        subprocess.run(command, check=True)

    print(f"{suite.get_task(args.task_id).language}\nSaved {args.output}")


if __name__ == "__main__":
    main()
