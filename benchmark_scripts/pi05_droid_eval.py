"""Evaluate pi05_droid_jointpos on LIBERO Spatial tasks in the DROID setup.

The policy server (pi05_serve_jointpos.py) must already be listening.
"""
import argparse
from collections import deque
import json
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_droid_jointpos"
MAX_STEPS = 530  # 35 s at DROID's 66 ms control period.


def configure_libero(libero_root):
    """Point LIBERO at this checkout before importing it, skipping its first-run prompt."""
    base = libero_root / "libero/libero"
    config_dir = Path(os.environ.setdefault("LIBERO_CONFIG_PATH", str(libero_root / ".libero")))
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(base), "bddl_files": str(base / "bddl_files"),
        "init_states": str(base / "init_files"), "assets": str(base / "assets"),
        "datasets": str(libero_root / "datasets"),
    }))
    sys.path.insert(0, str(libero_root))


def connect(host, port, inference_timeout=300):
    """OpenPI client that tolerates the server's long first JAX compilation."""
    from openpi_client import msgpack_numpy
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    import websockets.sync.client

    class Client(WebsocketClientPolicy):
        def _wait_for_server(self):
            # The server runs inference in its event loop, so it cannot pong while JAX compiles.
            while True:
                try:
                    connection = websockets.sync.client.connect(
                        self._uri, compression=None, max_size=None,
                        ping_interval=None, ping_timeout=None, open_timeout=10)
                    return connection, msgpack_numpy.unpackb(connection.recv(timeout=30))
                except ConnectionRefusedError:
                    logging.info("Waiting for policy server at %s", self._uri)
                    time.sleep(3)

        def infer(self, obs):
            self._ws.send(self._packer.pack(obs))
            response = self._ws.recv(timeout=inference_timeout)
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            return msgpack_numpy.unpackb(response)

    return Client(host, port)


def policy_request(obs, prompt):
    from openpi_client import image_tools

    def image(rgb):
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb, 224, 224))

    return {
        "observation/exterior_image_1_left": image(obs["exterior_image"]),
        "observation/wrist_image_left": image(obs["wrist_image"]),
        "observation/joint_position": obs["joint_position"].astype(np.float32),
        "observation/gripper_position": obs["gripper_position"].astype(np.float32),
        "prompt": str(prompt),
    }


def run_episode(env, client, init_state, prompt, episode_dir, trial, replan_steps, binary_gripper, sampling_seed):
    import imageio.v2 as imageio

    obs = env.reset(init_state)
    plan = deque()
    actions, states, inference_seconds = [], [], []
    positions = {name: [xyz] for name, xyz in env.object_positions().items()}
    success, steps = False, 0
    with imageio.get_writer(episode_dir / "trajectory.mp4", fps=15, codec="libx264", macro_block_size=1) as writer:
        writer.append_data(np.concatenate([obs["exterior_image"], obs["wrist_image"]], axis=1))
        for step in range(MAX_STEPS):
            if not plan:
                request = policy_request(obs, prompt)
                if sampling_seed is not None:
                    request["sampling_seed"] = int(np.random.SeedSequence([sampling_seed, trial, step]).generate_state(1)[0])
                tick = time.perf_counter()
                chunk = np.asarray(client.infer(request)["actions"])
                inference_seconds.append(time.perf_counter() - tick)
                if chunk.ndim != 2 or chunk.shape[1] != 8 or len(chunk) < replan_steps or not np.isfinite(chunk).all():
                    raise ValueError(f"Invalid action chunk: {chunk.shape}")
                plan.extend(chunk[:replan_steps])
            action = plan.popleft().copy()
            if binary_gripper:
                # OpenPI's DROID client closes the hand above 0.5.
                action[7] = float(action[7] > 0.5)
            actions.append(action)
            states.append(env.get_sim_state())
            obs, success = env.step(action)
            steps += 1
            writer.append_data(np.concatenate([obs["exterior_image"], obs["wrist_image"]], axis=1))
            for name, xyz in env.object_positions().items():
                positions[name].append(xyz)
            if success:
                break
    np.savez_compressed(episode_dir / "trajectory.npz", actions=np.asarray(actions), sim_states=np.asarray(states),
                        terminal_sim_state=env.get_sim_state(), initial_state=env.initial_state,
                        source_initial_state=np.asarray(init_state),
                        **{f"{name}_xyz": np.asarray(xyz) for name, xyz in positions.items()})
    return dict(success=bool(success), status="success" if success else "timeout", steps=steps,
                inference_seconds=inference_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--replan-steps", type=int, default=15, help="Actions executed per policy query")
    parser.add_argument("--binary-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--renderer", choices=("filament", "mujoco"), default="filament")
    parser.add_argument("--robotiq-asset", type=Path)
    parser.add_argument("--libero-revision", default="unknown")
    parser.add_argument("--openpi-revision", default="unknown")
    args = parser.parse_args()
    if args.trials < 1 or args.replan_steps < 1:
        parser.error("--trials and --replan-steps must be positive")

    libero_root = args.libero_root.resolve()
    configure_libero(libero_root)
    from libero.libero import benchmark
    from libero.libero.envs.droid import DEFAULT_ROBOTIQ_ASSET, DroidEnv
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pi05_pick_metrics import score_pick_episodes

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(args.task_id)
    bddl = suite.get_task_bddl_file_path(args.task_id)
    initial_states = suite.get_task_init_states(args.task_id)
    if args.trials > len(initial_states):
        parser.error(f"Task {args.task_id} has only {len(initial_states)} initial states")

    client = connect(args.host, args.port)
    metadata = client.get_server_metadata()
    if metadata.get("action_space") != "absolute_joint_position" or metadata.get("checkpoint") != CHECKPOINT:
        raise SystemExit(f"Expected a server for the absolute joint-position checkpoint {CHECKPOINT}")
    if args.sampling_seed is not None and (args.sampling_seed < 0 or not metadata.get("sampling_seed")):
        raise SystemExit("Seeded sampling needs a nonnegative seed and a server that supports it")

    env = DroidEnv(bddl, renderer=args.renderer, robotiq_asset=args.robotiq_asset or DEFAULT_ROBOTIQ_ASSET)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"checkpoint": CHECKPOINT, "server": metadata, "setup": env.describe(),
                "policy": {"replan_steps": args.replan_steps,
                           "gripper_command": "binary, close above 0.5" if args.binary_gripper else "continuous",
                           "image_preprocessing": "resize_with_pad to 224 x 224", "sampling_seed": args.sampling_seed},
                "suite": "libero_spatial", "task_id": args.task_id, "trials": args.trials,
                "libero_revision": args.libero_revision, "openpi_revision": args.openpi_revision}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    try:
        for trial in range(args.trials):
            episode_dir = output / "libero_spatial" / f"task_{args.task_id:03d}" / f"episode_{trial:03d}"
            episode_dir.mkdir(parents=True)
            started = time.perf_counter()
            row = dict(task_id=args.task_id, task=task.language, prompt=task.language, episode=trial,
                       initial_state_index=trial,
                       **run_episode(env, client, initial_states[trial], task.language, episode_dir, trial,
                                     args.replan_steps, args.binary_gripper, args.sampling_seed),
                       wall_seconds=time.perf_counter() - started)
            (episode_dir / "episode.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)
            print(f"task {args.task_id} trial {trial}: {row['status']} after {row['steps']} steps", flush=True)
    finally:
        env.close()
    successes = sum(row["success"] for row in rows)
    summary = {"overall": {"episodes": len(rows), "successes": successes, "success_rate": successes / len(rows),
                           "steps": {"max": max(row["steps"] for row in rows)}}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    score_pick_episodes(output, bddl, env.gripper_type, args.task_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
