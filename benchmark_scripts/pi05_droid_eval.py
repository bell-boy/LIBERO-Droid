"""Evaluate pi05_droid_jointpos on LIBERO Spatial tasks in the DROID setup.

The policy server (pi05_serve_jointpos.py) must already be listening.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import json
import logging
import multiprocessing
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
                    with ExitStack() as resources:
                        connection = resources.enter_context(websockets.sync.client.connect(
                            self._uri, compression=None, max_size=None,
                            ping_interval=None, ping_timeout=None, open_timeout=10,
                        ))
                        metadata = msgpack_numpy.unpackb(connection.recv(timeout=inference_timeout))
                        self._connection = resources.pop_all()
                        return connection, metadata
                except ConnectionRefusedError:
                    logging.info("Waiting for policy server at %s", self._uri)
                    time.sleep(3)

        def infer(self, obs):
            self._ws.send(self._packer.pack(obs))
            response = self._ws.recv(timeout=inference_timeout)
            if isinstance(response, str):
                raise RuntimeError(f"Error in inference server:\n{response}")
            return msgpack_numpy.unpackb(response)

        def close(self):
            self._connection.close()

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


class Episode:
    def __init__(self, directory, writer, obs, positions, source_state, initial_state, trial, started):
        self.directory = directory
        self.writer = writer
        self.obs = obs
        self.positions = {name: [xyz] for name, xyz in positions.items()}
        self.source_state = source_state
        self.initial_state = initial_state
        self.trial = trial
        self.plan = deque()
        self.actions, self.states, self.inference_seconds = [], [], []
        self.started = started
        self.write_frame()

    def write_frame(self):
        self.writer.append_data(np.concatenate([self.obs["exterior_image"], self.obs["wrist_image"]], axis=1))

    def request(self, prompt, step, sampling_seed):
        request = policy_request(self.obs, prompt)
        if sampling_seed is not None:
            seed = np.random.SeedSequence([sampling_seed, self.trial, step])
            request["sampling_seed"] = int(seed.generate_state(1)[0])
        return request

    def use_chunk(self, chunk, elapsed, replan_steps):
        chunk = np.asarray(chunk)
        if (chunk.ndim != 2 or chunk.shape[1] != 8 or len(chunk) < replan_steps or not np.isfinite(chunk).all()):
            raise ValueError(f"Invalid action chunk: {chunk.shape}")
        self.inference_seconds.append(elapsed)
        self.plan.extend(chunk[:replan_steps])

    def next_action(self, binary_gripper):
        action = self.plan.popleft().copy()
        if binary_gripper:
            action[7] = float(action[7] > 0.5)
        return action

    def record(self, action, state, obs, positions):
        self.actions.append(action)
        self.states.append(state)
        self.obs = obs
        self.write_frame()
        for name, xyz in positions.items():
            self.positions[name].append(xyz)

    def finish(self, terminal_state, success, prompt, task_id):
        self.writer.close()
        np.savez_compressed(
            self.directory / "trajectory.npz",
            actions=np.asarray(self.actions), sim_states=np.asarray(self.states),
            terminal_sim_state=terminal_state, initial_state=self.initial_state,
            source_initial_state=np.asarray(self.source_state),
            **{f"{name}_xyz": np.asarray(xyz) for name, xyz in self.positions.items()},
        )
        row = dict(
            task_id=task_id, task=prompt, prompt=prompt, episode=self.trial,
            initial_state_index=self.trial, success=bool(success),
            status="success" if success else "timeout", steps=len(self.actions),
            inference_seconds=self.inference_seconds, wall_seconds=time.perf_counter() - self.started,
        )
        (self.directory / "episode.json").write_text(json.dumps(row, indent=2) + "\n")
        print(f"task {task_id} trial {self.trial}: {row['status']} after {row['steps']} steps", flush=True)
        return row


def replan(episodes, client, prompt, step, args):
    waiting = [episode for episode in episodes if not episode.plan]
    batch_size = args.policy_batch_size
    for start in range(0, len(waiting), batch_size):
        group = waiting[start:start + batch_size]
        requests = [episode.request(prompt, step, args.sampling_seed) for episode in group]
        tick = time.perf_counter()
        if batch_size == 1:
            chunks = [client.infer(requests[0])["actions"]]
        else:
            # Keep the compiled batch shape stable when episodes finish early.
            requests.extend([requests[-1]] * (batch_size - len(requests)))
            chunks = np.asarray(client.infer({"observations": requests})["actions"])
            if chunks.ndim != 3 or len(chunks) != batch_size:
                raise ValueError(f"Invalid batched action shape: {chunks.shape}")
            chunks = chunks[:len(group)]
        elapsed = time.perf_counter() - tick
        for episode, chunk in zip(group, chunks, strict=True):
            episode.use_chunk(chunk, elapsed, args.replan_steps)


def run_trials(env, client, trials, prompt, args):
    import imageio.v2 as imageio

    use_warp = args.backend == "warp"
    rows = []
    batch_size = args.num_envs if use_warp else 1
    for start in range(0, len(trials), batch_size):
        started = time.perf_counter()
        batch = trials[start:start + batch_size]
        states = [state for _, state in batch]
        if use_warp:
            obs, transferred = env.reset_batch(states)
            get_state = env.get_sim_state
            get_positions = env.object_positions
        else:
            obs, transferred = {0: env.reset(states[0])}, [env.initial_state]
            get_state = lambda world: env.get_sim_state()
            get_positions = lambda world: env.object_positions()

        with ExitStack() as writers:
            episodes = {}
            # Each parallel world has its own encoder.
            encoder_options = {"output_params": ["-threads", "1"]} if args.num_envs > 1 else {}
            for world, (trial, state) in enumerate(batch):
                directory = args.output / "libero_spatial" / f"task_{args.task_id:03d}" / f"episode_{trial:03d}"
                directory.mkdir(parents=True)
                writer = writers.enter_context(
                    imageio.get_writer(directory / "trajectory.mp4", fps=15, codec="libx264",
                                       macro_block_size=1, **encoder_options)
                )
                episodes[world] = Episode(
                    directory, writer, obs[world], get_positions(world), state, transferred[world], trial, started,
                )
            max_steps = min(args.max_steps, MAX_STEPS)
            for step in range(max_steps):
                replan(episodes.values(), client, prompt, step, args)
                actions = {world: episode.next_action(args.binary_gripper)
                           for world, episode in episodes.items()}
                previous = {world: get_state(world) for world in episodes}
                if use_warp:
                    obs, successes = env.step_batch(actions)
                else:
                    observation, success = env.step(actions[0])
                    obs, successes = {0: observation}, {0: success}

                for world in list(episodes):
                    episode = episodes[world]
                    episode.record(actions[world], previous[world], obs[world], get_positions(world))
                    if successes[world] or step + 1 == max_steps:
                        rows.append(episode.finish(get_state(world), successes[world], prompt, args.task_id))
                        del episodes[world]
                if not episodes:
                    break
    return rows


def run_trial_in_process(bddl, asset, trial, prompt, args):
    from libero.libero.envs.droid import DroidEnv

    with ExitStack() as resources:
        env = DroidEnv(bddl, renderer=args.renderer or "filament", robotiq_asset=asset)
        resources.callback(env.close)
        client = connect(args.host, args.port)
        resources.callback(client.close)
        return run_trials(env, client, [trial], prompt, args)[0]


def run_process_trials(bddl, asset, trials, prompt, args):
    jobs = ((bddl, asset, trial, prompt, args) for trial in trials)
    return run_process_jobs(run_trial_in_process, jobs, args.num_envs)


def run_process_jobs(worker, jobs, max_workers):
    # Fresh interpreters own their MuJoCo and graphics contexts.
    ctx = multiprocessing.get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx)
    try:
        futures = [pool.submit(worker, *job) for job in jobs]
        return [future.result() for future in as_completed(futures)]
    finally:
        pool.shutdown(cancel_futures=True)


def open_warp_env(resources, bddl, asset):
    try:
        from libero.libero.envs.droid_warp import DroidWarpEnv, robosuite_mujoco_compatibility
    except ModuleNotFoundError as error:
        if error.name not in ("mujoco_warp", "warp"):
            raise
        raise RuntimeError("Install the pinned MuJoCo Warp dependencies listed in the README") from error
    resources.enter_context(robosuite_mujoco_compatibility())
    env = DroidWarpEnv(bddl, robotiq_asset=asset)
    resources.callback(env.close)
    return env


def run_warp_task(bddl, asset, trials, prompt, args):
    from pi05_pick_metrics import score_pick_episodes

    with ExitStack() as resources:
        env = open_warp_env(resources, bddl, asset)
        client = connect(args.host, args.port)
        resources.callback(client.close)
        rows = run_trials(env, client, trials, prompt, args)
        score_pick_episodes(args.output, bddl, env.gripper_type, args.task_id)
        return args.task_id, env.describe(), rows


def policy_metadata(client, args):
    metadata = client.get_server_metadata()
    if metadata.get("action_space") != "absolute_joint_position" or metadata.get("checkpoint") != CHECKPOINT:
        raise SystemExit(f"Expected a server for the absolute joint-position checkpoint {CHECKPOINT}")
    if args.sampling_seed is not None and (args.sampling_seed < 0 or not metadata.get("sampling_seed")):
        raise SystemExit("Seeded sampling needs a nonnegative seed and a server that supports it")
    if args.policy_batch_size > 1 and not metadata.get("batched_inference"):
        raise SystemExit("Batched inference requires a policy server that supports it")
    return metadata


def summarize(rows):
    successes = sum(row["success"] for row in rows)
    return {"episodes": len(rows), "successes": successes, "success_rate": successes / len(rows),
            "steps": {"max": max(row["steps"] for row in rows), "total": sum(row["steps"] for row in rows)}}


def run_warp_tasks(jobs, task_ids, manifest, args):
    client = connect(args.host, args.port)
    try:
        manifest["server"] = policy_metadata(client, args)
    finally:
        client.close()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest.update(task_ids=task_ids, task_workers=args.task_workers,
                    execution="warp_task_processes" if args.task_workers > 1 else "warp_batch")
    manifest_file = args.output / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.task_workers > 1:
        results = run_process_jobs(run_warp_task, jobs, args.task_workers)
    else:
        results = [run_warp_task(*job) for job in jobs]
    results.sort(key=lambda result: result[0])
    manifest["task_setups"] = {str(task_id): setup for task_id, setup, _ in results}
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {"overall": summarize([row for _, _, rows in results for row in rows]),
               "tasks": {str(task_id): summarize(rows) for task_id, _, rows in results}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    task_options = parser.add_mutually_exclusive_group()
    task_options.add_argument("--task-id", type=int)
    task_options.add_argument("--task-ids", help="Comma-separated Spatial task IDs for Warp evaluation")
    parser.add_argument("--task-workers", type=int, default=1, help="Concurrent CPU processes across Warp tasks")
    parser.add_argument("--trials", type=int, default=1, help="Number of initial states per task")
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--replan-steps", type=int, default=15, help="Actions executed per policy query")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="Episode step limit for throughput tests")
    parser.add_argument("--binary-gripper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", choices=("mujoco", "warp"), default="mujoco")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Concurrent trials of the same task (MuJoCo processes or Warp worlds)")
    parser.add_argument("--policy-batch-size", type=int, default=1, help="Seeded policy requests per Warp inference batch")
    parser.add_argument("--renderer", choices=("filament", "mujoco"), help="Renderer for the MuJoCo backend")
    parser.add_argument("--robotiq-asset", type=Path)
    parser.add_argument("--libero-revision", default="unknown")
    parser.add_argument("--openpi-revision", default="unknown")
    args = parser.parse_args()
    if min(args.trials, args.num_envs, args.replan_steps, args.task_workers) < 1:
        parser.error("--trials, --num-envs, --replan-steps and --task-workers must be positive")
    if not 1 <= args.max_steps <= MAX_STEPS:
        parser.error(f"--max-steps must be between 1 and {MAX_STEPS}")
    if not 1 <= args.policy_batch_size <= min(args.num_envs, args.trials):
        parser.error("--policy-batch-size must be between 1 and the number of concurrent trials")
    if args.policy_batch_size > 1 and (args.backend != "warp" or args.sampling_seed is None):
        parser.error("Batched inference requires --backend warp and --sampling-seed")
    if args.backend == "warp" and args.renderer:
        parser.error("--backend warp uses Warp RGB rendering and takes no --renderer argument")
    if args.backend != "warp" and (args.task_ids is not None or args.task_workers > 1):
        parser.error("--task-ids and --task-workers above 1 require --backend warp")
    try:
        task_ids = ([int(value) for value in args.task_ids.split(",")] if args.task_ids is not None
                    else [args.task_id if args.task_id is not None else 0])
    except ValueError:
        parser.error("--task-ids must be a comma-separated list of integers")
    if len(set(task_ids)) != len(task_ids):
        parser.error("--task-ids must be unique")

    libero_root = args.libero_root.resolve()
    configure_libero(libero_root)
    from libero.libero import benchmark
    from libero.libero.envs.droid import DEFAULT_ROBOTIQ_ASSET, DroidEnv
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pi05_pick_metrics import score_pick_episodes

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    if any(not 0 <= task_id < suite.get_num_tasks() for task_id in task_ids):
        parser.error(f"LIBERO Spatial task IDs must be between 0 and {suite.get_num_tasks() - 1}")
    args.num_envs = min(args.num_envs, args.trials)
    args.task_workers = min(args.task_workers, len(task_ids))
    use_processes = args.backend == "mujoco" and args.num_envs > 1

    asset = args.robotiq_asset or DEFAULT_ROBOTIQ_ASSET
    jobs = []
    for task_id in task_ids:
        initial_states = suite.get_task_init_states(task_id)
        if args.trials > len(initial_states):
            parser.error(f"Task {task_id} has only {len(initial_states)} initial states")
        trials = [(trial, np.asarray(state)) for trial, state in enumerate(initial_states[:args.trials])]
        task_args = argparse.Namespace(**{**vars(args), "task_id": task_id})
        jobs.append((suite.get_task_bddl_file_path(task_id), asset, trials, suite.get_task(task_id).language, task_args))

    manifest = {"checkpoint": CHECKPOINT,
                "policy": {"replan_steps": args.replan_steps,
                           "batch_size": args.policy_batch_size,
                           "gripper_command": "binary, close above 0.5" if args.binary_gripper else "continuous",
                           "image_preprocessing": "resize_with_pad to 224 x 224", "sampling_seed": args.sampling_seed},
                "suite": "libero_spatial", "trials": args.trials, "backend": args.backend, "num_envs": args.num_envs,
                "max_steps": min(args.max_steps, MAX_STEPS),
                "libero_revision": args.libero_revision, "openpi_revision": args.openpi_revision}
    if len(jobs) > 1:
        run_warp_tasks(jobs, task_ids, manifest, args)
        return

    bddl, asset, trials, prompt, args = jobs[0]
    output = args.output
    with ExitStack() as resources:
        if args.backend == "warp":
            env = open_warp_env(resources, bddl, asset)
        else:
            env = DroidEnv(bddl, renderer=args.renderer or "filament", robotiq_asset=asset,
                           camera_obs=not use_processes)
            resources.callback(env.close)

        client = connect(args.host, args.port)
        resources.callback(client.close)
        manifest.update(server=policy_metadata(client, args), setup=env.describe(), task_id=args.task_id,
                        execution="warp_batch" if args.backend == "warp" else ("processes" if use_processes else "serial"))
        output.mkdir(parents=True, exist_ok=False)
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if use_processes:
            rows = run_process_trials(bddl, asset, trials, prompt, args)
        else:
            rows = run_trials(env, client, trials, prompt, args)
        summary = {"overall": summarize(rows)}
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        score_pick_episodes(output, bddl, env.gripper_type, args.task_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
