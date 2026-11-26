"""Run the π0.5 DROID evaluation on Modal."""
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
from urllib.request import urlopen

import modal

app = modal.App("libero-droid-pi05")
ROOT = Path(__file__).resolve().parents[1]
OPENPI_REV = "215abfb217dbac7d5f1273282331b9b1866c0479"
LEROBOT_REV = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
SIM_DEPS = (
    "'numpy<2' 'robosuite==1.4.0' 'bddl==1.0.1' future 'gym==0.25.2' "
    "easydict matplotlib 'opencv-python<4.12' 'imageio[ffmpeg]' 'openpi-client==0.1.2' pyyaml scipy termcolor"
)
image = (
    modal.Image.from_registry("python:3.11-slim-bookworm")
    .run_commands(
        "sed -i s@http://deb.debian.org@https://deb.debian.org@g /etc/apt/sources.list.d/debian.sources",
        "apt-get update && apt-get install -y --no-install-recommends "
        "git curl libgl1 libglib2.0-0 libegl1 libgles2 libegl-mesa0 libgl1-mesa-dri "
        "libosmesa6 libvulkan1 mesa-vulkan-drivers && rm -rf /var/lib/apt/lists/*",
    )
    .pip_install("uv")
    .run_commands(
        "uv venv --python 3.11 /opt/sim",
        "uv pip install --python /opt/sim/bin/python --index https://download.pytorch.org/whl/cpu 'torch==2.5.1'",
        f"uv pip install --python /opt/sim/bin/python 'mujoco==3.2.6' {SIM_DEPS}",
        "uv venv --python 3.11 /opt/sim-warp",
        "uv pip install --python /opt/sim-warp/bin/python --index https://download.pytorch.org/whl/cpu 'torch==2.5.1'",
        "uv pip install --python /opt/sim-warp/bin/python "
        f"'mujoco==3.17.0' 'mujoco-warp==3.17.0' 'warp-lang==1.19.0' {SIM_DEPS}",
        "uv venv --python 3.12 /opt/filament",
        "uv pip install --python /opt/filament/bin/python 'mujoco==4.0.1'",
    )
    .run_commands(
        "apt-get update && apt-get install -y --no-install-recommends "
        "gcc libc6-dev linux-libc-dev && rm -rf /var/lib/apt/lists/*"
    )
    .run_commands(
        "git init /opt/openpi && git -C /opt/openpi fetch --depth=1 "
        f"https://github.com/Physical-Intelligence/openpi.git {OPENPI_REV} "
        "&& GIT_LFS_SKIP_SMUDGE=1 git -C /opt/openpi checkout FETCH_HEAD",
        f"curl -fsSL --retry 4 https://codeload.github.com/huggingface/lerobot/tar.gz/{LEROBOT_REV} -o /tmp/lerobot.tar.gz",
        "cd /opt/openpi && uv sync --frozen --no-dev --no-install-package lerobot",
        "uv pip install --python /opt/openpi/.venv/bin/python --no-deps /tmp/lerobot.tar.gz && rm /tmp/lerobot.tar.gz",
        "uv cache clean",
    )
)

if modal.is_local():
    gripper_assets = ROOT / ".libero/robotiq_2f85_v4"
    image = image.add_local_dir(ROOT / "libero", "/opt/LIBERO-Droid/libero",
                                ignore=["**/__pycache__/**", "**/.DS_Store"])
    for script in ("pi05_droid_eval.py", "pi05_pick_metrics.py", "pi05_serve_jointpos.py"):
        image = image.add_local_file(ROOT / "benchmark_scripts" / script, f"/opt/LIBERO-Droid/benchmark_scripts/{script}")
    if (gripper_assets / "2f85.xml").is_file():
        image = image.add_local_dir(gripper_assets, "/opt/LIBERO-Droid/.libero/robotiq_2f85_v4")


@app.function(image=image, gpu="A100", cpu=8, memory=32768, timeout=2400)
def evaluate(libero_revision: str, backend: str, eval_args: list[str], sweep: list[tuple[int, int, int]]):
    with tempfile.TemporaryDirectory(prefix="libero-droid-") as workdir:
        return run_evaluation(Path(workdir), libero_revision, backend, eval_args, sweep)


def run_evaluation(workdir: Path, libero_revision: str, backend: str, eval_args: list[str], sweep):
    started = time.monotonic()
    out = workdir / "pi05-result"
    out.mkdir()
    runtime = workdir / "xdg"
    runtime.mkdir(mode=0o700)
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        PYTHONPATH="/opt/LIBERO-Droid",
        LIBERO_FILAMENT_PYTHON="/opt/filament/bin/python",
        LIBERO_CONFIG_PATH=str(workdir / ".libero"),
        MUJOCO_GL="osmesa",
        PYOPENGL_PLATFORM="osmesa",
        EGL_PLATFORM="surfaceless",
        OPENPI_DATA_HOME="/tmp/openpi-cache",
        XLA_PYTHON_CLIENT_MEM_FRACTION="0.75",
        LP_NUM_THREADS="4",
        OMP_NUM_THREADS="2",
        XDG_RUNTIME_DIR=str(runtime),
    )
    port = 8000
    scripts = Path("/opt/LIBERO-Droid/benchmark_scripts")
    sim_python = "/opt/sim-warp/bin/python" if backend == "warp" else "/opt/sim/bin/python"
    if backend == "warp":
        env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    telemetry_file = (out / "gpu.csv").open("w")
    telemetry = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu",
         "--format=csv,nounits", "--loop=2"],
        stdout=telemetry_file,
    )
    server = subprocess.Popen(
        ["/opt/openpi/.venv/bin/python", str(scripts / "pi05_serve_jointpos.py"), "--port", str(port)],
        env=env, cwd="/opt/openpi",
    )
    try:
        for _ in range(1200):
            if server.poll() is not None:
                raise RuntimeError(f"Policy server exited with status {server.returncode}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("Policy server did not start within 20 minutes")
        ready = time.monotonic()
        runs = []
        for index, config in enumerate(sweep or [None]):
            directory = "evaluation"
            overrides = []
            if config is not None:
                workers, worlds, batch = config
                directory = f"{index:02d}-{workers}workers-{worlds}worlds-{batch}batch"
                overrides = ["--task-workers", str(workers), "--num-envs", str(worlds),
                             "--policy-batch-size", str(batch)]
            begin = time.monotonic()
            timestamp = time.time()
            print(f"Starting {directory}", flush=True)
            result = subprocess.run(
                [sim_python, str(scripts / "pi05_droid_eval.py"), "--backend", backend,
                 "--libero-root", "/opt/LIBERO-Droid", "--libero-revision", libero_revision,
                 "--openpi-revision", OPENPI_REV, "--port", str(port), "--output", str(out / directory),
                 *eval_args, *overrides],
                env=env, cwd=str(workdir),
            )
            run = {"directory": directory, "exit_code": result.returncode, "started_unix": timestamp,
                   "evaluation_seconds": time.monotonic() - begin, "finished_unix": time.time()}
            summary_file = out / directory / "summary.json"
            if summary_file.is_file():
                run.update(json.loads(summary_file.read_text())["overall"])
                if "total" in run["steps"]:
                    run["steps_per_second"] = run["steps"]["total"] / run["evaluation_seconds"]
            runs.append(run)
            print(json.dumps(run), flush=True)
        (out / "exit_code").write_text(str(next((run["exit_code"] for run in runs if run["exit_code"]), 0)))
        (out / "timing.json").write_text(json.dumps({
            "policy_startup_seconds": ready - started,
            "evaluation_seconds": time.monotonic() - ready,
            "total_seconds": time.monotonic() - started,
            "runs": runs,
        }, indent=2) + "\n")
    finally:
        telemetry.terminate()
        telemetry.wait()
        telemetry_file.close()
        server.terminate()
        try:
            server.wait(10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as bundle:
        bundle.add(out, arcname="pi05-result")
    return archive.getvalue()


@app.local_entrypoint()
def main(output: str = "", task_id: int | None = None, trials: int = 1, sampling_seed: int | None = None,
         replan_steps: int = 15, binary_gripper: bool = True, renderer: str = "",
         backend: str = "mujoco", num_envs: int = 1, task_ids: str | None = None, task_workers: int = 1,
         gpu: str = "A100", max_steps: int = 530, policy_batch_size: int = 1, sweep: str = ""):
    if backend not in ("mujoco", "warp"):
        raise SystemExit("--backend must be mujoco or warp")
    if renderer and renderer not in ("mujoco", "filament"):
        raise SystemExit("--renderer must be mujoco or filament")
    if backend == "warp" and renderer:
        raise SystemExit("--backend warp uses Warp RGB rendering and takes no --renderer argument")
    if task_id is not None and task_ids is not None:
        raise SystemExit("Select either --task-id or --task-ids")
    if backend != "warp" and (task_ids is not None or task_workers > 1):
        raise SystemExit("--task-ids and --task-workers above 1 require --backend warp")
    task_id = 0 if task_id is None else task_id
    try:
        selected_tasks = [int(value) for value in task_ids.split(",")] if task_ids is not None else [task_id]
    except ValueError:
        raise SystemExit("--task-ids must be a comma-separated list of integers") from None
    if any(not 0 <= selected < 10 for selected in selected_tasks) or len(set(selected_tasks)) != len(selected_tasks):
        raise SystemExit("Select unique LIBERO Spatial task IDs from 0 to 9")
    if min(trials, num_envs, replan_steps, task_workers) < 1:
        raise SystemExit("--trials, --num-envs, --replan-steps and --task-workers must be positive")
    if not 1 <= max_steps <= 530:
        raise SystemExit("--max-steps must be between 1 and 530")
    try:
        sweep_configs = [tuple(map(int, item.split("x"))) for item in sweep.split(",")] if sweep else []
    except ValueError:
        raise SystemExit("--sweep expects comma-separated workers x worlds x batch triples such as 2x4x2") from None
    if sweep_configs and backend != "warp":
        raise SystemExit("--sweep requires --backend warp")
    for config in sweep_configs or [(task_workers, num_envs, policy_batch_size)]:
        if len(config) != 3:
            raise SystemExit("--sweep expects comma-separated workers x worlds x batch triples such as 2x4x2")
        workers, worlds, batch = config
        if min(workers, worlds, batch) < 1 or batch > min(worlds, trials):
            raise SystemExit("Batch size must be positive and fit the concurrent trial count")
        if batch > 1 and (backend != "warp" or sampling_seed is None):
            raise SystemExit("Batched inference requires --backend warp and --sampling-seed")
    if sampling_seed is not None and sampling_seed < 0:
        raise SystemExit("--sampling-seed must be nonnegative")
    if not (gripper_assets / "2f85.xml").is_file():
        raise SystemExit(f"Install the gripper with bash {ROOT / 'benchmark_scripts/setup_molmobot_gripper.sh'}")
    task_label = "-".join(map(str, selected_tasks))
    destination = Path(output or f"outputs/pi05-droid/spatial-task{task_label}/result.tar.gz").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SystemExit(f"Choose a new output path. {destination} already exists.")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    task_args = ["--task-ids", task_ids] if task_ids is not None else ["--task-id", str(task_id)]
    eval_args = [*task_args, "--task-workers", str(task_workers), "--trials", str(trials), "--replan-steps", str(replan_steps),
                 "--num-envs", str(num_envs), "--max-steps", str(max_steps),
                 "--policy-batch-size", str(policy_batch_size),
                 "--binary-gripper" if binary_gripper else "--no-binary-gripper",
                 *(["--renderer", renderer] if renderer else []),
                 *(["--sampling-seed", str(sampling_seed)] if sampling_seed is not None else [])]
    archive = evaluate.with_options(gpu=gpu).remote(revision, backend, eval_args, sweep_configs)
    with destination.open("xb") as output_file:
        output_file.write(archive)
    print(f"Saved {destination}")
    with tarfile.open(destination) as bundle:
        status = int(bundle.extractfile("pi05-result/exit_code").read())
    if status:
        raise SystemExit(status)
