"""Run the three-episode π0.5 DROID evaluation on Modal."""
import io
import os
from pathlib import Path
import socket
import subprocess
import tarfile
import tempfile
import time

import modal

app = modal.App("libero-droid-pi05-spatial-task0-3qv")
ROOT = Path(__file__).resolve().parents[1]
OPENPI_REV = "215abfb217dbac7d5f1273282331b9b1866c0479"
LEROBOT_REV = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
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
        "uv pip install --python /opt/sim/bin/python 'numpy<2' 'robosuite==1.4.0' 'mujoco==3.2.6' "
        "'bddl==1.0.1' future 'gym==0.25.2' easydict matplotlib 'opencv-python<4.12' "
        "'imageio[ffmpeg]' 'openpi-client==0.1.2' pyyaml scipy",
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
    .run_commands("uv pip install --python /opt/sim/bin/python termcolor")
)

if modal.is_local():
    companion = Path(os.environ.get("PI05_LIBERO_ROOT", ROOT.parent / "pi05-libero")).resolve()
    evaluator = companion / "src/pi05_libero/evaluate.py"
    server_script = companion / "scripts/serve_jointpos.py"
    if not evaluator.is_file() or not server_script.is_file():
        raise SystemExit(f"Set PI05_LIBERO_ROOT to a pi05-libero checkout. Checked {companion}")
    image = (
        image.add_local_dir(ROOT / "libero", "/opt/LIBERO-Droid/libero", ignore=["**/__pycache__/**", "**/.DS_Store"])
        .add_local_dir(companion / "src/pi05_libero", "/opt/eval/src/pi05_libero", ignore=["**/__pycache__/**"])
        .add_local_file(server_script, "/opt/eval/scripts/serve_jointpos.py")
        .add_local_file(ROOT / "benchmark_scripts/modal_pi05_eval.py", "/opt/eval/run.py")
    )


@app.function(image=image, gpu="A100", cpu=8, memory=32768, timeout=2400)
def evaluate(libero_revision: str):
    with tempfile.TemporaryDirectory(prefix="libero-droid-") as workdir:
        return run_evaluation(Path(workdir), libero_revision)


def run_evaluation(workdir: Path, libero_revision: str):
    out = workdir / "pi05-result"
    out.mkdir()
    runtime = workdir / "xdg"
    runtime.mkdir(mode=0o700)
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        PYTHONPATH="/opt/eval/src:/opt/LIBERO-Droid",
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
    server = subprocess.Popen(
        ["/opt/openpi/.venv/bin/python", "/opt/eval/scripts/serve_jointpos.py", "--port", str(port)],
        env=env, cwd="/opt/openpi",
    )
    try:
        for _ in range(1200):
            if server.poll() is not None:
                raise RuntimeError(f"Policy server exited with status {server.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("Policy server did not start within 20 minutes")
        result = subprocess.run(
            [
                "/opt/sim/bin/python", "/opt/eval/run.py",
                "--libero-root", "/opt/LIBERO-Droid",
                "--libero-revision", libero_revision,
                "--openpi-revision", OPENPI_REV,
                "--port", str(port),
                "--output", str(out / "evaluation"),
            ],
            env=env, cwd="/opt/eval",
        )
        (out / "exit_code").write_text(str(result.returncode))
    finally:
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
def main(output: str = "outputs/modal-pi05-spatial-task0-3qv/result.tar.gz"):
    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SystemExit(f"Choose a new output path. {destination} already exists.")
    revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    archive = evaluate.remote(revision)
    with destination.open("xb") as output_file:
        output_file.write(archive)
    print(f"Saved {destination}")
    with tarfile.open(destination) as bundle:
        status = int(bundle.extractfile("pi05-result/exit_code").read())
    if status:
        raise SystemExit(status)
