"""Private, scale-to-zero shared pi05_droid_jointpos inference."""
import os
import subprocess
import time
from multiprocessing.connection import Client
from pathlib import Path

import modal
if modal.is_local():
    from modal_pi05 import image
else:
    image = modal.Image.debian_slim()

app = modal.App("libero-compare-jointpos-policy")
cache = modal.Volume.from_name("libero-compare-openpi-cache", create_if_missing=True)


@app.cls(image=image, gpu="A100", memory=32768, cpu=4, max_containers=1,
         scaledown_window=120, volumes={"/weights": cache}, startup_timeout=1800)
class JointPolicy:
    @modal.enter()
    def start(self):
        self.sock = "/tmp/joint-policy.sock"
        self.server = subprocess.Popen([
            "/opt/openpi/.venv/bin/python", "/opt/LIBERO-Droid/benchmark_scripts/pi05_serve_jointpos.py",
            "--local-socket", self.sock], cwd="/opt/openpi",
            env=dict(os.environ, OPENPI_DATA_HOME="/weights", XLA_PYTHON_CLIENT_MEM_FRACTION="0.80"))
        while not Path(self.sock).exists():
            if self.server.poll() is not None:
                raise RuntimeError("Policy process exited during checkpoint load")
            time.sleep(1)
        cache.commit()

    @modal.method()
    def infer(self, payload: bytes):
        with Client(self.sock, family="AF_UNIX") as conn:
            conn.send_bytes(payload)
            return conn.recv_bytes()
