"""Deploy both worker apps and spawn the headless comparison coordinator."""
import json
import os
from itertools import product
from multiprocessing.connection import Listener
from pathlib import Path
import subprocess
import tempfile
import threading

import modal

classic = modal.App("libero-compare-classic")
droid = modal.App("libero-compare-droid")
filament = modal.App("libero-compare-droid-filament")
volume = modal.Volume.from_name("libero-compare-results", create_if_missing=True)
scripts = Path("/opt/LIBERO-Droid/benchmark_scripts")
if modal.is_local():
    from modal_pi05 import image
    for filename in ("compare_progress.py", "compare_plot.py", "compare_episode.py"):
        image = image.add_local_file(Path(__file__).with_name(filename), str(scripts / filename))
else:
    image = modal.Image.debian_slim()
options = dict(image=image, volumes={"/results": volume}, cpu=2, memory=4096,
               max_containers=8, timeout=3600, retries=1)
suites = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def rollout(run, variant, suite, task, trial):
    dest = Path("/results") / run / variant / suite / str(task) / str(trial)
    if (dest/"episode.json").is_file() and (dest/"trajectory.mp4").stat().st_size:
        return json.loads((dest/"episode.json").read_text())
    policy = modal.Cls.from_name("libero-compare-jointpos-policy", "JointPolicy")()
    with tempfile.TemporaryDirectory() as tmp:
        sock = str(Path(tmp)/"broker")
        errors = []
        with Listener(sock, family="AF_UNIX") as listener:
            def serve():
                try:
                    with listener.accept() as conn:
                        while True:
                            try:
                                payload = conn.recv_bytes()
                            except EOFError:
                                return
                            conn.send_bytes(policy.infer.remote(payload))
                except Exception as exc:
                    errors.append(exc)
            threading.Thread(target=serve, daemon=True).start()
            env = dict(os.environ, PYTHONPATH="/opt/LIBERO-Droid", LIBERO_CONFIG_PATH=tmp,
                       MUJOCO_GL="osmesa", PYOPENGL_PLATFORM="osmesa", LP_NUM_THREADS="2",
                       OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", MPLCONFIGDIR=tmp)
            if variant == "droid_filament":
                env.update(LIBERO_FILAMENT_PYTHON="/opt/filament/bin/python", XDG_RUNTIME_DIR=tmp)
            result = subprocess.run(["/opt/sim/bin/python", str(scripts/"compare_episode.py"),
                                     variant, suite, str(task), str(trial), str(dest), sock], env=env)
            if errors:
                raise errors[0]
            result.check_returncode()
        volume.commit()
        return json.loads((dest/"episode.json").read_text())


@classic.function(**options)
def classic_episode(run, suite, task, trial):
    return rollout(run, "classic", suite, task, trial)


@droid.function(**options)
def droid_episode(run, suite, task, trial):
    return rollout(run, "droid", suite, task, trial)


@classic.function(image=image, volumes={"/results":volume}, timeout=8*3600)
def compare(run, trials=5):
    if not run.isalnum() or trials < 1:
        raise ValueError("Use an alphanumeric run ID and positive trial count")
    workers = (classic_episode, modal.Function.from_name("libero-compare-droid", "droid_episode"))
    # Verify each arm can finish a real episode before expanding the fanout.
    for call in [w.spawn(run, suites[0], 0, 0) for w in workers]:
        print("Pilot:", call.get(), flush=True)
    jobs = [(run,s,t,i) for i,s,t in product(range(trials),suites,range(10)) if (s,t,i) != (suites[0],0,0)]
    from concurrent.futures import ThreadPoolExecutor
    def collect(worker):
        for row in worker.starmap(jobs, order_outputs=False):
            print(row["variant"], row["suite"], row["task_id"], row["trial"], row["tp"], flush=True)
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(collect, workers))
    volume.reload()
    subprocess.run(["/opt/sim/bin/python", str(scripts/"compare_plot.py"), "/results/"+run, str(trials)], check=True)
    volume.commit()
    return run

@filament.function(**{**options, "max_containers": 24})
def filament_episode(run, suite, task, trial):
    return rollout(run, "droid_filament", suite, task, trial)


@filament.function(image=image, volumes={"/results": volume}, timeout=8*3600)
def compare_filament(run, trials=5):
    pilot = (run, suites[0], 0, 0)
    print("Filament pilot", filament_episode.remote(*pilot), flush=True)
    jobs = [(run, s, t, i) for i, s, t in product(range(trials), suites, range(10))]
    for row in filament_episode.starmap([job for job in jobs if job != pilot], order_outputs=False):
        print(row["suite"], row["task_id"], row["trial"], row["tp"], flush=True)
    volume.reload()
    subprocess.run(["/opt/sim/bin/python", str(scripts/"compare_plot.py"), "/results/"+run,
                    str(trials), "--filament"], check=True)
    volume.commit()
