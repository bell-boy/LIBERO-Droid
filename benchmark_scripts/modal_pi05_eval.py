"""Adapt the companion π0.5 evaluator to LIBERO-Droid's renderer and Robotiq."""

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import yaml


def run(libero_root, output, libero_revision, openpi_revision, port):
    from pi05_libero import adapters, evaluate

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

    class DroidRenderEnv(envs.OffScreenRenderEnv):
        def __init__(self, **kwargs):
            kwargs.update(gripper_types="Robotiq85Gripper", renderer="filament")
            super().__init__(**kwargs)

    policy_observation = adapters.policy_observation

    def droid_observation(obs, *args, **kwargs):
        policy_input = dict(obs, robot0_gripper_qpos=obs["robot0_panda_gripper_qpos"])
        return policy_observation(policy_input, *args, **kwargs)

    revisions = {libero_root: libero_revision, config_dir.parent / "openpi": openpi_revision}
    argv = [
        "pi05-libero", "--host", "127.0.0.1", "--port", str(port),
        "--suites", "libero_spatial", "--task-ids", "0", "--trials", "3",
        "--camera-rig", "droid_oblique", "--replan-steps", "15",
        "--libero-root", str(libero_root), "--output", str(output), "--fail-fast",
    ]
    try:
        with (
            patch("sys.argv", argv),
            patch.object(evaluate, "ROOT", config_dir.parent),
            patch.object(evaluate, "revision", revisions.__getitem__),
            patch.object(envs, "OffScreenRenderEnv", DroidRenderEnv),
            patch.object(adapters, "policy_observation", droid_observation),
        ):
            evaluate.main()
    finally:
        manifest = output / "manifest.json"
        if manifest.is_file():
            metadata = json.loads(manifest.read_text())
            metadata["simulation_renderer"] = "filament"
            metadata["args"]["gripper"] = "Robotiq85Gripper"
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--libero-revision", required=True)
    parser.add_argument("--openpi-revision", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.libero_root.resolve(), args.output, args.libero_revision, args.openpi_revision, args.port)
