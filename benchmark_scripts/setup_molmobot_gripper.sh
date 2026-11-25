#!/usr/bin/env bash
# Install the public Ai2 Robotiq 2F-85 asset and its upstream license.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/.libero/robotiq_2f85_v4"
[[ -f "$dest/2f85.xml" ]] && exit 0
mkdir -p "$root/.libero"
work=$(mktemp -d "$root/.libero/robotiq-install.XXXXXX")
trap 'rm -r "$work"' EXIT
shard=${1:-$work/shard.tar}
if [[ $# -eq 0 ]]; then
  curl -fL --retry 3 'https://huggingface.co/datasets/allenai/molmospaces/resolve/5f802a46816ffc3520e59b5ef49fa8dafee8eeca/mujoco/robots/franka_droid/20260127/shards/00000.tar' -o "$shard"
fi
python3 - "$shard" <<'PY'
import hashlib
from pathlib import Path
import sys

expected = "9570245a1cd2bc820051a5faeccdb22f531d655508451a0b80bda12d2de49d3e"
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != expected:
    raise SystemExit("Unexpected SHA-256 for the Ai2 Franka asset shard")
PY
tar -xOf "$shard" franka_droid.tar.zst | zstd -dc | tar -xf - -C "$work" robotiq_2f85_v4
mv "$work/robotiq_2f85_v4" "$dest"
