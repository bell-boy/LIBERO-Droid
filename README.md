# LIBERO-Droid

LIBERO-Droid evaluates DROID-trained robot policies on LIBERO tasks. The main experiment runs the same OpenPI `pi05_droid_jointpos` checkpoint in three setups and plots success rate and task progress overall and by suite.

| Setup | Robot and cameras | Policy images and video |
|---|---|---|
| Classic · MuJoCo | LIBERO Panda gripper, camera poses, and saved initial state, with an absolute joint-position adapter | MuJoCo |
| Droid · MuJoCo | DROID Robotiq 2F-85, DROID reset pose, hybrid joint impedance controller, and calibrated exterior and wrist cameras | MuJoCo |
| Droid · Filament | Same Droid setup and MuJoCo physics | Filament RGB with shadows |

The three-arm comparison runs physics and rendering on CPU. A separate Modal app serves the policy on one shared A100. Each episode supplies two camera images, seven measured joint positions, and gripper closure. The evaluator executes 15 actions per policy query, binarizes gripper closure at 0.5, and allows up to 530 control steps. [Droid environment](libero/libero/envs/droid.py) · [Paired episode runner](benchmark_scripts/compare_episode.py)

## Run the three-way evaluation on Modal

Run these commands from the repository root. You need `uv`, Python 3, Bash, `curl`, `tar`, `zstd`, and an authenticated Modal profile. For a new profile, run `uvx modal setup`. Modal builds the simulation and policy environments remotely.

### 1. Install the gripper and deploy the apps

The setup script verifies the MolmoSpaces asset and installs it under `.libero/`. Run it before deploying so the Modal image includes the gripper.

```sh
bash benchmark_scripts/setup_molmobot_gripper.sh
uvx modal deploy benchmark_scripts/modal_compare_policy.py
uvx modal deploy benchmark_scripts/modal_compare.py::droid
uvx modal deploy benchmark_scripts/modal_compare.py::classic
uvx modal deploy benchmark_scripts/modal_compare.py::filament
```

### 2. Launch Classic and Droid with MuJoCo rendering

Choose a fresh alphanumeric run ID. Five starts for each of the ten tasks in Spatial, Object, Goal, and Long (`libero_10`) give 200 episodes per setup. Use the same run ID and trial count in both phases. Reusing a run ID resumes from saved episodes, so choose a new ID when changing the experiment.

```sh
export RUN_ID=paired001
export TRIALS=5
uv run --no-project --with modal python - <<'PY'
import os
import modal

compare = modal.Function.from_name("libero-compare-classic", "compare")
call = compare.spawn(os.environ["RUN_ID"], int(os.environ["TRIALS"]))
print(call.object_id)
PY
```

The call continues on Modal after the terminal closes. Follow the `compare` function in the Modal dashboard or stream its logs with `uvx modal app logs libero-compare-classic --follow`. The coordinator first runs one pilot per setup, then submits the remaining episodes.

### 3. Add the Filament arm

After the first phase completes, run the following command with the same `RUN_ID` and `TRIALS`. This coordinator evaluates Droid with Filament and reads the two saved MuJoCo arms to generate the three-way comparison.

```sh
uv run --no-project --with modal python - <<'PY'
import os
import modal

compare = modal.Function.from_name("libero-compare-droid-filament", "compare_filament")
call = compare.spawn(os.environ["RUN_ID"], int(os.environ["TRIALS"]))
print(call.object_id)
PY
```

Follow this phase with `uvx modal app logs libero-compare-droid-filament --follow`. Each completed episode saves a two-camera MP4 and an `episode.json` containing its task, trial, steps, success, TP, and milestone values. [Modal coordinators](benchmark_scripts/modal_compare.py)

### 4. Download the results

Once the Filament coordinator finishes, download the run from the shared Modal volume into the git-ignored `outputs/` directory.

```sh
mkdir -p outputs
uvx modal volume get libero-compare-results "/$RUN_ID" outputs/
```

| Path under `outputs/$RUN_ID/` | Contents |
|---|---|
| `three-way/comparison.png`, `three-way/summary.csv` | Final three-arm plot and aggregate scores |
| `comparison.png`, `summary.csv` | First-phase Classic and Droid comparison |
| `<arm>/<suite>/<task_id>/<trial>/` | `trajectory.mp4` and `episode.json`, with arm names `classic`, `droid`, and `droid_filament` |

To regenerate the three-arm plot and CSV locally from the downloaded episode records:

```sh
uv run --no-project --with numpy --with matplotlib python \
  benchmark_scripts/compare_plot.py "outputs/$RUN_ID" "$TRIALS" --filament
```

## How to read the scores

**Success rate (SR)** is the fraction of rollouts satisfying LIBERO’s native task predicate. **Task progress (TP)** averages episode scores on a 0–1 scale. Native success always earns 1. A failed episode retains its best recorded milestones. [Scorer](benchmark_scripts/compare_progress.py)

- **Single-object placement** earns one third for both finger pads contacting the target, one third for subsequently reaching its assigned destination, and one third for native success. A bowl that touches the correct plate but misses the native placement condition can score ⅔. Delivery to a named region uses the region geometry and a position tolerance.
- **Compound tasks** give equal weight to each target’s grasp and delivery, each applicable fixture action, and native success. For example, putting two objects in a basket has five milestones. Fixture travel can earn fractional credit.
- **Standalone drawer opening, stove activation, and plate pushing** score the best fraction of required travel achieved. Halfway travel earns 0.5.

Each arm uses the same LIBERO initial-state index and policy-noise seed at matching task indices, trials, and replanning steps. This pairs the renderer comparison even when different camera images lead to different actions. [Policy server](benchmark_scripts/pi05_serve_jointpos.py)

## Run a single task or try MuJoCo Warp

For a quick Droid run, the Spatial evaluator bundles the simulator and policy server into one Modal job. This example runs task 0 from one start with Filament and saves an archive with the video, simulator trajectory, and pickup diagnostics.

```sh
uvx modal run benchmark_scripts/modal_pi05.py --task-id 0 --trials 1 \
  --renderer filament --sampling-seed 7 --output outputs/spatial-filament.tar.gz
```

For GPU-batched physics and RGB, select Warp. This example runs four starts of each of two Spatial tasks. Warp uses its own renderer and solver, so its results form a separate experiment. [Spatial evaluator](benchmark_scripts/modal_pi05.py) · [Warp environment](libero/libero/envs/droid_warp.py)

```sh
uvx modal run benchmark_scripts/modal_pi05.py --backend warp \
  --task-ids 0,1 --task-workers 2 --trials 4 --num-envs 4 \
  --sampling-seed 7 --output outputs/spatial-warp.tar.gz
```

## Upstream and licensing

This fork builds on *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning* (Liu et al., 2023). Upstream maintains the original training workflows and dataset documentation. [LIBERO repository](https://github.com/Lifelong-Robot-Learning/LIBERO) · [Paper](https://arxiv.org/abs/2306.03310) · [Documentation](https://lifelong-robot-learning.github.io/LIBERO/)

The code uses the MIT license. The LIBERO datasets use CC BY 4.0. The gripper installer retains the asset’s upstream license. [Code license](LICENSE) · [Dataset license](https://creativecommons.org/licenses/by/4.0/legalcode)
