<div align="center">
<img src="https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images/libero_logo.png" width="360">


<p align="center">
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/actions">
<img alt="Tests Passing" src="https://github.com/anuraghazra/github-readme-stats/workflows/Test/badge.svg" />
</a>
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/graphs/contributors">
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/Lifelong-Robot-Learning/LIBERO" />
</a>
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/issues">
<img alt="Issues" src="https://img.shields.io/github/issues/Lifelong-Robot-Learning/LIBERO?color=0088ff" />

## **Benchmarking Knowledge Transfer for Lifelong Robot Learning**

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone

[[Website]](https://libero-project.github.io)
[[Paper]](https://arxiv.org/pdf/2306.03310.pdf)
[[Docs]](https://lifelong-robot-learning.github.io/LIBERO/)
______________________________________________________________________
![pull_figure](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images//fig1.png)
</div>

**LIBERO** is designed for studying knowledge transfer in multitask and lifelong robot learning problems. Successfully resolving these problems require both declarative knowledge about objects/spatial relationships and procedural knowledge about motion/behaviors. **LIBERO** provides:
- a procedural generation pipeline that could in principle generate an infinite number of manipulation tasks.
- 130 tasks grouped into four task suites: **LIBERO-Spatial**, **LIBERO-Object**, **LIBERO-Goal**, and **LIBERO-100**. The first three task suites have controlled distribution shifts, meaning that they require the transfer of a specific type of knowledge. In contrast, **LIBERO-100** consists of 100 manipulation tasks that require the transfer of entangled knowledge. **LIBERO-100** is further splitted into **LIBERO-90** for pretraining a policy and **LIBERO-10** for testing the agent's downstream lifelong learning performance.
- five research topics.
- three visuomotor policy network architectures.
- three lifelong learning algorithms with the sequential finetuning and multitask learning baselines.

---


# Contents

- [Installation](#Installation)
- [Datasets](#Dataset)
- [Getting Started](#Getting-Started)
  - [Task](#Task)
  - [Training](#Training)
  - [Evaluation](#Evaluation)
- [Citation](#Citation)
- [License](#License)


# Installtion
Please run the following commands in the given order to install the dependency for **LIBERO**.
```
conda create -n libero python=3.8.13
conda activate libero
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

Then install the `libero` package:
```
pip install -e .
```

# Datasets
We provide high-quality human teleoperation demonstrations for the four task suites in **LIBERO**. To download the demonstration dataset, run:
```python
python benchmark_scripts/download_libero_datasets.py
```
By default, the dataset will be stored under the ```LIBERO``` folder and all four datasets will be downloaded. To download a specific dataset, use
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where ```DATASET``` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal`.

**NEW!!!**

Alternatively, you can download the dataset from HuggingFace by using:
```python
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

This option can also be combined with the specific dataset selection:
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET --use-huggingface
```

The datasets hosted on HuggingFace are available at [here](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).


# Getting Started

For a detailed walk-through, please either refer to the documentation or the notebook examples provided under the `notebooks` folder. In the following, we provide example scripts for retrieving a task, training and evaluation.

## Task

The following is a minimal example of retrieving a specific task from a specific task suite.
```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
for step in range(10):
    obs, reward, done, info = env.step(dummy_action)
env.close()
```
Currently, we only support sparse reward function (i.e., the agent receives `+1` when the task is finished). As sparse-reward RL is extremely hard to learn, currently we mainly focus on lifelong imitation learning.

## Training
To start a lifelong learning experiment, please choose:
- `BENCHMARK` from `[LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_90, LIBERO_10]`
- `POLICY` from `[bc_rnn_policy, bc_transformer_policy, bc_vilt_policy]`
- `ALGO` from `[base, er, ewc, packnet, multitask]`

then run the following:

```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python libero/lifelong/main.py seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=POLICY \
                               lifelong=ALGO
```
Please see the documentation for the details of reproducing the study results.

## Evaluation

All task suites default to robosuite's model of DROID's Robotiq 2F-85. When loading a bundled initial state, the simulator restores the saved arm and scene state and initializes the Robotiq in its reset pose. [DROID hardware docs](https://droid-dataset.github.io/droid/hardware-setup/shopping-list)

`robot0_gripper_qpos` reports the six Robotiq joint positions. The built-in evaluator uses `robot0_panda_gripper_qpos`, a two-value estimate from the Robotiq's pad opening, for existing LIBERO policies. Pass `gripper_types="default"` to `OffScreenRenderEnv` to load the original Panda gripper for replay.

The built-in evaluator uses Filament with cast shadows for policy RGB
observations and rollout videos. Install MuJoCo 4 in a separate environment
because robosuite 1.4 uses the MuJoCo 3 physics API. Run these commands from the
repository root.

```shell
uv venv --python 3.12 outputs/.filament-venv
uv pip install --python outputs/.filament-venv/bin/python "mujoco==4.0.1"
export LIBERO_FILAMENT_PYTHON="$PWD/outputs/.filament-venv/bin/python"
```

Run evaluation from your regular LIBERO environment. For a custom evaluator,
pass `renderer="filament"` to `OffScreenRenderEnv`. Select `--renderer mujoco`
for the classic renderer in the standalone evaluator, or `eval.renderer=mujoco`
for evaluation during training. Select the classic renderer when requesting
depth or segmentation.

By default the policies will be evaluated on the fly during training. If you have limited computing resource of GPUs, we offer an evaluation script for you to evaluate models separately.

```shell
python libero/lifelong/evaluate.py --benchmark BENCHMARK_NAME \
                                   --task_id TASK_ID \ 
                                   --algo ALGO_NAME \
                                   --policy POLICY_NAME \
                                   --seed SEED \
                                   --ep EPOCH \
                                   --load_task LOAD_TASK \
                                   --device_id CUDA_ID
```

### DROID setup

`libero.libero.envs.droid.DroidEnv` runs any LIBERO task in the DROID robot
setup, so a DROID-trained policy can use it without adapters:

- Filament rendering with shadows and the original scene lighting.
- The MolmoSpaces DROID Robotiq 2F-85, driven by its tendon actuator.
- The DROID reset arm pose. The scene comes from LIBERO's saved initial state.
- DROID's hybrid joint impedance controller with a 66 ms control period. Each
  step moves the arm at most 0.2 rad per joint toward the target.
- A DROID exterior camera placed relative to the robot base. Its pose is the
  median of the calibrated left-side DROID cameras whose view holds the robot
  shoulder, the reset gripper and the table. It uses the median DROID ZED 2
  calibration, 320 × 180 with a 101° × 69° field of view.
- The calibrated left lens of DROID ZED Mini 19824535 as the wrist camera.

Observations hold upright `exterior_image` and `wrist_image` arrays, the seven
measured `joint_position` values, and the measured `gripper_position`
(0 open, 1 closed). Actions are seven absolute joint targets in radians and a
gripper closure in [0, 1].

```python
from libero.libero.envs.droid import DroidEnv

env = DroidEnv(bddl_file)
obs = env.reset(init_state)
obs, success = env.step(action)
```

Install the gripper asset into `.libero/` once.

```shell
bash benchmark_scripts/setup_molmobot_gripper.sh
```

### π0.5 DROID evaluation on Modal

`benchmark_scripts/modal_pi05.py` evaluates `pi05_droid_jointpos` in the DROID
setup on LIBERO Spatial tasks. The policy server is
`benchmark_scripts/pi05_serve_jointpos.py`, and the evaluation loop is
`benchmark_scripts/pi05_droid_eval.py`. The loop resizes both images to
224 × 224 with padding, executes all 15 predicted actions before the next
query, and closes the hand when the predicted closure exceeds 0.5. Episodes
last up to 530 steps (35 s). Use an authenticated Modal profile.

```shell
uvx --from modal modal run benchmark_scripts/modal_pi05.py --task-id 0
```

The result is saved to `outputs/pi05-droid/spatial-task<ID>/result.tar.gz`
unless `--output` names another path. It contains the run manifest, a summary,
and one MP4, trajectory, `episode.json`, and `pick.json` per episode under
`pi05-result/evaluation/`. Use `--trials N` for the first N initial states and
`--sampling-seed` for repeatable policy sampling. `--replan-steps`,
`--no-binary-gripper`, and `--renderer mujoco` change the loop and renderer.

With classic MuJoCo, `--num-envs` sets the number of worker processes. Each
worker owns its simulation, renderer, and WebSocket connection. As workers
finish, they pick up the next trial. All workers share the OpenPI server,
which handles one inference request at a time.

```shell
uvx --from modal modal run benchmark_scripts/modal_pi05.py \
  --task-id 0 --backend mujoco --trials 16 --num-envs 4 \
  --output outputs/pi05-droid/spatial-task0-processes/result.tar.gz
```

With a policy server already listening, the standalone
`benchmark_scripts/pi05_droid_eval.py` accepts the same flags. Pass an output
directory as `--output`. Use `--sampling-seed` to keep policy sampling tied to
the trial and step when concurrent requests arrive in a different order.

To run trials of the same task in parallel on NVIDIA GPUs, select the MuJoCo
Warp backend and set the number of concurrent environments. This runs 16
initial states in groups of 8 and writes the same per-episode artifacts.

```shell
uvx --from modal modal run benchmark_scripts/modal_pi05.py \
  --task-id 0 --backend warp --trials 16 --num-envs 8 \
  --output outputs/pi05-droid/spatial-task0-warp/result.tar.gz
```

Warp batches physics and both RGB cameras. By default the OpenPI server
processes one observation at a time. Each trial keeps its own action chunk, sampling
seed, and termination condition. Warp RGB and solver behavior can differ
from Filament and classic MuJoCo. The manifest records the backend and batch
size. [MuJoCo Warp documentation](https://mujoco.readthedocs.io/en/3.17.0/mjwarp/index.html)

To run different Warp tasks concurrently, pass a comma-separated list to
`--task-ids` and set `--task-workers`. Each worker process owns one task model
at a time and batches up to `--num-envs` trials of that task on the GPU.
`--trials` applies to each task. For example, this runs two tasks concurrently,
each with four episodes in its Warp batch.

```shell
uvx --from modal modal run benchmark_scripts/modal_pi05.py \
  --backend warp --task-ids 0,1 --task-workers 2 --trials 4 --num-envs 4 \
  --output outputs/pi05-droid/spatial-tasks0-1-warp/result.tar.gz
```

Workers share the Modal GPU and the OpenPI server, so GPU memory use grows
with both worker count and batch size. The archive contains per-task episode
directories, per-task metrics, and an overall success rate.

For a throughput comparison, choose the Modal GPU with `--gpu` and pass
`--sweep` as comma-separated `workers x worlds x policy batch size` triples.
The sweep loads one policy server and saves timings and GPU utilization
alongside each configuration's results. Set `--sampling-seed` for policy
batches larger than one. `--max-steps` can shorten the trials for a smoke
test. Omit it for the full episode limit.

For local Warp evaluation, use a separate copy of the DROID evaluation
environment with `mujoco==3.17.0`, `mujoco-warp==3.17.0`, and
`warp-lang==1.19.0`. With the OpenPI server listening, run
`benchmark_scripts/pi05_droid_eval.py` with the same task, backend, trial, and
environment flags. Pass an output directory that does not exist yet.

A pickup requires both finger pads to contact the target bowl while its
center is more than 2 cm above its starting height, clear of all other bodies
except the gripper, for 10 consecutive frames. `pick.json` records the first
qualifying frame and longest hold. The standard LIBERO report scores the full
placement.

With one initial state per task, Spatial tasks 0 to 4 picked up the target
bowl in 4 of 5 episodes and completed 2. An earlier three-quarter exterior
camera (57° field of view, 256 × 256) picked up 3 and completed 1.

# Citation
If you find **LIBERO** to be useful in your own research, please consider citing our paper:

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

# License
| Component        | License                                                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Codebase         | [MIT License](LICENSE)                                                                                                                      |
| Datasets         | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode)                 |
