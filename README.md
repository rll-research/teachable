# Official codebase for the paper Teachable Reinforcement Learning via AdviceDistillation.

The codebase contains elements from the [BabyAI platform](https://github.com/mila-iqia/babyai), the [meta-mb](https://github.com/iclavera/meta-mb) codebase, the [d4rl codebase](https://github.com/rail-berkeley/d4rl), and [denisyarats/pytorch_sac codebase](https://github.com/denisyarats/pytorch_sac).

## Replicating NeurIPS Results

The main branch of this repository is updated frequently, so use that if you want the newest features. If you are looking to replicate or compare against the results from the NeurIPS paper, please use this branch. Exact commands used in the paper can be found [here](https://docs.google.com/spreadsheets/d/1bFcwLeM-eT-dpISRGRHB25fxBtYHNSkN3fK7qkoMe6U/edit?usp=sharing).


## Installation

First, clone the repo.

Run the following commands:

```
git clone https://github.com/AliengirlLiv/teachable.git
conda create --name teachable_rl python=3.7
cd teachable
conda activate teachable_rl
sudo apt install libglew-dev libsm6 libxrender1 libosmesa6-dev patchelf
pip install -r reqs.txt

```
Add the project directory `MY/PATH/TO/PROJECT/teachable` to your PYTHONPATH.

If you want to use the Point Maze or Ant MAze environments, [install Mujoco](https://github.com/openai/mujoco-py).

Add these to your `~/.bashrc` file (note version numbers may have changed):
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
```


Then run:
```
cd envs/d4rl
pip install -v --editable .
```

The most common installation error is mujoco not installing correctly. To check if this is the problem, open a python interpreter and run `import mujoco_py`.


## Usage

To train a model, run the `train_model.py` script from the `teachable` directory.

Example: `python scripts/train_model.py  --prefix EXP_NAME --collect_with_rl_policy --train_level --frames_per_proc 200 --num_envs 20 --rl_teacher OffsetWaypoint --env point_mass`

This script trains on the procedurally-generated set of mazes used for training in the NeurIPS paper. To train on mazes of a different size, or on a static maze, replace `--train_level` with a particular level (e.g. `--level 2`).
