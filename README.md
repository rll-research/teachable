test

# Official codebase for the paper Teachable Reinforcement Learning via AdviceDistillation.

The codebase contains elements from the [BabyAI platform](https://github.com/mila-iqia/babyai), the [meta-mb](https://github.com/iclavera/meta-mb) codebase, the [d4rl codebase](https://github.com/rail-berkeley/d4rl), and [denisyarats/pytorch_sac codebase](https://github.com/denisyarats/pytorch_sac).

## Replicating NeurIPS Results

The main branch of this repository is updated frequently. If you are looking to replicate or compare against the results from the NeurIPS paper, please use the code on the neurips_21 branch.


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

Example: `python scripts/train_model.py  --prefix EXP_NAME --collect_with_rl_policy --level 2 --frames_per_proc 200 --num_envs 20 --rl_teacher OffsetWaypoint --env point_mass`


## Useful Commands

### RL training
```
python scripts/train_model.py --prefix EXP_NAME --collect_with_rl_policy --train_level --frames_per_proc 200 --num_envs 20 --rl_teacher OffsetWaypoint --env point_mass --n_advice 1000000 --seed 1 --reward_type oracle_dist
```

### Distillation to no advice 
```
python scripts/train_model.py --buffer_capacity 1000000 --frames_per_proc 40 --num_envs 20 --collect_policy PATH_TO_RL_TRAINED_POLICY --distill_teacher none --collect_teacher OffsetWaypoint --env point_mass --level 2 --n_advice 300000 --seed 1 --prefix EXP_NAME
```

### Distilling from one advice form to another
```
python scripts/train_model.py --prefix EXP_NAME --buffer_capacity 1000000 --train_level --frames_per_proc 200 --num_envs 20 --distillation_steps 15 --collect_policy PATH_TO_RL_TRAINED_POLICY --distill_teacher OTHER_TEACHER_NAME --collect_teacher OffsetWaypoint --env point_mass --n_advice 1000000 --seed 1
```

```
--prefix
persisted_models_distill/original_distillation
--collect_policy
logs/persisted_models_distill/claire
--rl_policy
logs/persisted_models_distill/claire
--level
2
--frames_per_proc
200
--num_envs
100
--collect_teacher Direction
--distill_teacher OffsetWaypoint
--env point_mass
--save_option all
--n_itr 10000
```

## Running experiments faster

* Use the easiest form of advice (for PointMaze, try OffsetWaypoint or Direction advice)
* Use dense rewards for RL training. This should be enabled by default. In the experiments spreadsheet we default to using a semi-sparse reward which improves advice efficiency but decreases sample efficiency.
* Use a small maze size.
* Consider training on a single maze rather than a procedurally generated set of mazes (though I'd expect this change would mean advice doesn't get learned as well/will be less likely to generalize to other mazes)
* If you use babyAI, there are even smaller mazes than those used in the paper. These could be faster.
* Use a GPU, if you have one available.
* You might see small efficiency gains if you sweep over learning rate and frames_per_proc

### Experiments we ran

#### Distillation 
```
--prefix persisted_models_distill/original_distillation_cont
--collect_policy logs/persisted_models_distill/claire
--collect_teacher Direction
--rl_policy logs/persisted_models_distill/claire
--level 2
--frames_per_proc 200
--num_envs 20

--distill_teacher OffsetWaypoint
--env point_mass
--save_option all
--n_itr
10000
```

This runs with: 
rl_agent will be **None**
distilling_agent is a *newly initialized* PPO Agent
relabel_policy is **None** 

Collect Teacher is Direction -> 
collect_policy is the Grounded Agent **with weights**


```
--collect_with_rl_policy
--rl_policy logs/persisted_models_distill/grounded
--rl_teacher Direction
--use_input_converter
--env point_mass
--level 2
--frames_per_proc 200
--num_envs 20
--save_option
all
--n_itr
10000
```


