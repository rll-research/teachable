# Official codebase for the paper Teachable Reinforcement Learning via AdviceDistillation.

The codebase is based on the [BabyAI platform](https://github.com/mila-iqia/babyai).

## Replicating NeurIPS Results

The main branch of this repository is updated frequently. If you are looking to replicate or compare against the results from the NeurIPS paper, please use the code on the il branch.

## Installation

Requirements:
- Python 3.5+
- OpenAI Gym
- NumPy
- PyTorch 0.4.1+
- blosc
- Mujoco
- osmesa
- patchelf
- absl-py
- libosmesa-6

Start by manually installing PyTorch. See the [PyTorch website](http://pytorch.org/)
for installation instructions specific to your platform.


### Installation using Conda (Alternative Method)

If you are using conda, you can create a `babyai` environment with all the dependencies by running:

```
git clone https://github.com/mila-iqia/babyai.git
cd babyai
conda env create -f environment.yaml
source activate babyai
```

After that, execute the following commands to setup the environment.

```
cd ..
git clone https://github.com/maximecb/gym-minigrid.git
cd gym-minigrid
pip install --editable .
```

The last command installs the repository in editable mode. Move back to the `babyai` repository and install that in editable mode as well.

```
cd ../babyai
pip install --editable .
```

Do the same for the `d4rl` directory

```
cd d4rl
pip install --editable .
```

## Usage

To train a model, use the `train_model.py` script.
To evaluate a trained model, train a model through bootstrapping, or finetune on a new level, use the `test_generalization.py` script.
