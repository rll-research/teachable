[![Build Status](https://api.travis-ci.com/jonasrothfuss/ProMP.svg?branch=master)](https://travis-ci.com/jonasrothfuss/ProMP)
[![Docs](https://readthedocs.org/projects/promp/badge/?version=latest)](https://promp.readthedocs.io)

# meta-mb
The code is written in Python 3 and builds on [Tensorflow](https://www.tensorflow.org/). 
Many of the provided reinforcement learning environments require the [Mujoco](http://www.mujoco.org/) physics engine.
Overall the code was developed under consideration of modularity and computational efficiency.
Many components of the Meta-RL algorithm are parallelized either using either [MPI](https://mpi4py.readthedocs.io/en/stable/) 
or [Tensorflow](https://www.tensorflow.org/) in order to ensure efficient use of all CPU cores.

## Documentation

An API specification and explanation of the code components can be found [here](https://promp.readthedocs.io/en/latest/).
Also the documentation can be build locally by running the following commands

```
# ensure that you are in the root folder of the project
cd docs
# install the sphinx documentaiton tool dependencies
pip install requirements.txt
# build the documentaiton
make clean && make html
# now the html documentation can be found under docs/build/html/index.html
```

## Installation / Dependencies
The provided code can be either run in A) docker container provided by us or B) using python on
your local machine. The latter requires multiple installation steps in order to setup dependencies.

### A. Docker
If not installed yet, [set up](https://docs.docker.com/install/) docker on your machine.
Pull our docker container ``iclavera/meta-mb`` from docker-hub:

```
docker pull iclavera/meta-mb
```

All the necessary dependencies are already installed inside the docker container.

### B. Anaconda or Virtualenv

##### B.1. Installing MPI
Ensure that you have a working MPI implementation ([see here](https://mpi4py.readthedocs.io/en/stable/install.html) for more instructions). 

For Ubuntu you can install MPI through the package manager:

```
sudo apt-get install libopenmpi-dev
```

##### B.2. Create either venv or conda environment and activate it

###### Virtualenv
```
pip install --upgrade virtualenv
virtualenv <venv-name>
source <venv-name>/bin/activate
```

###### Anaconda 
If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#linux).
Then reate a anaconda environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
conda env create -f docker/environment.yml
```

##### B.4. Set up the Mujoco physics engine and mujoco-py
For running the majority of the provided Meta-RL environments, the Mujoco physics engine as well as a 
corresponding python wrapper are required.
For setting up [Mujoco](http://www.mujoco.org/) and [mujoco-py](https://github.com/openai/mujoco-py), 
please follow the instructions [here](https://github.com/openai/mujoco-py).

### Setting up the doodad experiment launcher with EC2 support

Install AWS commandline interface

```
sudo apt-get install awscli
```

and configure the asw cli

```
aws configure
```

Clone the doodad repository 

```
git clone git@github.com:jonasrothfuss/doodad.git
```

Install the extra package requirements for doodad
```
cd doodad && pip install -r requirements.txt
```

Configure doodad for your ec2 account. First you have to specify the following environment variables in your ~/.bashrc: 
AWS_ACCESS_KEY, AWS_ACCESS_KEY, DOODAD_S3_BUCKET

Then run
```
python scripts/setup_ec2.py
```

Set S3_BUCKET_NAME in experiment_utils/config.py to your bucket name


Finally add doodad and the meta-mb repositories to your PYHTONPATH.

## Run Benchmark
To run svg run:

```
python run_scripts_bm/svg_run.py
```

The bash script can be run with
```
bash run_scripts_bm/svg_search.sh gym_cheetah
```

To run mbmpo run:

```
python run_scripts_bm/mbmpo_run.py
```

The hyperparameters for each algoritm are the run file as a dict. 

## TODO:
Add all the existing algorithms and the run scripts
* MB-MPC (CEM and RS)
* MB-MPO
* ME-TRPO
* RL^2
* MAML
* ProMP
* E-MAML
* VPG
* PPO
* TRPO

## Acknowledgements
This repository includes environments introduced in ([Duan et al., 2016](https://arxiv.org/abs/1611.02779), 
[Finn et al., 2017](https://arxiv.org/abs/1703.03400)). This repository is based on ([Promp](https://github.com/jonasrothfuss/ProMP)).
