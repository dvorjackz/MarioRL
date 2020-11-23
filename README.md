# MarioRL

Using [Gym Super Mario Bros](https://pypi.org/project/gym-super-mario-bros/) as the environment and [Stable Baselines](https://github.com/hill-a/stable-baselines), a fork of OpenAI's popular [Baselines](https://github.com/openai/baselines) reinforcement learning library, we apply concepts highlighted in recent influential papers in the RL space to traing an agent to beat Super Mario Bros for NES as quickly as possible. 

The final report on our findings is included in the repo as paper.pdf. 

## Setup
*Important*: must use Python version < 3.8, preferrably Python-3.7.6
```
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

## Potential PMIX Error Fix
Solve by using the following line:
```
export PMIX_MCA_gds=hash
```
