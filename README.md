# MarioRL

## Setup

```
python3 -m virtualenv env
source env/bin/activate
pip3 install -r requirements.txt
```

There will be an error installing baselines using the last command, at which point use the following lines:

```
pip3 install "git+https://github.com/openai/baselines.git@master#egg=baselines-0.1.6"
pip3 install -r requirements.txt
```

Potential PMIX error can be fixed with command:

```
export PMIX_MCA_gds=hash
```

## Run

Train model with DQN algorithm to play Super Mario Bros, placing desired hyperparameters in `hyperparameters.py`.

### Double DQN with Stable Baselines

Train model using [Stable Baselines](https://github.com/hill-a/stable-baselines) DQN implementations.

```
python3 train.py
```

Evaluate model:

```
python3 eval.py
```

### QMap DQN

Conduct tests with [Goal-based exploration](https://arxiv.org/pdf/1807.02078.pdf). QMap implementation taken from [Fabien Pardo's QMap project](https://github.com/fabiopardo/qmap) which uses [OpenAI Baselines](https://github.com/openai/baselines).

```
python3 train.py --qmap
```

Analyze training and performance by plotting data recorded in CSV files:

```
python3 -m qmap.qmap.utils.plot
```
