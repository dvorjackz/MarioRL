# MarioRL

## Setup
```
python3 -m virtualenv .env
source .env/bin/activate
pip3 install -r requirements.txt
```

## Baselines
There will be an error installing baselines using the above commands, at which point use the following line:
```
pip3 install "git+https://github.com/openai/baselines.git@master#egg=baselines-0.1.6"
```
