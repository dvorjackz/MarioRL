# MarioRL

## Setup
```
python3 -m virtualenv .env
source .env/bin/activate
pip3 install -r requirements.txt
```
There will be an error installing baselines using the last command, at which point use the following lines:
```
pip3 install "git+https://github.com/openai/baselines.git@master#egg=baselines-0.1.6"
pip3 install -r requirements.txt
```