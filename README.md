# MarioRL

## Setup
*Important*: must use Python version < 3.8, preferrably Python-3.7.6
```
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```
There will be an error installing baselines using the last command, at which point use the following lines:
```
pip3 install "git+https://github.com/openai/baselines.git@master#egg=baselines-0.1.6"
pip3 install -r requirements.txt
```
## Potential Matplotlib Error
```
python -m pip install -U matplotlib==3.2.0rc1
```
## Potential PMIX Error
Solve by using the following line:
```
export PMIX_MCA_gds=hash
```