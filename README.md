# MarioRL

## Setup
*Important*: must use Python version < 3.8, preferrably Python-3.7.6
```
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
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
