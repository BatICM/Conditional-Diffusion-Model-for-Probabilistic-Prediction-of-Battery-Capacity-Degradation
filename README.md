# A Conditional Diffusion Model for Battery Capacity Probabilistic Prediction

## Installation

~~~
conda create -n my_project_env python=3.9 
conda activate my_project_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
~~~

## Usage

~~~sh
lstm:
nohup python ./lstm8.py > 8.log 2>&1 &
nohup python ./lstm16.py > 16.log 2>&1 &
nohup python ./lstm24.py > 24.log 2>&1 &
nohup python ./lstm32.py > 32.log 2>&1 &

seqtoseq:
nohup python ./seq8.py > 8.log 2>&1 &
nohup python ./seq6.py > 16.log 2>&1 &
nohup python ./seq24.py > 24.log 2>&1 &
nohup python ./seq32.py > 32.log 2>&1 &

cdua:
nohup python ./cdua8.py > 8.log 2>&1 &
nohup python ./cdua.py > 16.log 2>&1 &
nohup python ./cdua24.py > 24.log 2>&1 &
nohup python ./cdua32.py > 32.log 2>&1 &
~~~

