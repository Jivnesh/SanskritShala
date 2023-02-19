#!/usr/bin/env bash
# source /home/guest/anaconda3/envs/SaCTI/bin/activate SaCTI
eval "$(conda shell.bash hook)"
conda activate SaCTI
cd ~/SanShala-Models/SaCTI
python main.py --model_path='models1'