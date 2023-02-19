#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate TransLIST
cd ~/SanShala-Models/TransLIST/V0
python interactive_module.py --sentence="$1"