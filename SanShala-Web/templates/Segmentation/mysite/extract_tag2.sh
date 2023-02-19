#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate LemmaTag
cd /home/jivneshs/SanShala-Models/LemmaTag
python lemmatag.py
cd /home/jivneshs/SanShala-Web/templates/POS-tagging