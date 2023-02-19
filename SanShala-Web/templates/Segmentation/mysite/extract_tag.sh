#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate LemmaTag
cd /home/jivneshs/SanShala-Models/LemmaTag/Morph_Scrapper/
python final_scrap_.py
cd /home/jivneshs/SanShala-Web/templates/POS-tagging

# python /home/jivnesh/SanShala-Models/LemmaTag/lemmatag.py
