#!/bin/bash

PATH_TO_ENVS="$1"
st='hack-ngram'

cd sktWS
cp hackathon/* .
cd ..

sed -i "310 s/^setting =.*/setting = '${st}'/" $PATH_TO_ENVS/tlat0/lib/python3.7/site-packages/fastNLP/core/dataset.py

cp Hackathon_data/embeds/* .



