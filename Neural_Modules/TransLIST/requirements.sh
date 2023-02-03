#!/bin/bash

PATH_TO_ENVS="$1"

cp fastnlp-copy/core/tester.py $PATH_TO_ENVS/tlat0/lib/python3.7/site-packages/fastNLP/core/

cp fastnlp-copy/core/dataset.py $PATH_TO_ENVS/tlat0/lib/python3.7/site-packages/fastNLP/core/

cp fastnlp-copy/core/metrics.py $PATH_TO_ENVS/tlat0/lib/python3.7/site-packages/fastNLP/core/

cp fastnlp-copy/modules/decoder/crf.py $PATH_TO_ENVS/tlat0/lib/python3.7/site-packages/fastNLP/modules/decoder/


