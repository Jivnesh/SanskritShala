# LemmaTag: A Sanskrit Morphological Tagger


## Overview

There are two main ideas:

1. Since part-of-speech tagging and lemmatization are related tasks, sharing the initial layers of the network is mutually beneficial. This results in higher accuracy and requires less training time.
2. The lemmatizer can further improve its accuracy by looking at the tagger's predictions, i.e., taking the output of the tagger as an additional lemmatizer input.


## Getting Started

### Requirements

The code uses Python 3.5+ running TensorFlow (tested working with TF 1.12).
Install the python packages in `requirements.txt` if you don't have them already.

```bash
pip install -r ./requirements.txt
```

### Training and Testing

To start training on a sample dataset with default parameters, run

```bash
python lemmatag.py
```


## Web-based tool
You can interact with SaCTI on our SanskritShala's web-based platform: [`Link`](https://cnerg.iitkgp.ac.in/tramp/)

## License
This project is licensed under the terms of the `Apache license 2.0`.


## Acknowledgements
We have built our model on the top of [`LemmaTag`](https://github.com/hyperparticle/LemmaTag).