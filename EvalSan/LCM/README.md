# LCM: Our proposed pretraining

## Requirements

* Python 3.7 
* Pytorch 1.1.0 
* Cuda 9.0 
* Gensim 3.8.1

We assume that you have installed conda beforehand. 

```
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
pip install gensim==3.8.1
```
## Data
* Pretrained FastText embeddings for Sanskrit can be obtained from [here](https://drive.google.com/drive/folders/1JJMBjUZdqUY7WLYefBbA2zKaMHH3Mm18?usp=sharing). Make sure that `.vec` file is placed at approprite position.
* For Multilingual experiments, we use [UD treebanks](https://universaldependencies.org/) and [Pretrained FastText embeddings](https://fasttext.cc/docs/en/crawl-vectors.html)


## How to train model
If you want to run complete model pipeline: (1) Pretraining (2) Integration, then simply run bash script `run_san_LCM.sh`. Here, we showcase how to use our proposed pretraining module for the dependency parsing task. If you want to use it for custom data and different task. We recommend to do the needful modifications in it.

```bash
bash run_san_LCM.sh

```


## Acknowledgements
Much of the base code is from ["DCST Implementation"](https://github.com/rotmanguy/DCST)
