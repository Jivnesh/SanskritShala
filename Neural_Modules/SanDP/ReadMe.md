Official code for the paper ["Systematic Investigation of Strategies Tailored for Low-Resource Settings for Low-Resource Dependency Parsing"](https://arxiv.org/abs/2201.11374).
If you use this code please cite our paper.

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

## Pretrained embeddings for Sanskrit
* Pretrained FastText embeddings for STBC/VST can be obtained from [here](https://drive.google.com/drive/folders/1SwdEqikTq-N2vOL7QSUX2vqi3faZE7bq?usp=sharing). Make sure that `.txt` file is placed at `data/`
* The main results are reported on the systems trained by combining train and dev splits. 


## How to train model for Sanskrit
To run proposed system: (1) Pretraining (2) Integration, then simply run bash script `run_STBC.sh` or `run_VST.sh` for the respective dataset. With these scripts you will be able to reproduce our results reported in Section-3 and Table 2.

```bash
bash run_STBC.sh

```

## Citations
```
@misc{sandhan_systematic,
  doi = {10.48550/ARXIV.2201.11374},
  url = {https://arxiv.org/abs/2201.11374},
  author = {Sandhan, Jivnesh and Behera, Laxmidhar and Goyal, Pawan},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Systematic Investigation of Strategies Tailored for Low-Resource Settings for Low-Resource Dependency Parsing},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## Acknowledgements
Our ensembled system is built on the top of ["DCST Implementation"](https://github.com/rotmanguy/DCST)
