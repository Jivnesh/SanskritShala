# SaCTI: Sanskrit Compound Type Identifier

Official code for the paper ["A Novel Multi-Task Learning Approach for  Context-Sensitive Compound Type Identification in Sanskrit"](https://aclanthology.org/2022.coling-1.358/). If you use this code please cite our paper.
 
## Requirements
* Python 3.9.x
* Pytorch 1.11.0
* Cuda 11.3
* Transformers(huggingface) 4.17.0
* sklearn:0.22.1

Please install the following dependecies.

```
pip3 install -r requirements.txt
```

## Datasets

The datasets are available in [data]() folder: `English`, `Marathi`, `saCTI-base coarse labels`, `saCTI-base fine labels`, `saCTI-large coarse labels`, `saCTI-large labels`.


## How to train model
To train the model, you need to run `main.py` with the following flags.
* `model_path` : path to save model.
* `experiment` : the name of the dataset on which you want to experiment. The list of datasets are given below. Default: `saCTI-base coarse`
* `epochs` : Number of epochs. Default:70
* `batch_size` : Size of batch. Default:50

The list of datasets: `english`, `marathi`, `saCTI-base coarse`, `saCTI-base fine`, `saCTI-large coarse`, `saCTI-large fine`. (Refer to `data_config.py` file for more details.)

```
python3 main.py --model_path='save_models' --experiment='english' --epochs=70 --batch_size=75
```

## For inference 
`Note`: Please note that the results reported in our paper are averaged over 4 runs.
```
python3 main.py --model_path='save_models' --experiment='english' --training= False
```

## Data annotation framework
If you are interested in our data annotation framework, you can check [`Annotation_Framework`](https://github.com/hrishikeshrt/classification-annotation) for the more details.

## Web-based tool
Coming soon ...


## Citation
If you use our tool, we'd appreciate if you cite our paper:
```
@inproceedings{sandhan-etal-2022-novel,
    title = "A Novel Multi-Task Learning Approach for Context-Sensitive Compound Type Identification in {S}anskrit",
    author = "Sandhan, Jivnesh  and Gupta, Ashish  and Terdalkar, Hrishikesh  and Sandhan, Tushar  and Samanta, Suvendu  and Behera, Laxmidhar  and Goyal, Pawan",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.358",
    pages = "4071--4083",
    abstract = "The phenomenon of compounding is ubiquitous in Sanskrit. It serves for achieving brevity in expressing thoughts, while simultaneously enriching the lexical and structural formation of the language. In this work, we focus on the Sanskrit Compound Type Identification (SaCTI) task, where we consider the problem of identifying semantic relations between the components of a compound word. Earlier approaches solely rely on the lexical information obtained from the components and ignore the most crucial contextual and syntactic information useful for SaCTI. However, the SaCTI task is challenging primarily due to the implicitly encoded context-sensitive semantic relation between the compound components. Thus, we propose a novel multi-task learning architecture which incorporates the contextual information and enriches the complementary syntactic information using morphological tagging and dependency parsing as two auxiliary tasks. Experiments on the benchmark datasets for SaCTI show 6.1 points (Accuracy) and 7.7 points (F1-score) absolute gain compared to the state-of-the-art system. Further, our multi-lingual experiments demonstrate the efficacy of the proposed architecture in English and Marathi languages.",
}
```

## License
This project is licensed under the terms of the `Apache license 2.0`.

## Acknowledgements
Much of the base code is from [Trankit](https://github.com/nlp-uoregon/trankit)


