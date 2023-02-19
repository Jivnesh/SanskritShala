# SaCTI: Sanskrit Compound Type Identifier
 
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
You can interact with SaCTI on our SanskritShala's web-based platform: [`Link`](https://cnerg.iitkgp.ac.in/sacti/)


## License
This project is licensed under the terms of the `Apache license 2.0`.

## Acknowledgements
Much of the base code is from [Trankit](https://github.com/nlp-uoregon/trankit)


