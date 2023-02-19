# TransLIST : A Transformer-Based Linguistically Informed Sanskrit Tokenizer

## Requirements
* Python 3.7.3
* Pytorch 1.5.0
* CUDA 9.2


## Installation
Create an environment using the `tlat0.yml` file given in the root directory.
```
 conda env create -f tlat0.yml
 conda activate tlat0
 bash requirements.sh [PATH to your conda environments] (e.g. ~/anaconda3/envs)
 ```
In case you are not able to run `requirements.sh`; complete the following steps manually. Replace the files `tester.py`, `dataset.py`, `metrics.py` and `crf.py` as per the files given in `fastnlp-copy` directory inside the root directory. You can find the paths of these files at:
```
tlat0/lib/python3.7/site-packages/fastNLP/core --> 'tester.py', 'dataset.py', 'metrics.py'
tlat0/lib/python3.7/site-packages/fastNLP/modules/decoder --> 'crf.py'
```

## Datasets
We have used two datasets: (1) [`SIGHUM`](https://zenodo.org/record/803508#.YRdZ43UzaXJ) (2) [`Hackathon`](https://sanskritpanini.github.io/dataset.html)

## Setup
Download [these](https://drive.google.com/drive/folders/1SCU_MwzePYai0ymAE3B5xuC8JhBZlxMd?usp=sharing) required files and keep them in root directory, then run ```bash setup.sh```

## How to train Models?
We propose 2 model variants:`SHR` and `Ngram` on two datasets: `SIGHUM` and `Hackathon`. The total of 4 settings are as follows

1. Training "SIGHUM Translist Ngram" : 

	Run ```bash set_sighum_ngram.sh [PATH to your conda environments]``` in the root directory or do the following modifications manually,
	- Place SIGHUM data in sktWS folder ( from root/sktWS dir run : ```cp SIGHUM/* .``` )
	- tlat0/.../fastNLP/core/dataset.py : line 310 : ```setting = "sighum-ngram"```
	- Use the embeddings in the SIGHUM_embeds folder. (from root dir run : ```cp SIGHUM_embeds/* .```)
	
	Next:
	- From V0 directory run : ```python flat_main_bigram.py --status train --batch 4```
	- Rename saved model in V0/saved_model


2. Training "SIGHUM Translist SHR"
	
	Run ```bash set_sighum_shr.sh [PATH to your conda environments]``` in the root directory or do the following modifications manually,
	- Place SIGHUM data in sktWS folder ( from root/sktWS dir run : ```cp SIGHUM/* .``` )
	- tlat0/.../fastNLP/core/dataset.py : line 310 : ```setting = "sighum-shr"```
	- Use the embeddings in the SIGHUM_embeds folder. (from root dir run : ```cp SIGHUM_embeds/* .```)
	
	Next:
	- From V0 directory run : ```python flat_main_bigram.py --status train --batch 8```
	- Rename saved model in V0/saved_model
	
3. Training "Hackathon Translist Ngram"
	
	Run ```bash set_hack_ngram.sh [PATH to your conda environments]``` in the root directory or do the following modifications manually,
	- Place Hackathon data in sktWS ( from root/sktWS dir run : ```cp hackathon/* .``` )
	- tlat0/.../fastNLP/core/dataset.py : line 310 : ```setting = "hack-ngram"```
	- Use the embeddings in the Hackathon_data/embeds folder. (from root dir run : ```cp Hackathon_data/embeds/* .```)
	
	Next:
	- From V0 directory run : ```python flat_main_bigram.py --status train --batch 4```
	- Rename saved model in V0/saved_model

4. Training "Hackathon Translist SHR"
	
	Run ```bash set_hack_shr.sh [PATH to your conda environments]``` or do manually...
	- Place Hackathon data in sktWS ( from root/sktWS dir run : ```cp hackathon/* .``` )
	- tlat0/.../fastNLP/core/dataset.py : line 310 : ```setting = "hack-shr"```
	- Use the embeddings in the Hackathon_data/embeds folder. (from root dir run : ```cp Hackathon_data/embeds/* .```)
	
	Next:
	- From V0 directory run : ```python flat_main_bigram.py --status train --batch 8```
	- Rename saved model in V0/saved_model

    
## Inference
Note that we also call our `Path Ranking for Corrupted Predictions (PRCP)` module as `Constrained Inference (CI)` module. Please note that for all the 4 settings; we share our pretrained models in [saved_models](https://drive.google.com/file/d/11RP9TqrO3-dUuPSCNa9GWEGOPcaFVv1h/view?usp=sharing).

1. Testing "SIGHUM Translist Ngram" :
	
	Run ```bash set_sighum_ngram.sh [PATH to your conda environments]```
	- From V0 run ```python flat_main_bigram.py --status test --test_model best_sighum_ngram2```
	- Don't run CI (not applicable in this case)
    
2. Testing "SIGHUM Translist SHR" :
	
	Run ```bash set_sighum_shr.sh [PATH to your conda environments]```
	- From V0 run ```python flat_main_bigram.py --status test --test_model best_sighum_shr2```
	- Run CI from root dir : ```python constrained_inference.py --dataset sighum```

3. Testing "Hackathon Translist Ngram" :
	
	Run ```bash set_hack_ngram.sh [PATH to your conda environments]```
	- From V0 run ```python flat_main_bigram.py --status test --test_model best_hack_ngram2```
	- don't run CI (not applicable in this case)
    
4. Testing "Hackathon Translist SHR" :
	
	Run ```bash set_hack_shr.sh [PATH to your conda environments]```
	- From V0 run ```python flat_main_bigram.py --status test --test_model best_hack_shr2```
	- run CI from root dir : ```python constrained_inference.py --dataset hackathon``` 

		
## Interactive Mode 
In interactive mode, the user can provide any sentence and the associated graphml file will be generated by scraping the SHR on the fly. The graphml file will then be converted to lattice and a pretrained model `best_sighum_shr2` will give its prediction on the sentence. Finally, Constrained Inference will be applied and it will output the segmented sentence. From root/V0 directory, run the following script.

```
python interactive_module.py --sentence="[YOUR SENTENCE HERE]" (Sentence expected to be in SLP format)
```

Demo example is illustrated below. 
```
python interactive_module.py --sentence="ahaM sOBapateH senAm AyasEr BujagEr iva"
Output : 
inp_data_i :  ['ahaM', 'sOBapateH', 'senAm', 'AyasEr', 'BujagEr', 'iva']
Model prediction :  ['aham', 'sOBa_pateH', 'senAm', 'AyasEH', 'BujagEH', 'iva']
Final Segmentation :  aham sOBa pateH senAm AyasEH BujagEH iva
```

## Web-based tool
You can interact with TransLIST on our SanskritShala's web-based platform: [`Link`](https://cnerg.iitkgp.ac.in/translist/)

## License
This project is licensed under the terms of the `Apache license 2.0`.


## Acknowledgements
We build our system on top of the codebase released by [Flat-Lattice](https://github.com/LeeSureman/Flat-Lattice-Transformer).
