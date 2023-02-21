# SanskritShala: A Neural Sanskrit NLP Toolkit

Official code for the paper ["SanskritShala: A Neural Sanskrit NLP Toolkit with Web-Based Interface for Pedagogical and Annotation Purposes"](https://arxiv.org/abs/2302.09527). If you use this code please cite our paper.


## Web-based tool
You can interact with our SanskritShala's web-based platform: [`Link`](https://cnerg.iitkgp.ac.in/sanskritshala/). We encourage you to check our [demo video](https://youtu.be/x0X31Y9k0mw4) to get familiar with our platform.


## Neural Modules of SanskritShala for 4 NLP tasks
You may find more details of codebases in [`Neural Modules`](https://github.com/Jivnesh/SanskritShala/tree/master/Neural_Modules) folder for word segementaion, morphological tagging, depedency parsing and compound type identification task.

## Deploy SanskritShala on your local machine
First you need to install the individual modules on your machine as instructed in the above section. You need not have a GPU in oder to make these pretrained systems work on your local machine. You may find more details on how to deploy toolkit on your local machine in [`SanShala-Web`](https://github.com/Jivnesh/SanskritShala/tree/master/SanShala-Web) folder.

## EvalSan: Evaluation Toolkit for Sanskrit Embeddings
SanEval is a toolkit for evaluating the quality of Sanskrit embeddings. We assess their generalization power by using them as features on a broad and diverse set of tasks. We include a suite of **4 intrinsic tasks** which evaluate on what linguistic properties are encoded in word embeddings. Our goal is to ease the study and the development of general-purpose fixed-size word representations for Sanskrit. You may find more details of codebases in [`EvalSan`](https://github.com/Jivnesh/SanskritShala/tree/master/EvalSan) folder.


### Evaluation tasks

#### Intrinsic tasks
* SanEval includes a series of *Intrinsic* tasks to evaluate what linguistic properties are encoded in your word embeddings.
* We use `SLP1` transliteration scheme for our data. You can change it to another scheme using [this](https://colab.research.google.com/drive/1vdrQ8hJjZf-es-34tLHIWP8VBFf-o-fW?usp=sharing) code.

| Task     	| Metric                         	| #dev 	| #test 	|
|----------	|------------------------------	|-----------:|----------:|
| [Relatedness](https://github.com/Jivnesh/SanskritShala/tree/master/EvalSan/evaluations/Intrinsic/Data/automated_relatedness_AK_test.csv)	| F-score	| 4.5k     	| 9k    	|
| [Similarity](https://github.com/Jivnesh/SanskritShala/tree/master/EvalSan/evaluations/Intrinsic/Data/final_synonym_MCQs_AK.csv)	| Accuracy	| na     	| 3k    	|
| [Categorization Syntactic](https://github.com/Jivnesh/SanskritShala/tree/master/EvalSan/evaluations/Intrinsic/Data/final_syntactic_categorization.csv)	| Purity	| na     	| 1.1k    	|
| [Categorization Semantic](https://github.com/Jivnesh/SanskritShala/tree/master/EvalSan/evaluations/Intrinsic/Data/final_semantic_categorization.csv)	| Purity	| na     	| 150    	|
| [Analogy Syntactic](https://github.com/Jivnesh/SanskritShala/tree/master/EvalSan/evaluations/Intrinsic/Data/final_syntactic_analogies.csv)	| Accuracy	| na    	| 10k    	|
| [Analogy Semantic](https://github.com/Jivnesh/SanskritShala/tree/master/EvalSan/evaluations/Intrinsic/Data/Final_semantic_analogies.csv)	| Accuracy	| na    	| 6.4k    	|

### Pretrained word embeddings
* You can download the pretrained models from [this](https://iitk-my.sharepoint.com/:u:/g/personal/jivnesh_iitk_ac_in/ESQmKNWjkfBAgmghymAC1pcBT3sj0XxtIGdRgXatpWiymw?e=H13LCR) link. `README.md` is given for each model.
* Place the `models` folder in the parent directory path.
* Pretrained vectors can be downloaded from [this](https://iitk-my.sharepoint.com/:u:/g/personal/jivnesh_iitk_ac_in/EVpoZqJYLwBMiAM0NzSqiFwBiV9hfpSl7ZQ1Yq4b2aW-og?e=NjYEiY) link. Place this folder in `EvalSan/evaluations/Intrinsic/` path. This vectors are being used in evaluation script.
* Our proposed LCM pretraining is available at [`EvalSan/LCM`](https://github.com/Jivnesh/SanskritShala/tree/master/EvalSan/LCM) folder. For more details please visit this link.

## Citation
If you use our tool, we'd appreciate if you cite our paper:
```
@misc{Sandhan_SanskritShala,
  doi = {10.48550/ARXIV.2302.09527},
  url = {https://arxiv.org/abs/2302.09527},
  author = {Sandhan, Jivnesh and Agarwal, Anshul and Behera, Laxmidhar and Sandhan, Tushar and Goyal, Pawan},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {SanskritShala: A Neural Sanskrit NLP Toolkit with Web-Based Interface for Pedagogical and Annotation Purposes},
  publisher = {arXiv},
  year = {2023},
  copyright = {Creative Commons Attribution 4.0 International}
}

```

## License
This project is licensed under the terms of the `Apache license 2.0`.


## Acknowledgements
We'd like to say thanks to everyone who helped us make the different neural models for SanskritShala.
