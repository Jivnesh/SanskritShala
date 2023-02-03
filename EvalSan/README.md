Official code for the paper ["Evaluating Neural Word Embeddings for Sanskrit"](https://aclanthology.org/2023.wsc-csdh.2/).

# EvalSan: Evaluation Toolkit for Sanskrit Embeddings
SanEval is a toolkit for evaluating the quality of Sanskrit embeddings. We assess their generalization power by using them as features on a broad and diverse set of tasks. We include a suite of **4 intrinsic tasks** which evaluate on what linguistic properties are encoded in word embeddings. Our goal is to ease the study and the development of general-purpose fixed-size word representations for Sanskrit.

## Dependencies
This code is written in python. The dependencies are:
* Python 3.6
```bash
pip install -r requirements.txt
```

## Evaluation tasks

### Intrinsic tasks
* SanEval includes a series of *Intrinsic* tasks to evaluate what linguistic properties are encoded in your word embeddings.
* We use `SLP1` transliteration scheme for our data. You can change it to another scheme using [this](https://colab.research.google.com/drive/1vdrQ8hJjZf-es-34tLHIWP8VBFf-o-fW?usp=sharing) code.

| Task     	| Metric                         	| #dev 	| #test 	|
|----------	|------------------------------	|-----------:|----------:|
| [Relatedness](https://github.com/Jivnesh/EvalSan/blob/main/evaluations/Intrinsic/Data/automated_relatedness_AK_test.csv)	| F-score	| 4.5k     	| 9k    	|
| [Similarity](https://github.com/Jivnesh/EvalSan/blob/main/evaluations/Intrinsic/Data/final_synonym_MCQs_AK.csv)	| Accuracy	| na     	| 3k    	|
| [Categorization Syntactic](https://github.com/Jivnesh/EvalSan/blob/main/evaluations/Intrinsic/Data/final_syntactic_categorization.csv)	| Purity	| na     	| 1.1k    	|
| [Categorization Semantic](https://github.com/Jivnesh/EvalSan/blob/main/evaluations/Intrinsic/Data/final_semantic_categorization.csv)	| Purity	| na     	| 150    	|
| [Analogy Syntactic](https://github.com/Jivnesh/EvalSan/blob/main/evaluations/Intrinsic/Data/final_syntactic_analogies.csv)	| Accuracy	| na    	| 10k    	|
| [Analogy Semantic](https://github.com/Jivnesh/EvalSan/blob/main/evaluations/Intrinsic/Data/Final_semantic_analogies.csv)	| Accuracy	| na    	| 6.4k    	|

## Pretrained models
* You can download the pretrained models from [this](https://iitk-my.sharepoint.com/:u:/g/personal/jivnesh_iitk_ac_in/ESQmKNWjkfBAgmghymAC1pcBT3sj0XxtIGdRgXatpWiymw?e=H13LCR) link. `README.md` is given for each model.
* Place the `models` folder in the parent directory path.
* Pretrained vectors can be downloaded from [this](https://iitk-my.sharepoint.com/:u:/g/personal/jivnesh_iitk_ac_in/EVpoZqJYLwBMiAM0NzSqiFwBiV9hfpSl7ZQ1Yq4b2aW-og?e=NjYEiY) link. Place this folder in `EvalSan/evaluations/Intrinsic/` path. This vectors are being used in evaluation script.

## How to train the models
Please refer to the `models` folder for more details.
```bash
bash train_embeddings.sh
```

## How to run evaluation
To evaluate your word embeddings, run the following command:
```bash
bash run_SanEval.sh
```

## Citation
If you use our tool, we'd appreciate if you cite the following paper:
```
@inproceedings{sandhan-etal-2023-evaluating,
    title = "Evaluating Neural Word Embeddings for {S}anskrit",
    author = "Sandhan, Jivnesh  and
      Paranjay, Om Adideva  and
      Digumarthi, Komal  and
      Behra, Laxmidhar  and
      Goyal, Pawan",
    booktitle = "Proceedings of the Computational {S}anskrit {\&} Digital Humanities: Selected papers presented at the 18th World {S}anskrit Conference",
    month = jan,
    year = "2023",
    address = "Canberra, Australia (Online mode)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.wsc-csdh.2",
    pages = "21--37",
}

```
## License
This project is licensed under the terms of the `Apache license 2.0`.
