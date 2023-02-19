# SanskritShala Front-end
First you need to install the individual neural modules on your machine. You need not have a GPU in oder to make these pretrained systems work on your local machine. You may find more details on how to deploy toolkit on your local machine in the following section

## How to deploy SanskritShala on your local machine
We encourage you to create a separate enviroment for each task and install the necesary packages needed for front-end of the respective task.

### Installation
* `Django`: The front-end of the Word segemntor is built using this.
* `ReactJS`: Homepage and interactive dependency parsing annotator is built using this framework.
* `Flask`: Our Morphological tagger and compound identifier is build using this framework.

### Homepage
Install nvm using [this](https://gist.github.com/d2s/372b5943bce17b964a79) link.
```
cd SanskritShala/SanShala-Web/sanskritshala.github.io
nvm install v16.16.0
npm run build
npm start
```

### Word segmentor
Active the environment of Django installation and run the following command
```
cd SanskritShala/SanShala-Web/templates/Segmentation/mysite
python manage.py runserver 0.0.0.8000
```

### Morphological parser
Active the environment of Flask installation and run the following command
```
cd SanskritShala/SanShala-Web/templates/Pos-tagging
python app.py
```

### Dependency parser
Active the environment of Flask installation and run the following command
```
cd SanskritShala/SanShala-Web/templates/standalone_dp
python app.py
```

### Compound Identifier
Active the environment of Flask installation and run the following command
```
cd SanskritShala/SanShala-Web/templates/Compound_classifier
python run_SaCTI.py
```