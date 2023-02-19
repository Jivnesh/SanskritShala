# smart-sanskrit-annotator
Sanskrit Annotator App Using Machine Learning.
Download the zip file and extract it to a local directory.

## The modules and libraries required to run the app
```
python 3.6
pandas
bs4
requests
django
django-datatable
django_datatables_view
```
## Step-1
```
Install required above-mentioned Django packages. (Django verson 2.0, Python version 3.x
```

## Apply Migrations for the database with following command in Segmentation/mysite folder
```
python manage.py makemigrations
python manage.py migrate
```
## Populate Database with morph information with custom management command in Segmentation/mysite folder
```
python manage.py scrap
python manage.py scrap2
```
## Run this command in Segmentation/mysite/annotatorapp/management/commands
```
python scrap3.py
```
## To run the app run the following command inside outer mysite directory:
```
python manage.py runserver
```
The app will be hosted in your localserver
