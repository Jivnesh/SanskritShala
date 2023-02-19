# Generated by Django 2.0.4 on 2018-06-12 05:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('annotatorapp', '0003_auto_20180610_1551'),
    ]

    operations = [
        migrations.CreateModel(
            name='Indeclinables',
            fields=[
                ('ind_id', models.AutoField(primary_key=True, serialize=False)),
                ('sh', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Noun',
            fields=[
                ('noun_id', models.AutoField(primary_key=True, serialize=False)),
                ('sh', models.CharField(max_length=50)),
            ],
        ),
        migrations.CreateModel(
            name='Verbs',
            fields=[
                ('verb_id', models.AutoField(primary_key=True, serialize=False)),
                ('sh', models.CharField(max_length=50)),
            ],
        ),
    ]
