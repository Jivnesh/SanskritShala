import os
from django.core.management.base import BaseCommand
from annotatorapp.models import Noun, Indeclinables, Verbs
#collecting information on Morphs to populate the database
#The column SH in each table should be used for auto-completion, based on corresponding radio button.
#this information is being fetched from data.txt file in the location annotatorapp/managements/commands/data.txt

dirname = os.path.dirname(__file__)
path = os.path.join(dirname,'data.txt')

class Command(BaseCommand):
    help = 'Populates the database with Morph information'
 #collecting nouns for populating database
    def data_nouns(self):
        f = open(path)
        for _ in range(33):
            next(f)
        ln = f.readline()
        j = 1
        list = []
        while ln:
            j = j + 1
            if ln.startswith("<tr>"):
                for i in range(4):
                    ln = f.readline()
                list.append(ln[5:-6])
            ln = f.readline()
            if j == 216:
                break

        for k in list:
            try:
                nouns = Noun(sh=k)
                nouns.save()
            except Exception as e:
                print(e)
        f.close()
#collecting indeclinables for populating database
    def data_indeclinables(self):
        f = open(path)
        for _ in range(547):
            next(f)
        ln = f.readline()
        j = 1
        list = []
        while ln:
            j = j + 1
            if ln.startswith("<tr>"):
                for i in range(4):
                    ln = f.readline()
                list.append(ln[5:-6])
            ln = f.readline()
            if j == 21:
                break

        for k in list:
            try:
                indeclinables = Indeclinables(sh=k)
                indeclinables.save()
            except Exception as e:
                print(e)
        f.close()
#collecting verbs for populating database
    def data_verbs(self):
        f = open(path)
        for _ in range(606):
            next(f)
        ln = f.readline()
        j = 1
        list = []
        while ln:
            j = j + 1
            if ln.startswith("<tr>"):
                for i in range(4):
                    ln = f.readline()
                list.append(ln[5:-6])
            ln = f.readline()
            if j == 10404:
                break

        for k in list:
            try:
                verbs = Verbs(sh=k)
                verbs.save()
            except Exception as e:
                print(e)
        f.close()

    def handle(self, *args, **options):
        self.data_nouns()
        self.data_indeclinables()
        self.data_verbs()

