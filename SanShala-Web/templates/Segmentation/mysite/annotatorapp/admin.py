#django-admin.py is Django’s command-line utility for administrative tasks

from django.contrib import admin
from .models import Sentences,linetypes,WordOptions,Wordsinsentence,User,Noun,Indeclinables,Verbs,Exsentences


admin.site.register(Sentences)
admin.site.register(Wordsinsentence)
admin.site.register(WordOptions)
admin.site.register(linetypes)
admin.site.register(User)
admin.site.register(Noun)
admin.site.register(Indeclinables)
admin.site.register(Verbs)
admin.site.register(Exsentences)
