#manage.py is automatically created in each Django project. manage.py is a thin wrapper around django-admin.py that takes care of two things before delegating to django-admin.py:
#       >> It puts the project’s package on sys.path.
#       >>It sets the DJANGO_SETTINGS_MODULE environment variable so that it points to the project’s settings.py file.
#!/usr/bin/env python
import os
import sys

if __name__ == "__main__":
    class DCS:
        def __init__(self, sent_id, sentence):
             self.sent_id = sent_id
             self.sentence = sentence
             self.dcs_chunks = []
             self.lemmas = []
             self.cng = []
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
