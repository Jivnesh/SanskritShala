#landing page inputs taken as a form input.
from django import forms
from . import models


class inputlineform(forms.Form):
	model = forms
	line = forms.CharField(max_length = 100)
	linetype = forms.CharField(max_length = 100)
	