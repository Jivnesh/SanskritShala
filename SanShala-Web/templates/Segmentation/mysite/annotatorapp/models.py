from django.db import models


class Sentences(models.Model):
	"""docstring for Sentence"""
	line = models.CharField(max_length=200)
	linetype = models.CharField(max_length=100)
	line_header = models.CharField(max_length=200)
	
	def __str__(self):
		return str(self.id) +"/ "+ self.line+"/"+self.linetype

	class Meta:
		db_table = "Sentences"

# class Example-Sentences

class linetypes(models.Model):
	"""docstring for linetypes"""
	linetype = models.CharField(max_length=100)

	def __str__(self):
		return str(self.linetype)
		
class Wordsinsentence(models.Model):
	"""docstring for Wordsinsentence"""
	sentence = models.ForeignKey(Sentences, on_delete=models.CASCADE,null=True)
	word = models.CharField(max_length=100)
	parent = models.IntegerField(default=0)
	children = models.CharField(max_length=100)
	relation = models.CharField(max_length=100)
	wordoptions = models.CharField(max_length=100)
	chunkno = models.IntegerField(default=0)

	def __str__(self):
		return str(self.id)+self.word
	
	class Meta:
		unique_together = (("sentence", "word","parent","children","relation",'wordoptions'),)
		db_table = "Wordsinsentence"

class WordOptions(models.Model):
	"""docstring for WordOptions"""
	sentence = models.ForeignKey(Sentences, on_delete=models.CASCADE,null=True)
	level = models.IntegerField(default=0)
	color_class = models.CharField(max_length=100)
	position = models.IntegerField(default=0)
	chunk_no = models.IntegerField(default=0)
	lemma = models.CharField(max_length=100)
	word =  models.CharField(max_length=100,default="")
	pre_verb = models.CharField(max_length=100)
	morph = models.CharField(max_length=100)
	colspan = models.IntegerField(default=0)
	wordlength = models.IntegerField(default=0)
	aux_info = models.CharField(max_length=100)
	isEliminated = models.BooleanField(default=False)
	isSelected = models.BooleanField(default=False)
	parent = models.IntegerField(default=-1)
	children = models.CharField(max_length=100,default='')
	relation = models.CharField(max_length=100,default='')

	def __str__(self):
		return str(self.id) +self.word+ "( "+ self.lemma+", "+self.morph+" )"

	class Meta:
		unique_together = (("sentence","level","color_class","position","chunk_no","lemma","pre_verb","morph","colspan","wordlength","aux_info"),)		
		db_table  = "WordOptions"

class User(models.Model):
	"""docstring for User"""
	user_id = models.AutoField(primary_key=True)
	savedSentence = models.CharField(max_length=100)
	clickSequence = models.CharField(max_length=100)
	init_time = models.CharField(max_length=100)
	end_time = models.CharField(max_length=100)

class Noun(models.Model):
	"""docstring for Noun"""
	noun_id = models.AutoField(primary_key=True)
	sh = models.CharField(max_length=50)

class Indeclinables(models.Model):
	"""docstring for Indeclinables"""
	ind_id = models.AutoField(primary_key=True)
	sh = models.CharField(max_length=50)

class Verbs(models.Model):
	"""docstring for Verbs"""
	verb_id = models.AutoField(primary_key=True)
	sh = models.CharField(max_length=50)

class Exsentences(models.Model):
	"""docstring for Example Sentences"""
	xsent_id = models.AutoField(primary_key=True)
	line = models.CharField(max_length=200)
	chunks = models.CharField(max_length=200)
	lemmas = models.CharField(max_length=200)
	morph_cng = models.CharField(max_length=200)



