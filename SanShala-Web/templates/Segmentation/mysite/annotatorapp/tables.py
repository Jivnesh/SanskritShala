from .models import Sentences,WordOptions,Wordsinsentence
from table import Table
from table.columns import Column

#this has been defined for the datatables on the site

#WordOptionsTable holds required data displayed when clicked on the collapsible in the nav bar
class WordOptionsTable(Table):
	id = Column(field='id',header='id')
	word = Column(field='word',header='word')
	lemma = Column(field='lemma',header='Lemma')
	morph = Column(field='morph',header='Morph')
	aux_info = Column(field='aux_info',header='aux_info')
	pre_verb = Column(field='pre_verb',header='pre_verb')

#SentenceTable holds required data displyed when clicked on the collapsible in the nav bar
class SentencesTable(Table):
	id = Column(field='id',header='id')
	line = Column(field='line',header='Sentence')

#WordsinsentenceTable holds required data displyed when clicked on the collapsible in the nav bar
class WordsinsentenceTable(Table):
	id = Column(field='id',header='id')
	word = Column(field='word',header='word')
	parent = Column(field='parent',header='parent')
	children = Column(field='children',header='children')
	relation = Column(field='relation',header='relation')
	wordoptions = Column(field='wordoptions',header='wordoptions')