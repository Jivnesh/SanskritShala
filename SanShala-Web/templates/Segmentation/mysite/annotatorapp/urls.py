from django.urls import path, re_path
from . import views
#urls are specified which are mapped to corresponding views over the path specified

app_name = 'annotatorapp'
urlpatterns = [
	path('', views.index, name='index'),
	path('wordtable/',views.wordtableview,name='wordtableview'),
	path('linetable/',views.sentenceview,name='sentenceview'),
	path('wordsinsentencetable/',views.wordsinsentenceview,name='wordsinsentenceview'),
	path('sentences/<int:batch_id>',views.xsentenceview,name='xsentenceview'),
	path('presentdata/',views.presentdataview,name='presentdataview'),
	re_path(r'select/(?P<sent_id>[0-9]+)/(?P<wordoption_id>[0-9]+)/$',views.select_wordoptionview,name='select_wordoption'),
	re_path(r'eliminate/(?P<sent_id>[0-9]+)/(?P<wordoption_id>[0-9]+)/$',views.eliminate_wordoptionview,name='eliminate_wordoption'),
	re_path(r'refresh/(?P<sent_id>[0-9]+)/$',views.reset_allselectionview,name='reset_allselection'),
	path('presentdata/ajax/save_data/',views.save_dragdata,name='save_dragdata'),
	path('presentdata/ajax/get_data/',views.get_dragdata,name='get_dragdata'),
	path('presentdata/ajax/save_data_to_db/',views.save_data_to_db,name='save_data_to_db'),
	path('presentdata/ajax/get_word_form/',views.get_form_data,name='get_form_data'),
	path('presentdata/ajax/get_xsent_sol/',views.get_sol_data,name='get_sol_data'),
	path('presentdata/ajax/save_to_server/',views.go_to_prodigy,name='go_to_prodigy'),
	path('presentdata/ajax/go_to_pos/',views.go_to_lemma,name='go_to_lemma'),
	path('presentdata/ajax/extract_segmentation/',views.predict_segmentation,name='predict_segmentation')
	]
