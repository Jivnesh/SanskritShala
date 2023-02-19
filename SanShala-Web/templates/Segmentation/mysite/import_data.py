#set up django environment
import os
import django
from django.conf import settings


os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
django.setup()

#parameters
data_dir = "data"

#script
from annotatorapp.models import Sentences,WordOptions
import pandas as pd
from tqdm import tqdm

sentences = os.listdir(data_dir)

for i in tqdm(range(len(sentences))):
	sent_id = sentences[i]
	
	with open(data_dir+"/"+sent_id+"/input_line.txt",encoding="utf8") as f:
		for line in f.readlines():
			line1 = line
			break
	try:	
		sent = Sentences(
							line = line1[13:],
							linetype = 'wx',
				)
		sent.save()
		df = pd.read_csv(data_dir+"/"+sent_id+"/dataframe.txt",encoding="utf8")
		
		for i in range(df.shape[0]):
			row = df.iloc[i]
			try:
				word_option = WordOptions(sentence = sent,
										level = row["level"],
										color_class = row["color_class"],
										position = row["position"],
										chunk_no = row["chunk_no"],
										lemma = row["lemma"],
										pre_verb = row["pre_verb"],
										morph = row["morph"],
										colspan = row["colspan"],
										wordlength = row["wordlenth"],
										aux_info = row["aux_inf"],
										word = row["word"],
										)
				word_option.save()
			except Exception as e:
				print("WordOption not inserted : ",sent_id,"/")
				print(e)

	except Exception as e:
		print(e)
		print("sent not inserted : ",sent_id)
		