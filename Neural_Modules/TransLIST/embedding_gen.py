import pandas
import torch
from torch import nn
from tqdm import tqdm

data = pandas.read_csv('./LREC-Data/hellwig_LREC_data_complete.csv')
char_emb_path = 'hellwig_embeds/skt.char.vec'
word_emb_path = 'hellwig_embeds/skt.word.vec'
word_char_mix_path = 'hellwig_embeds/skt_word_char_mix.txt'
bigram_emb_path = 'hellwig_embeds/skt.bigram.vec'
characters = []
words = []
bigrams = []
words0 = set()
words1 = set()
words2 = set()
words3 = set()
words4 = set()
for i,row in tqdm(data.iterrows()):
        temp = list(set(list(row['input'].replace(' ','_'))))
        characters = list(set(characters+temp))
        temp = list(set([row['input'].replace(' ','_')[i:i+2] for i in range(len(row['input'])-1)]))
        bigrams = list(set(bigrams+temp))
        temp = list(set(list(row['output'].replace(' ','_'))))
        characters = list(set(characters+temp))
        temp = list(set([row['output'].replace(' ','_')[i:i+2] for i in range(len(row['output'])-1)]))
        bigrams = list(set(bigrams+temp))
        temp = row['output'].split('_')
        #words = list(set(words+list(set(temp))))
        if i%5==0 : words0.update(temp)
        if i%5==1 : words1.update(temp)
        if i%5==2 : words2.update(temp)
        if i%5==3 : words3.update(temp)
        if i%5==4 : words4.update(temp)

words = list(words0|words1|words2|words3|words4)

#print('Characters: ',characters[0:10],str(len(words) != len(set(words))))
#print('Words: ',words[0:10],str(len(words) != len(set(words))))

embedding = nn.Embedding(len(characters),50)
f = open(char_emb_path,'w')
for i in range(len(characters)):
	temp = embedding(torch.LongTensor([i])).tolist()
	f.write(characters[i]+' '+str(' '.join(str(v) for v in temp))+'\n')
f.close()

embedding = nn.Embedding(len(bigrams),50)
f = open(bigram_emb_path,'w')
for i in range(len(bigrams)):
    temp = embedding(torch.LongTensor([i])).tolist()
    f.write(bigrams[i]+' '+' '.join([str(x) for x in temp[0]])+'\n')
f.close()


embedding = nn.Embedding(len(words),50)
f = open(word_emb_path,'w')
for i in range(len(words)):
	temp = embedding(torch.LongTensor([i])).tolist()
	f.write(words[i]+' '+str(' '.join(str(v) for v in temp))+'\n')
f.close()


lexicon_f = open(word_emb_path,'r')
char_f = open(char_emb_path,'r')
output_f = open(word_char_mix_path,'w')

lexicon_lines = lexicon_f.readlines()
for l in lexicon_lines:
    l_split = l.strip().split()
    if len(l_split[0]) != 1:
        print(l.strip(),file=output_f)

char_lines = char_f.readlines()
for l in char_lines:
    print(l.strip(),file=output_f)

