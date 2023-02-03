
import numpy as np
from tqdm import tqdm
import pandas as pd
import networkx as nx
import pickle
import difflib
import random

import flair, torch
flair.device = torch.device('cpu') 

import fitlog
from datetime import datetime
import pickle
import sys
sys.path.append('../')
from load_data import *
import argparse
from paths import *
from fastNLP.core import Trainer
from fastNLP.core import Tester
# from trainer import Trainer
from fastNLP.core import Callback
from V0.models import Lattice_Transformer_SeqLabel, Transformer_SeqLabel
import torch
import collections
import torch.optim as optim
import torch.nn as nn
from fastNLP import LossInForward
from fastNLP.core.metrics import SpanFPreRecMetric,AccuracyMetric
from fastNLP.core.callback import WarmupCallback,GradientClipCallback,EarlyStopCallback,FitlogCallback
from fastNLP import LRScheduler
from torch.optim.lr_scheduler import LambdaLR
# from models import LSTM_SeqLabel,LSTM_SeqLabel_True
from fastNLP import logger
from fastNLP.io.model_io import ModelLoader
from utils import get_peking_time
from V0.add_lattice import equip_chinese_ner_with_lexicon
from load_data import load_toy_ner

import traceback
import warnings
import sys
import pdb
from utils import print_info
from fastNLP.io.model_io import ModelLoader
import inspect
import subprocess
import argparse


#import flair
#import flair, torch
#flair.device = torch.device('cpu') 

#lang_model_path = "../lang_model_sanskrit.pkl"
#with open(lang_model_path,'rb') as f:
#    language_model = pickle.load(f)
   

def mapper(a,b):
    sep = '_'
    err = 0
#     print("a,b : ",a,b)
    mapping = []
    last = ''
    s = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for t in s.get_opcodes():
        tag,i1,i2,j1,j2 = t

        if(tag == 'equal'):
            for k, l in zip(list(a[i1:i2]), list(b[j1:j2])):
                mapping.append((k,l))


        if(tag == 'insert'):

            if len(mapping)==0 : return -1
            mapping[-1] = (mapping[-1][0],mapping[-1][1]+str(b[j1:j2]))

        if(tag == 'delete'):
            
            return -1



        if(tag == 'replace'):
            if(len(a[i1:i2])==1):
                mapping.append((str(a[i1:i2]),str(b[j1:j2])))

            elif(len(a[i1:i2])==len(b[j1:j2])):
                for k, l in zip(list(a[i1:i2]), list(b[j1:j2])):
                    mapping.append((k,l))


            else:
                charsa = list(a[i1:i2])
                charsb = list(b[j1:j2])
                while(sep in charsb):
                    index = charsb.index(sep)
                    charsb[index - 1] = charsb[index - 1] + charsb[index]
                    charsb.pop(index)

                if(len(charsa)==len(charsb)):
                    for k in range(len(charsa)):
    #                     temp.append(charsa[k]+' '+charsb[k]+'\n')
                        mapping.append((charsa[k],charsb[k]))

                elif(len(charsa) < len(charsb)):
                    charsb[len(charsa)-1] = ''.join(charsb[len(charsa)-1:len(charsb)])
                    for k in range(len(charsa)):
    #                     temp.append(charsa[k]+' '+charsb[k]+'\n')
                        mapping.append((charsa[k],charsb[k]))

                else:

                    return -1

                    for k in range(len(charsb)):
    #                     temp.append(charsa[k]+' '+charsb[k]+'\n')
                        mapping.append((charsa[k],charsb[k]))

                    for k in range(len(charsb), len(charsa)):
    #                     temp.append(charsa[k]+' '+'.'+'\n')
                        mapping.append((charsa[k],'.'))
    return mapping


def manual_mapper(a,b):
    mapping = []
    sep = '_'
    sp = b.split(sep)

    if a[0]==b[0] and a[-1]==b[-1]:
#         print("yes")
        i = 0
        j = 0
        while(i<len(a) and j<len(b)):
#             print("i,j : ",i,j)
            if (a[i]==b[j] and j+1<len(b) and  b[j+1] != sep) or j==len(b)-1:
#                 print("if : ",(a[i],b[j]))
                mapping.append((a[i],b[j]))
                i += 1
                j += 1

            else:
                s = ''
                for z in range(3):
#                     print(a[i],s)
                    s += b[j]
                    if j+1 >= len(b) or i+1 >= len(a)  : return -1
                    if a[i+1] == b[j+1]: break
                    elif z<2 : j += 1 
                mapping.append((a[i],s))
#                 print("else : ",(a[i],s))
                i += 1
                j += 1
            
        return mapping
    else: return -1

def char_mapper(a,b):
    m = mapper(a,b)
    if m == -1:
        m = manual_mapper(a,b)
    return m


def all_possible_words(t,tokens):
    sep = '_'
    s,e = t 
#     print("range : ",s,e)
    apw = []
    for w,pos in tokens:
#         a,b  = tokens[(iw,w)]
        a,b = pos
        if a == s: 
#             print("word : ",w)
            x = b-1
            y = b
            z = b+1

            if z < e :
                wz = all_possible_words((z,e),tokens)

                tokens_w = [(k,pk) for k,pk in tokens if (k,pk)!=(w,pos)]
                wy = all_possible_words((y,e),tokens_w)
#                 print("wy : ", wy)
                ws = wz+wy
                _ws = [w+sep+x for x in ws]
                apw += _ws

            else:
                apw += [w]

    return apw


def constrained_inference(inp_data_i, pred_data_i, nodes):
	#pred_data_i = ['etat', 'cau_anyat', 'ca', 'kOravya', 'prasoNgi', 'kawuka_udayam']
	p = '_'.join(pred_data_i).replace('_', ' ')
	#print("p : ", p)
	sp = pred_data_i
	prll = [t.split('_') for t in sp]
	#print("prll : ", prll)
	o = '_'.join(inp_data_i).replace("_",' ')
	inp_spl = o.split()
	
	#print("prll : ", prll)
	#print("inp_spl : ", inp_spl)
	#print("nodes : ", nodes)

	tokens = {}
	for nd in nodes:
		#print("tokens : ", tokens)
		#print("nd : ", nd)
		w,s,e,ch = nd
		if ch-1 not in tokens:
			tokens[ch-1] = [(w,(s,e))]
		else:
			if (w,(s,e)) not in tokens[ch-1]:
				tokens[ch-1].append((w,(s,e)))
	#print("tokens : ", tokens)

	words_l = [w[0] for k in tokens for w in tokens[k]]
	
	if len(set(p.split()) - set(words_l)) > 0: ## some subwords are missing in token
		#print("Operation start...")
		abs_index = 0
		for j in range(len(prll)): ## for every chunk
		    #print("------------")
		    if j!= 0 : 
		        abs_index += len(inp_spl[j-1])
		        abs_index += 1
		    #print("abs_index : ",abs_index)
		    pred_j = prll[j]
		    #print("\npred_j : ",pred_j)
		    tokens_j = [(t[0],(t[1][0]-abs_index,t[1][1]-abs_index)) for t in tokens[j]]
		    
		    #print("tokens_j : ",tokens_j)

		    token_j_words = [tpl[0] for tpl in tokens_j]
		    bad_chunk = 0 ## deafault : no
		    for subw in pred_j:
		        if subw not in token_j_words:
		            bad_chunk = 1 # yes
		    #print("bad chunk : ", bad_chunk)
		    if bad_chunk == 0 : continue
		
	#             size = max([t[1] for t in tokens_j.values()])+1
		    size = max([t[1][1] for t in tokens_j])+1
		    ref = min([t[1][0] for t in tokens_j])
		  
		    chunk_len = len(inp_spl[j])
		    #print("size,chunk_len : ",size,chunk_len)
		    #pwords = all_possible_words((0,size-1),tokens_j)
		    pwords = all_possible_words((ref,size-1),tokens_j)
		    #print("pwords : ",pwords)
		    if len(pwords) == 0 : continue

		    e_max = 0
		    e_p_max = 0
		    p_min = 99999999999999
		    
		    for w in pwords:
	#                 print("inp,w : ",inp_spl[j],w)
		        mapp = char_mapper(inp_spl[j],w)
	#                 print("mapp len : ",len(mapp))
	#                 print("mapp : ",mapp)
		        if mapp == -1 : continue
		        if len(mapp) != chunk_len : continue
		        
		        ## From Flat-Lattice Logits
		        energy = 0
		        for v in range(chunk_len):
	#                     print("v : ",v)
		            c_in, c_out = mapp[v]
		            if c_out in char2id_fl:
	#                         print("j : ",abs_index+v)
		                energy += logits_fl_new_i[abs_index+v][0][char2id_fl[c_out]]
		      
		        energy /= len(w.split('_')) ## penalty fn 1
		     
		        perplexity = 1 #language_model.calculate_perplexity(w) # 1 ## to bypass LM

		        e_p = energy/perplexity
		        #print(w,energy,perplexity)


		        if e_p > e_p_max:
		            e_p_max = e_p
		            w_max = w


	#         print("chosen : ",w_max)
		    prll[j] = w_max.split('_')

		predicted_list = [i for l in prll for i in l]
	#             print("predicted list : ",predicted_list)
		new_p = ' '.join(predicted_list)

	else:
		new_p = p

	return new_p

def _build_args(func, **kwargs):
    """
    根据func的初始化参数，从kwargs中选择func需要的参数

    :param func: callable
    :param kwargs: 参数
    :return:dict. func中用到的参数
    """
    spect = inspect.getfullargspec(func)
    if spect.varkw is not None:
        return kwargs
    needed_args = set(spect.args)
    defaults = []
    if spect.defaults is not None:
        defaults = [arg for arg in spect.defaults]
    start_idx = len(spect.args) - len(defaults)
    output = {name: default for name, default in zip(spect.args[start_idx:], defaults)}
    output.update({name: val for name, val in kwargs.items() if name in needed_args})
    return output

def _data_forward(func, x, _predict_func_wrapper):
    """A forward pass of the model. """
    
    x = _build_args(func, **x)
    y = _predict_func_wrapper(**x)
    return y

#'bigrams': torch.tensor([[101,   6, 163, 615,  36, 153,  23, 160,   3, 163, 615,  36,  19,   2,
#          38, 482, 483,   5,  26, 150,   3,   2,  18,  50,   5,  13,  17, 313,
#         270, 464,  30,  38,  41, 415, 520, 180, 312, 353,  55,  43,   3,  10,
#          53]]),

#'target': torch.tensor([[13,  5,  2,  5,  3, 19, 35, 10, 11,  2,  5,  3, 19,  2,  3, 16, 43,  7,
#          2,  8, 11,  2,  3, 15,  7,  2, 12,  2, 53, 26,  9,  3, 16,  2, 42, 14,
#         16, 50, 17,  2, 11,  2,  6]])

#============================== MAIN Function =================================#

loader = ModelLoader()
model = loader.load_pytorch_model('saved_models/best_sighum_shr2')
#print("\nmodel loaded...\n")
model.eval()

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', type=str, required=True)
args = parser.parse_args()
#sent = "etac cAnyac ca kOravya prasaNgi kawukodayam"
# ahaM sOBapateH senAm AyasEr BujagEr iva,aham sOBa pateH senAm AyasEH BujagEH iva
sent = args.sentence
subprocess.run(["python","create_graphml/create_graphml.py",f"--sentence={sent}"])
sent = '_'.join(sent.split())
#print("sent : ", sent)
n = len(sent)
#print("n : ", n)
bg = torch.tensor([[1]*len(sent)])
with open('created_graphml.txt','r') as f:
	lines = f.readlines()
nodes = []
for line in lines:
	line = line[:-1]
	sp = line.split(',')
	w,s,e,ch = sp[0], int(sp[1]), int(sp[2]), int(sp[3])
	if not model.vocabs['lattice'].has_word(w) :
		continue
	nodes.append((w,s,e,ch))
	
#print("len(nodes) = ", len(nodes))
#print("nodes : ",nodes)
	
ltc = [model.vocabs['lattice'].word2idx[c] for c in sent] + [model.vocabs['lattice'].word2idx[node[0]] for node in nodes]
#print("ltc  : ", ltc)
sti = list(range(0,n)) + [node[1] for node in nodes]
#print("sti : ", sti)
edi = list(range(0,n)) + [node[2] for node in nodes]
#print("edi : ", edi)


"""
batch_x = {'target': bg, 'bigrams': bg, 'seq_len': torch.tensor([n]), 'lex_num': torch.tensor([59]), 'lattice': torch.tensor([[   13,     5,     2,    27,     3,    27,     4,    10,     9,     2,
            27,     3,    27,     2,     3,    17,   137,     6,     2,     7,
             9,     2,     3,    15,     6,     2,    12,     2,   263,    42,
             8,     3,    17,     2,   122,    14,    17,    30,    18,     2,
             9,     2,    11, 42366,    25,    19,    49,    19,   297,    67,
          8120,   136,  3226,  2903,  1826,   192,   332,    20,  2015,   933,
            35,  1027,    90,   112, 11696,    62,  7245, 10880,    19,  1665,
           217,   192,    59,   447, 22618,  1261,    47,   160,   120,   188,
            32,  1039,  4094,   924,  3312,   334, 20009,  3419,   496,  1413,
          4865,    82,   687,    20,    32,    21,   134,   745,  1456,   809,
            37,  2839]]), 'pos_s': torch.tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 23, 41,  8, 18, 40,  0, 38, 32, 40, 26, 15,
         32,  9, 39, 24, 38, 29, 25, 16, 23, 39, 37, 39, 15, 38, 20, 16, 35,  2,
         32, 36, 25, 34,  6, 24,  0,  5,  5,  6,  0, 15, 17, 24, 26, 27,  0, 37,
         32, 36, 27, 17, 12,  1, 25, 33, 32, 37, 26, 15]]), 'pos_e': torch.tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 30, 42,  9, 19, 41,  2, 39, 36, 42, 28, 18,
         35, 10, 42, 25, 41, 30, 26, 17, 25, 41, 42, 40, 21, 42, 21, 18, 36,  3,
         33, 37, 30, 35,  7, 26,  1,  6,  6, 10,  3, 16, 21, 27, 30, 30,  3, 39,
         36, 36, 29, 18, 13,  2, 27, 34, 34, 39, 27, 17]])}

"""
batch_x = {'target': bg, 'bigrams': bg, 'seq_len': torch.tensor([n]), 'lex_num': torch.tensor([len(nodes)]), 'lattice': torch.tensor([ltc]), 'pos_s': torch.tensor([sti]), 'pos_e': torch.tensor([edi])}

         
#print(batch_x)

"""
batch_x = {'target': torch.tensor([[11, 13, 10, 59, 17,  2,  6,  3,  8, 11,  2, 12,  2, 10,  2,  6,  3, 15,
          7,  4, 15,  5, 45,  3, 25,  2,  8,  2, 10,  5, 30,  3, 17, 11, 37,  5,
         20, 16,  4,  7,  9,  5,  2,  6]]), 'bigrams': torch.tensor([[104,  92, 181, 229,  55,   7,   4,  24, 150,   3,  13,  17,  15,  14,
           7,   4,  18,  50,  32, 110, 289,  46,  21,  78,  82,  26,   9,  15,
          84, 188,  37,  54, 195, 566, 215,   6,  39,  85,  51, 102,  42,   6,
          10,  53]]), 'seq_len': torch.tensor([44]), 'lex_num': torch.tensor([84]), 'lattice': torch.tensor([[    9,    13,    10,    13,    18,     2,    16,     3,     7,     9,
             2,    12,     2,    10,     2,    16,     3,    15,     6,     4,
            15,     5,     4,     3,    40,     2,     7,     2,    10,     5,
            30,     3,    18,     9,    71,     5,     2,    17,     4,     6,
             8,     5,     2,    11,   323, 11563,   676,    51,  1058,    46,
           212,  6982,    26, 23493,    56,  6982,  6450,  3870,    37,  3801,
           107,  2578,   323,   134, 13165,  2135,    88,   266,    23,    66,
           150,   647,    21,   303,   676,   224,    35,    78, 24063,   129,
          6982,  3135,  9877,   561,  9698,   597,   470,   218,   281,    69,
          1127,    61,    99,   557, 18565,   219,   676,  3055,   289, 24648,
           323, 15639,  1560,  6981,    33,    33,  1593,    26,  6708,   109,
           104,   953,   441,    25,   744,    46,    23,    19,  1433, 12313,
            67,    82,   469,   296,   648,    49,    21,    24]]), 'pos_s': torch.tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 34,  8, 34, 18, 24,  0, 13, 32, 13, 37,
         36, 32,  8, 36, 11, 17, 39,  1, 34, 10,  2, 37, 24, 38, 14, 21, 40, 17,
         35, 17, 34, 29, 10, 25,  0, 19, 32, 19,  8, 35, 24,  3,  4,  9, 26, 38,
         10, 40, 41, 40, 37,  2, 34, 10, 37, 38, 34, 24, 33, 32, 27, 12, 32,  2,
         17,  0,  1, 11, 11, 42,  0,  0,  5,  9,  9,  0,  4, 37,  1, 24,  0, 25,
         41, 26]]), 'pos_e': torch.tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
         36, 37, 38, 39, 40, 41, 42, 43, 37, 15, 40, 19, 28,  0, 15, 40, 14, 43,
         37, 36, 14, 40, 12, 22, 40,  3, 36, 12,  6, 40, 25, 40, 15, 22, 42, 20,
         36, 19, 36, 30, 11, 27,  6, 20, 37, 22, 15, 37, 30,  6,  6, 11, 28, 39,
         13, 41, 43, 43, 42,  3, 37, 14, 39, 43, 40, 30, 34, 34, 28, 13, 34,  2,
         22,  1,  2, 14, 13, 43,  2,  1,  6, 10, 12,  3,  5, 38,  2, 27,  2, 26,
         42, 27]])}
"""
       

#pred dict :  {'pred': tensor([[13,  5,  2,  5,  3, 19, 35, 10, 11,  2,  5,  3, 19,  2,  3, 16, 43,  7,
#          2,  8, 11,  2,  3, 15,  7,  2, 12,  2, 53, 26,  9,  3, 16,  2, 42, 14,
#         16, 50, 17,  2, 11,  2,  6]], device='cuda:0'), 



#t_sent = ','.join([model.vocabs['label'].idx2word[i] for i in batch_x['target'].tolist()[0]])
#print("target sent : ",t_sent)

ps = batch_x['pos_s'].tolist()[0]
pe = batch_x['pos_e'].tolist()[0]
inp_sent = ','.join([model.vocabs['lattice'].idx2word[i]+':'+str(ps[k])+'-'+str(pe[k]) for k,i in enumerate(batch_x['lattice'].tolist()[0])])
i_sent = ''.join([model.vocabs['lattice'].idx2word[i] for i in batch_x['lattice'].tolist()[0][:n]])
#print("inp_lattices : ", inp_sent)
#print("\ninp_sent : ", i_sent)
#print("lattice vocab : ",model.vocabs['lattice'].idx2word)

#print(hasattr(model, 'predict')) ## false

#print(model.forward)

#print(batch_x)

_predict_func = model.forward
_predict_func_wrapper = model.forward

pred_dict = _data_forward(_predict_func, batch_x, _predict_func_wrapper)
#print(pred_dict)
pred_ids = pred_dict['pred'].tolist()[0]
last_tags = pred_ids
seq_len = len(last_tags)
preds = [model.vocabs['label'].to_word(i) for i in pred_ids]
#print("preds : ",preds)
inps = list(i_sent)
#print("inps : ",inps)
assert len(inps) == len(preds)

inp_data_i = i_sent.split('_')
pred_data_i = []
pd = ""
for i in range(len(inps)):
	inp = inps[i]
	pred = preds[i]
	if inp == '_':
		pred_data_i.append(pd)
		pd = ""
	else:
		pd += pred	
pred_data_i.append(pd)

id2char_fl = model.vocabs['label'].idx2word
char2id_fl = {}
for i in id2char_fl:
	char2id_fl[id2char_fl[i]] = i 
	
logits_mix = pred_dict['logits'] 
logits_mix = [lij.cpu() for lij in logits_mix]
logits_i = []
for j in range(seq_len-1):
    lg = logits_mix[j]
    lt = last_tags[j+1]
    logits_i.append(lg.transpose(1,2)[0][lt].reshape(1,-1).numpy())
    
logits_i.append(logits_mix[-1])
softmax = nn.Softmax(dim=0)
logits_fl_new_i = [softmax(torch.tensor(l)).numpy() for l in logits_i]

## Add CI : 
"""
nodes = all graphml nodes (w,s,e,ch). 
Make token dict. 
if some someword missing in token
then investigate chunk by chunk
replace with best among all possible words
See constrained_inference.py
"""
#print("\nModel Output : ", ''.join(preds)) ## '_' separated

print("\ninp_data_i : ", inp_data_i)
print("\nModel prediction : ", pred_data_i)

ci_pred = constrained_inference(inp_data_i, pred_data_i, nodes)

#print(f"final pred : <start>{ci_pred}<end>\n")
print("\nFinal Segmentation : ",ci_pred)

