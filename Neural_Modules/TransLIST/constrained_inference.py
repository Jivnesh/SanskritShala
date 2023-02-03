
import numpy as np
from tqdm import tqdm
import pandas as pd
import networkx as nx
import pickle
import difflib
import random
import flair, torch
flair.device = torch.device('cpu') 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
args = parser.parse_args()
dataset = args.dataset



#dataset = 'hackathon' ## 'hackathon' or 'sighum'

test_data_path = 'sktWS/skt.test'
test_pred_path = 'test_predictions_flat_lattice.txt' 
label_vocab_path = 'V0/label_vocab.pkl' 
test_logits_path = 'test_logits_FL.pkl' 
lang_model_path = "lang_model_sanskrit.pkl"
pos_correct = False

if dataset == 'hackathon':
	lrec_data_path =  'LREC-Data/Hackathon_dcs.csv' 
	graphml_path =  "Hackathon_data/final_graphml_test"
	pos_correct = True
	enable_lang_model = False
	
elif dataset == 'sighum':
	lrec_data_path =  'LREC-Data/new_LREC_data_complete.csv'
	graphml_path =  'skt/After_graphml'
	enable_lang_model = True


enable_penalty = True
enable_translat_voting = True

with open(test_data_path,'r') as f: 
    test_data_fl = f.readlines()
    
inp_data = [] ## list of list of words
gold_data = [] ## list of list of words
inp_sent = ""
gold_sent = ""
for line in tqdm(test_data_fl):
    if line == '\n':
        inp_data.append(inp_sent.split('_'))
        inp_sent = ""
        gold_data.append(gold_sent.split('_'))
        gold_sent = ""
    else:
        line = line[:-1]
        sp = line.split()
        inp_sent += sp[0]
        gold_sent += sp[1]
        
with open(test_pred_path,'r') as f:
    test_pred_fl = f.readlines()
    
pred_data = [] ## list of list of chunks splitted by $
for line in tqdm(test_pred_fl):
    line = line[:-1]
    pred_data.append(line.split('$'))
    

with open(label_vocab_path,'rb') as f:
  id2char_fl = pickle.load(f)
    
"""
## FL 257 vocab
id2char_fl = {0: '<pad>', 1: '<unk>', 2: 'a', 3: '_', 4: 'A', 5: 't', 6: 'm', 7: 'r', 8: 'v', 9: 'i', 10: 'n', 11: 'y', 12: 's', 13: 'e', 14: 'u', 15: 'p', 16: 'k', 17: 'd', 18: 'H', 19: 'c', 20: 'a_', 21: 'z', 22: 'S', 23: 'h', 24: 'R', 25: 'B', 26: 'g', 27: 'j', 28: 'D', 29: 'l', 30: 'aH', 31: 'I', 32: 'f', 33: 'T', 34: 'o', 35: 'a_a', 36: 'H_', 37: 'U', 38: 'b', 39: 'E', 40: 't_', 41: 'i_', 42: 'w', 43: 'O', 44: 'Y', 45: 'AH', 46: 'm_', 47: 'M', 48: 'q', 49: 'a_e', 50: 'a_u', 51: 'K', 52: 'u_', 53: 'N', 54: 'G', 55: 'C', 56: 'W', 57: 'A_a', 58: 'A_', 59: 'a_i', 60: 'P', 61: 'I_', 62: 'A_e', 63: 'n_', 64: 'A_u', 65: 'i_i', 66: 'H_a', 67: 'A_i', 68: 'ya_', 69: 'va_', 70: '_f', 71: 'ra_', 72: 'Q', 73: 'ma_', 74: 'aH_', 75: 'na_', 76: 'ca_', 77: 'k_', 78: 'ta_', 79: 'a_I', 80: 'A_A', 81: 'f_', 82: 'Ra_', 83: 'la_', 84: '_a', 85: 'u_u', 86: 'w_', 87: 'a_o', 88: 'ka_', 89: 'At_', 90: 'pa_', 91: 'at_', 92: 'ha_', 93: 'Da_', 94: 'ga_', 95: 'U_', 96: 'a_U', 97: 'da_', 98: 'TA_', 99: 'Ta_', 100: 'sa_', 101: 'd_', 102: 'za_', 103: 'I_i', 104: 'Sa_', 105: 'tA_', 106: 'm_m', 107: 'Ka_', 108: 'wa_', 109: 'ja_', 110: 'J', 111: 'yA_', 112: 'am_', 113: 'am', 114: 'A_I', 115: 'Ba_', 116: 'vA_', 117: 'et_', 118: 'DA_', 119: 'Ga_', 120: 'nA_', 121: 'Pa_', 122: 'ni_', 123: 'dA_', 124: 'Wa_', 125: 'qa_', 126: 'A_U', 127: 'jA_', 128: 'uByam', 129: 'lA_', 130: 'rA_', 131: 'ri_', 132: 'ti_', 133: 'aH_u', 134: 'I_I', 135: 'ava', 136: 'e_', 137: 'sA_', 138: 'pi_', 139: 'e_i', 140: 'mA_', 141: 'Am_a', 142: 'wA_', 143: 'Qa_', 144: 'ama', 145: '_h', 146: 'n_an', 147: 'gi_', 148: 'BA_', 149: 'Ca_', 150: 'Ya_', 151: 'an', 152: 'kA_', 153: 'A_o', 154: '\\xf1', 155: 'aH_a', 156: 'AH_A', 157: 'It_', 158: 'O_', 159: 'KA_', 160: 'rI_', 161: 'gA_', 162: 'taH_', 163: 'it_', 164: 'ika_', 165: 'QA_', 166: 'ft_', 167: 'AH_', 168: 'zu', 169: 'tu_', 170: 'aH_i', 171: 'O_a', 172: 'qA_', 173: 'RA_', 174: 'CA_', 175: 'aN', 176: 'At_a', 177: 'An_a', 178: '_am', 179: 'smAkam', 180: 'sya', 181: 'At_A', 182: 'di_', 183: 'am_a', 184: 'uzva', 185: 'WA_', 186: 'ayA_', 187: 'S_', 188: 'taH', 189: 'ru_', 190: 'ba_', 191: 'm_Am', 192: 'na_n', 193: 'om', 194: 'nI_', 195: 'SA_', 196: 'yAy', 197: 'ya_a', 198: 'tAy', 199: 'm_am', 200: 'asva', 201: 'Ami', 202: 'e_I', 203: 'tAt_', 204: 'At', 205: 'ni', 206: 'am_i', 207: 'ipAsi', 208: 'dI_', 209: 'wi_', 210: 'ka_k', 211: 'n_An', 212: 'Am', 213: 'Ani_', 214: 'An_', 215: 'YA_', 216: 'zA_', 217: 'om_', 218: 'hu_', 219: 'cA_', 220: 'tO_', 221: 'hi_a', 222: 'li_', 223: 'su', 224: 'Ri_', 225: 'I_a', 226: 'yAm_', 227: 'iH_', 228: 'DeBy', 229: 'sya_', 230: 'DaH_', 231: 'nam_', 232: 'Ut_', 233: 'tay', 234: 'ena_a', 235: 'mi_', 236: 'D_', 237: 'hi_', 238: 'zu_', 239: 'u_izu_', 240: '_ISam', 241: 'p_', 242: 'tAH_', 243: 'sy', 244: 'yA', 245: 'j_', 246: 'vI_', 247: 'danI', 248: 'asya', 249: 'Di_', 250: 'A_nA', 251: 'ami', 252: 'ezu', 253: 'at_a', 254: 'Am_', 255: 'ti', 256: 'yu_'}
"""

char2id_fl = {}
for i in id2char_fl:
    char2id_fl[id2char_fl[i]] = i   
with open(test_logits_path,'rb') as f:
    logits_fl = pickle.load(f)

logits_fl_new = logits_fl

with open(lang_model_path,'rb') as f:
    language_model = pickle.load(f)


df = pd.DataFrame(pd.read_csv(lrec_data_path))
dcs_dict = {}
inputs = df['input'].tolist()
dcs_ids = df['DCS-ID'].tolist()
# len(inputs), len(dcs_ids)
for i in range(len(inputs)):
    dcs_dict[inputs[i]] = dcs_ids[i]
    
rev_dcs_dict = {}
for k in dcs_dict.keys():
    rev_dcs_dict[dcs_dict[k]] = k
    
def position_corrector(pos,w,chunk):
	
	max_pos = 0
	max_match = 0

	for i in range(len(chunk)):
		k = i
		match = 0
		for j in range(len(w)):
			if chunk[k] == w[j]: match += 1
			k += 1
			if (k>len(chunk)-1) : break
		if match > max_match : 
			max_match = match
			max_pos = i
			
	if max_match == 0 : return -1
	
	best_pos = max_pos
	min_dist = abs(best_pos - pos)	
	for i in range(len(chunk)):
		k = i
		match = 0
		for j in range(len(w)):
			if chunk[k] == w[j]: match += 1
			k += 1
			if (k>len(chunk)-1) : break
		if match == max_match : 
			if abs(i-pos) < min_dist:
				min_dist = abs(i-pos)
				best_pos = i
			
	return best_pos


def generate_matrix(index): ## to generate "tokens" which contain all possible words for a sentence 

    file_path = f'{graphml_path}/{index}.graphml'
    # file_path = 'output.graphml'

    graph = nx.read_graphml(file_path)
    nodes = list(graph.nodes(data = True))
    
    orig_sent = rev_dcs_dict[index]
    orig_sp = orig_sent.split()
    
    #print(orig_sp)
    
    words = [] ## --> list of tuples (word,cng,lemma,morph,position,chunk_no,length_word)
    for item in nodes:
        first = item[1]
        word = first['word']
        cng = first['cng']
        lemma = first['lemma']
        pre_verb = ""
        if 'pre_verb' in first.keys():
            pre_verb = first['pre_verb']
        else:
            pre_verb = ""
        morph = first['morph']
        position = first['position']
        chunk_no = first['chunk_no']
        length_word = first['length_word']
        chunk_word = orig_sp[chunk_no-1]
        
        #print(word,chunk_no,position,chunk_word)
        
        if pos_correct : 
        	position = position_corrector(position,word,chunk_word)
        	#print("-->",word,chunk_no,position,chunk_word)
        	if position == -1 : continue
        	
        
      
        words.append((word,cng,lemma,morph,position,chunk_no,length_word))
    #         print(item)
    
    #print("words : ",words)

    
    orig_sp = [w+'.' for w in orig_sp]
    orig_sent = ' '.join(orig_sp)

    matrix = []
    chunk_visited = []
    chunk = []
    o_c = 1
    chunk_flag = 0
    tokens = {}
    for it in range(len(words)):
        t = words[it]
#         print("-->",t)
        w = t[0]
        p = t[4]
        n_c = t[5]
        o_w = orig_sp[n_c-1]
#         wl = t[6]
        wl = len(w)
    
        if n_c != o_c: chunk_flag = 0
       
          
        if p == 1 :
            if wl!=1 and w[:-1] == o_w[:wl-1]:
#                 print(w,p,wl,w[:-1],o_w[:wl-1])
#                 p = 0
                chunk_flag = 1
#                 print(f"Chunk flag -1 for {w,p,n_c}")
        if p == -1:
#             print(w,p,wl,o_w,w[:-1],o_w[:wl-1])
            if w[:-1] == o_w[:-1][-wl:-1]:
                p = len(o_w)-1-wl
            elif w[:-1] == o_w[:wl-1]:
                p = 0
                chunk_flag = -1
            else:
                continue
                
        if chunk_flag == 1 : 
            p = p-1

                
        ###### word level #######
        if n_c-1 in tokens.keys():
#             tokens[n_c-1].add(w)
#             tokens[n_c-1][(it,w)] = (p,p+len(w)-1)
            if (w,(p,p+len(w)-1)) not in tokens[n_c-1]:
                tokens[n_c-1].append((w,(p,p+len(w)-1)))
#             print(n_c-1,it,w,(p,p+len(w)-1))
        else:
#             
            tokens[n_c-1] = [(w,(p,p+len(w)-1))]
        #########################
        
        
        s = '.'*p+w
        if n_c != o_c:
            if n_c in chunk_visited: continue
            matrix.append(chunk)
#             tokens[o_c] = list(set(_chunk))
            chunk_visited.append(o_c)
            chunk = [s]
#             _chunk = [w]
            o_c = n_c
            
        else:
            chunk.append(s)
#             _chunk.append(w)

    matrix.append(chunk)


    max_len = 0        
    for l in matrix:
        if len(l)>max_len:
            max_len = len(l)

    for l in matrix:
        l += (max_len - len(l))*['.']


    orig_len = len(orig_sent)

    final_matrix = [] ## list of strings
    for i in range(max_len):
        s = ''
        for j in range(len(matrix)):
            l = matrix[j]
            s += (l[i]+(len(orig_sp[j])-len(l[i]))*'.'+' ')
        _s = s + str(len(s)-1)

        final_matrix.append(s)
    return final_matrix, tokens ## tokens = {chunk_id : [(word,(pos1,pos2)),(...),(...)]}


    
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



## NEW word level post processing : all possible, max energy, LM, penalty in more split


corr = 0
err = 0
total = 0
rect = 0
r_pred_data = []
lim = 3
with open('post_processed_preds_fl.txt','w') as pp_writer:
    for i in tqdm(range(len(inp_data))):
        #print("inp : ",inp_data[i]) # ['etac', 'cAnyac', 'ca', 'kOravya', 'prasaNgi', 'kawukodayam']
        #print("gold : ",gold_data[i]) # ['etat', 'ca', 'anyat', 'ca', 'kOravya', 'prasaNgi', 'kawuka', 'udayam']
        #print("pred : ",pred_data[i]) # ['etat', 'ca_anyat', 'ca', 'kOravya', 'prasaNgi', 'kawuka_udayam']
        g = ' '.join(gold_data[i])
        #print("g : ",g) # etat ca anyat ca kOravya prasaNgi kawuka udayam
        p = '_'.join(pred_data[i]).replace('_', ' ') 
        #print("p : ",p) # etat ca anyat ca kOravya prasaNgi kawuka udayam
        sp = pred_data[i]
        prll = [t.split('_') for t in sp]
        #print("prll : ",prll) # [['etat'], ['ca', 'anyat'], ['ca'], ['kOravya'], ['prasaNgi'], ['kawuka', 'udayam']] ## chunk-wise
        o = '_'.join(inp_data[i]).replace("_",' ')
        #print("o : ",o) # etac cAnyac ca kOravya prasaNgi kawukodayam
        inp_spl = o.split()

        #print("inp : ",o)
        #print("g : ",g)
        #print("p : ",p)

        if o not in dcs_dict.keys(): 
            r_pred_data.append(p.split())
            #print("continue1...")
            continue
        output = generate_matrix(dcs_dict[o]) ## check whther any prediction contain words which are not in tokens
#         print("output : ",output)
        if output == -1 : 
            r_pred_data.append(p.split())
            #print("continue2...")
            continue

        total += 1
        fm, tokens = output
        #print("tokens : ",tokens) # {chunk_id : (word,(pos1,pos2))}
        # tokens :  {0: [('etat', (0, 3))], 1: [('ca', (0, 1)), ('anyat', (1, 5))], 2: [('ca', (0, 1))], 3: [('kOravya', (0, 6)), ('kO', (0, 1)), ('ravya', (2, 6))], 4: [('prasaNgi', (0, 7))], 5: [('kawuka', (0, 5)), ('kawukA', (0, 5)), ('kawu', (0, 3)), ('kA', (4, 5)), ('uda', (5, 7)), ('udayam', (5, 10)), ('Uda', (5, 7)), ('yam', (8, 10))]}
        huge = 0
        words_l = [w[0] for k in tokens for w in tokens[k]]
        #print("words_l : ",words_l) # ['etat', 'ca', 'anyat', 'ca', 'kOravya', 'kO', 'ravya', 'prasaNgi', 'kawuka', 'kawukA', 'kawu', 'kA', 'uda', 'udayam', 'Uda', 'yam']
        if len(set(p.split()) - set(words_l)) > 0: ## some subwords are missing in token
            #print("\nNeed modification")
            err += 1
            abs_index = 0
            #print("prll : ",prll)
            for j in range(len(prll)): ## for every chunk
                if j!= 0 : 
        #             for s in prll[j-1]:
        #                 abs_index += len(s)
                    abs_index += len(inp_spl[j-1])
                    abs_index += 1
        #         print("abs_index : ",abs_index)
                pred_j = prll[j]
                #print("pred_j : ",pred_j)
                tokens_j = tokens[j]
                #print("tokens_j : ",tokens_j)

                token_j_words = [tpl[0] for tpl in tokens_j]
                bad_chunk = 0 ## no
                for subw in pred_j:
                    if subw not in token_j_words:
                        bad_chunk = 1 # yes
                if bad_chunk == 0 : continue
                #print("bad chunk : ",bad_chunk)

    #             size = max([t[1] for t in tokens_j.values()])+1
                size = max([t[1][1] for t in tokens_j])+1
                chunk_len = len(inp_spl[j])
    #             print("size,chunk_len : ",size,chunk_len)
                pwords = all_possible_words((0,size-1),tokens_j)
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
                            energy += logits_fl_new[i][abs_index+v][0][char2id_fl[c_out]]
                    
                    if enable_penalty:
                        energy /= len(w.split('_')) ## penalty fn 1
                    if not enable_translat_voting:
                        energy = 10 ## bypass translat voting
                    
                    if enable_lang_model:
                        perplexity = language_model.calculate_perplexity(w) ### change
                    else:
                        perplexity = 1 ## to bypass LM

                    e_p = energy/perplexity
    #                 print(w,energy,perplexity)


                    if e_p > e_p_max:
                        e_p_max = e_p
                        w_max = w


        #         print("chosen : ",w_max)
                prll[j] = w_max.split('_')

            predicted_list = [i for l in prll for i in l]
            r_pred_data.append(predicted_list)
#             print("predicted list : ",predicted_list)
            new_p = ' '.join(predicted_list)
            if new_p == g : rect += 1
    #         else : 
    #             print("i : ",i)
    #             print("inp : ",o)
    #             print("g : ",g)
    #             print("p : ",p)
    #             print("new_p : ",new_p)
        else:
            r_pred_data.append(p.split())

        ### save the post processed prediction with chunk boundary $
        pp_pred = '$'.join(['_'.join(pl) for pl in prll])
        pp_writer.write(pp_pred+'\n')



## Evaluation : word level metric


# inp_dict = {}
# for i in tqdm(range(len(inp_data))):
#     sent_i = ''.join(inp_data[i])
#     if sent_i not in inp_dict:
#         inp_dict[sent_i] = i
        
# inp_unique_ids = sorted(list(inp_dict.values()))

def list_intersection(la,lb):
    a = la.copy()
    b = lb.copy()
    inter = []
    while(len(a)*len(b)):
        x = a[0]
        if x in b:
            inter.append(x)
            a.remove(x)
            b.remove(x)
        else:
            a.remove(x)
    return inter

precisions = []
recalls = []
accuracies = []

with open('test_output_word_level_gitcode.txt', 'w') as p:

    for inp, outp, gen in zip(inp_data, gold_data, r_pred_data): ## r_pred_data

#     for i in inp_unique_ids:
#         inp = inp_data[i]
#         outp = gold_data[i]
#         gen = r_pred_data[i]

        inp_raw = ' '.join(inp)
        outp_raw = ' '.join(outp)
        gen_raw = ' '.join(gen)
    
#         print('i : ',i)
#         print('inp : ',inp)
#         print('outp : ',outp)
#         print('gen : ',gen)
        
    ## set based intersection
#         intersection = set(outp).intersection(gen)
#         prec = len(intersection)*1.0/len(set(gen))
#         recall = len(intersection)*1.0/len(set(outp))

    ## list based intersection
        intersection = list_intersection(gen,outp)
        prec = len(intersection)*1.0/len(gen)
        recall = len(intersection)*1.0/len(outp)


        if outp == gen:
            accuracies.append(1.0)
        else:
            accuracies.append(0.0)

        precisions.append(prec)
        recalls.append(recall)


#         log_line = str(inp_raw).replace('\n', '').lstrip() + ';' + str(outp_raw).replace('\n', '').lstrip() + ';' + str(gen_raw).replace('\n', '').lstrip() + ';' + str(prec).replace('\n', '') + ';' + str(recall).replace('\n', '') + '\n'
        log_line = inp_raw + ';' + '_'.join(outp) + ';' + '_'.join(gen) + '\n'
        p.write(log_line)



avg_prec = np.mean(precisions)*100.0
avg_recall = np.mean(recalls)*100.0
f1_score = 2*avg_prec*avg_recall/(avg_prec + avg_recall)
avg_acc = np.mean(accuracies)


print("Precision: " + str(avg_prec)) 
print("Recall: " + str(avg_recall))
print("F1_score: " + str(f1_score))
print("Accuracy: " + str(avg_acc))

