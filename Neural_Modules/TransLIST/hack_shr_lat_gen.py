from devconvert import dev2slp, iast2slp, slp2dev, slp2iast, slp2tex, slp2wx, wx2slp, dev2wx
import pandas as pd
import random
from tqdm import tqdm
import os
import networkx as nx



lrec_data_path = "LREC-Data/Hackathon_dcs.csv"
debug = False
form = 'iast'


df = pd.DataFrame(pd.read_csv(lrec_data_path))
inputs = df['input'].tolist()
dcs_ids = df['DCS-ID'].tolist()
dcs2sent = {}
for i in range(len(inputs)):
	dcs2sent[dcs_ids[i]] = inputs[i]
	
#print(dcs2sent[435527])
#----------------------------------------------------------------------------------------------------------------------------------------------------

def graphml_corrector(sent_slp, triplets_slp):
	
	"""
	in case of sandhi first word will be short, second word would be long. e.g. [0, 2, 'yena'] + [3, 6, 'idam'] = yenedaM {Exception : end of a chunk, end of sentence, sandhi between short and long od samw vowel}
	 
	Find max match in -2 to -2 window
	
	if same match then prefer min deviation
	"""
	
	#triplets = [(t[0],t[1],t[2],t[3]) for t in triplets_slp]
	
	#triplets = list(set(triplets_slp)) ## erase duplicates if any
	
	corrected_triplets = []
	
	for t in triplets_slp:
		if debug : print("\n\ntriplet : ",t)
		w,s,e,c = t
		
		max_pos = 0
		max_match = 0
		my_range = [s,s+1,s-1,s+2,s-2,s+3,s-3,s+4,s-4,s+5,s-5] ## min deviation rule taken care
		if debug : print("my range : ", my_range)
		good_range = [r for r in my_range if r>=0 and r<=len(sent_slp)-1]
		if debug : print("good range : ",good_range)
		for i in good_range:
			k = i
			match = 0
			for j in range(len(w)):
				
				if k>=0 and k<=len(sent_slp)-1 and sent_slp[k] == w[j]: match += 1
				k += 1
			if debug : print("match : ", match)	
			if match > max_match : 
				max_match = match
				max_pos = i
		
		if max_match == 0 : continue
				
		s = max_pos
		e = s + len(w) -1
		if debug : print(sent_slp[e],w[-1])
		if debug : print(s,e)
		dirgha = (sent_slp[e] == w[-1].upper())
		if form == 'iast' : dirgha = (iast2slp.convert(sent_slp[e]) == iast2slp.convert(w[-1]).upper())
		if e == len(sent_slp)-1 or sent_slp[e+1] == ' ' or dirgha : e = e 
		elif sent_slp[e] != w[-1] : e -= 1
		if debug : print(s,e)
		
		corrected_triplets.append((w,s,e,c))
		#print(f"{w}, {sent_slp[s:e+1]} ")
				
	return corrected_triplets


def graphml2lattice(gfile, tag):

	dcs = gfile.split('.')[0]
	sent = dcs2sent[int(dcs)]

	if tag == 'train' : folder = 'Hackathon_data/final_graphml_train'
	elif tag == 'dev' : folder = 'Hackathon_data/final_graphml_dev'
	elif tag == 'test' : folder = 'Hackathon_data/final_graphml_test'
	
	graph_input = open(folder+'/'+gfile, mode='rb')
	graph = nx.read_graphml(graph_input)
	nodes = list(graph.nodes(data = True))

	tpls = []
	for node in nodes:
		k,d = node
		w = d['word']
		s = d['char_pos']
		e = s + len(w) - 1
		ch = d['chunk_no']
		tpls.append((w,s,e,ch))

	tpls = sorted(list(set(tpls)))	
	
	corr_tpls = graphml_corrector(sent,tpls)
	
	latt = [(t[1],t[2],t[0]) for t in corr_tpls] ## start, end, word
	
	return latt
	
	
	
def handle_multiple_graphmls(graphmls,tag):
	
	for gfile in tqdm(graphmls):
		try:
			latt = graphml2lattice(gfile,tag) 
		except:
			continue
		dcs = gfile.split('.')[0]
		with open(f"hack_shr_lattice_files/{dcs}.lat",'w') as g:
			g.write('start,end,word\n')
			for t in latt:
			    g.write(f'{t[0]},{t[1]},{t[2]}\n')





#----------------------------------------------------------------------------------------------------------------------------------------------------




## test if number of sentences (dcs) == number of graphmls

#print(len(inputs))

#print( len(os.listdir('Hackathon_data/final_graphml_dev'))+len(os.listdir('Hackathon_data/final_graphml_test')) + len(os.listdir('Hackathon_data/final_graphml_train')))

## Test Successful

#latt = graphml2lattice('435527.graphml','dev')
#print(latt)


graphmls = os.listdir('Hackathon_data/final_graphml_train')
handle_multiple_graphmls(graphmls,'train')

graphmls = os.listdir('Hackathon_data/final_graphml_dev')
handle_multiple_graphmls(graphmls,'dev')

graphmls = os.listdir('Hackathon_data/final_graphml_test')
handle_multiple_graphmls(graphmls,'test')

		


