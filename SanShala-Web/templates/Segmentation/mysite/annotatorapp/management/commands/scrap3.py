import os
import pickle
import romtoslp as conv
#In case of any clarification, email to - amrith@iitkgp.ac.in
# The latest modifications can be found at: www.amrithkrishna.com 
# Class definition for pickle files
# currently the pickle files can be opened only with Python 3


#This file loads the sample 200 sentences from answers.txt onto a table
# in the db so that batches of it can be taken for testing

cur_dir = os.path.dirname(os.path.realpath(__file__))
dir_p = os.path.join(cur_dir,'pickle/')
dir_graphml = os.path.join(cur_dir,'graphml/')
class DCS:
	def __init__(self, sent_id, sentence):
         self.sent_id = sent_id
         self.sentence = sentence
         self.dcs_chunks = []
         self.lemmas = []
         self.cng = []
node = {'word':'', 'morph':'', 'morph_no':'',};
new_cng = []
_out = open('answers.txt','w',encoding='utf8')

for filename in os.listdir(dir_p):
    if('.p' in filename):

        #loading the .p files

        output_load = pickle.load(open(dir_p+filename, "rb"), encoding='utf-8')
        lemmas = sum(output_load.lemmas,[])
        roman_lemmas = sum(output_load.lemmas,[])
        for i in range(len(lemmas)):
            lemmas[i] = conv.rom_slp(lemmas[i])
        
        cng = sum(output_load.cng,[])
        g_file = dir_graphml+filename[:-1]+'txt'
        print(g_file)
        nodes = []
        with open(g_file,'r') as f:
            for line in f:
                if('edge source' in line):
                    break
                if('data key=\"d0\"' in line):
                    node['morph_no'] = line.replace('<','>').split('>')[2]
                if('data key=\"d4\"' in line):
                    node['word'] = line.replace('<','>').split('>')[2]
                if('data key=\"d5\"' in line):
                    node['morph'] = line.replace('<','>').split('>')[2]
                if('/node' in line):
                    nodes.append(node.copy())    
        nodes = [dict(t) for t in set([tuple(d.items()) for d in nodes])]
        
        # compare

        for lemma,_cng in zip(lemmas,cng):
            for cdict in nodes:
                if lemma == cdict['word'] and _cng == cdict['morph_no']:
                    new_cng.append(cdict['morph'])
        _out.write(str(output_load.sentence)+'#'+str(output_load.dcs_chunks)+'#'+str(roman_lemmas)+'#'+str(new_cng)+'\n')
        new_cng.clear()
_out.close()

	

