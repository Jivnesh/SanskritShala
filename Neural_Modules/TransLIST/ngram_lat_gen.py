import pandas as pd
import random
from tqdm import tqdm

r = 1 #fraction of lattice nodes per sentence

def create_lattice(sent):
    lattices = []
    chunks = sent.split()
    a = 0
    for chunk in chunks:
        l = len(chunk)
        for j in range(l):
            if j+1 < l : lattices.append((a+j,a+j+1,sent[a+j:a+j+2]))
            if j+2 < l : lattices.append((a+j,a+j+2,sent[a+j:a+j+3]))
            if j+3 < l : lattices.append((a+j,a+j+3,sent[a+j:a+j+4]))
        a = a + l + 1
    return lattices

df = pd.read_csv("LREC-Data/hack_LREC_data_complete.csv")
inp_sents = df['input'].tolist()
dcs_ids = df['DCS-ID'].tolist()
for i in tqdm(range(len(inp_sents))):
    sent = inp_sents[i]
    tuples = create_lattice(sent)
    if r < 1:
      tuples = random.sample(tuples,int(r*len(tuples)))
    dcs_id = dcs_ids[i]
    with open(f'hack_lattice_files/{dcs_id}.lat','w') as g:
        g.write('start,end,word\n')
        for t in tuples:
            g.write(f'{t[0]},{t[1]},{t[2]}\n')
