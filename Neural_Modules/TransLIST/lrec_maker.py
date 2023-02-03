from tqdm import tqdm

g = open('LREC-Data/hack_LREC_data_complete.csv','w')

g.write('split,input,output,DCS-ID\n')

did = 0

with open('sktWS/skt.train','r') as f:
  lines = f.readlines()

inp_sent = ""
gold_sent = ""
for line in tqdm(lines):
  if line == '\n':
    g.write(f'train,{inp_sent},{gold_sent},{did}\n')
    did += 1
    inp_sent = ""
    gold_sent = ""
  else:
    sp = line[:-1].split()
    inp_sent += sp[0]
    gold_sent += sp[1]



with open('sktWS/skt.test','r') as f:
  lines = f.readlines()

inp_sent = ""
gold_sent = ""
for line in tqdm(lines):
  if line == '\n':
    g.write(f'test,{inp_sent},{gold_sent},{did}\n')
    did += 1
    inp_sent = ""
    gold_sent = ""
  else:
    sp = line[:-1].split()
    inp_sent += sp[0]
    gold_sent += sp[1]
    
    
with open('sktWS/skt.dev','r') as f:
  lines = f.readlines()

inp_sent = ""
gold_sent = ""
for line in tqdm(lines):
  if line == '\n':
    g.write(f'dev,{inp_sent},{gold_sent},{did}\n')
    did += 1
    inp_sent = ""
    gold_sent = ""
  else:
    sp = line[:-1].split()
    inp_sent += sp[0]
    gold_sent += sp[1]

g.close()




