from tqdm import tqdm
import subprocess
import numpy as np
######################################## HELPER FUNCTIONS #########################################
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

###################################### MAIN #########################################
test_data_path = '../sktWS/skt.test'

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
        
        
#print(inp_data[:3]) 
# [['etac', 'cAnyac', 'ca', 'kOravya', 'prasaNgi', 'kawukodayam'], ['paTyaM', 'ca', 'BarataSrezWa', 'nigfhRIyAM', 'balena', 'tam'], ['yenedaM', 'vyasanaM', 'prAptA', 'Bavanto', 'dyUtakAritam']]
#print(gold_data[:3])
#[['etat', 'ca', 'anyat', 'ca', 'kOravya', 'prasaNgi', 'kawuka', 'udayam'], ['paTyam', 'ca', 'Barata', 'SrezWa', 'nigfhRIyAm', 'balena', 'tam'], ['yena', 'idam', 'vyasanam', 'prAptAH', 'BavantaH', 'dyUta', 'kAritam']]

outf = open('interactive_pred.txt','w')
for l in tqdm(inp_data[:100]):
	inp_sent = ' '.join(l)
	subprocess.run(["python","interactive_module.py",f"--sentence={inp_sent}"], stdout=outf)
outf.close()

with open('interactive_pred.txt','r') as f:
	lines = f.readlines()
	
pred_data = []
for line in lines:
	line = line[:-1]
	pred_data.append(line.split())
	
#print(pred_data[:3])

############################################ EVALUATION ###############################################

with open('eval.txt', 'w') as p:

    for inp, outp, gen in zip(inp_data[:100], gold_data[:100], pred_data[:100]):

        inp_raw = ' '.join(inp)
        outp_raw = ' '.join(outp)
        gen_raw = ' '.join(gen)
    
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

