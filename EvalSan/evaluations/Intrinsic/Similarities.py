import pandas as pd
from tqdm import tqdm
import numpy as np
import itertools
def evaluate_similarity_MCQs(emb, model, similarities_testfile = "Data/final_synonym_MCQs_AK.csv"):
    words = emb.words
    data = pd.read_csv(similarities_testfile, sep=",")
    ans = []
    pred = []
    writer = open('Data/MCQs-Predictions/'+model+'.txt','w')
    for i in range(len(data)):
        q = data.iloc[i,0]
        a = data.iloc[i,1]
        b = data.iloc[i,2]
        c = data.iloc[i,3]
        d = data.iloc[i,4]
        if a in words and b in words and c in words and d in words and q in words:
            ans.append(a)
            p = ''
            vec_a = emb[a]
            vec_b = emb[b]
            vec_c = emb[c]
            vec_d = emb[d]
            vec_q = emb[q]
            cosa = vec_a.dot(vec_q.T)/(np.linalg.norm(vec_a)*np.linalg.norm(vec_q))
            cosb = vec_b.dot(vec_q.T)/(np.linalg.norm(vec_b)*np.linalg.norm(vec_q))
            cosc = vec_c.dot(vec_q.T)/(np.linalg.norm(vec_c)*np.linalg.norm(vec_q))
            cosd = vec_d.dot(vec_q.T)/(np.linalg.norm(vec_d)*np.linalg.norm(vec_q))
            if cosa >= max([cosb, cosc, cosd]):
                pred.append(a)
                p = a
            elif cosb >= max([cosa, cosc, cosd]):
                pred.append(b)
                p = b
            elif cosc >= max([cosa, cosb, cosd]):
                pred.append(c)
                p = c
            else:
                pred.append(d)
                p = d
            writer.write(','.join([q,a,b,c,d,p])+'\n')    
    writer.close()
    print("Similarity (Accuracy): {}%".format((np.sum(np.asarray(ans) == np.asarray(pred))/len(ans))*100))