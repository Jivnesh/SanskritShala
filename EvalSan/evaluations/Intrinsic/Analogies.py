import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from urllib.request import urlopen
from multiprocessing.dummy import Pool as ThreadPool
import itertools
def mini_evaluator(emb, a, b, c, d):
    if a in emb.words and b in emb.words and c in emb.words and d in emb.words:
        w1, w2, w3, w4 = a, b, c, d
        p = emb.nearest_neighbors(emb[w2] - emb[w1] + emb[w3], exclude=[w1, w2, w3])
    else:
        p = d
    return p

def analogy_evaluator(Analogy_csv, emb, model,output, savefile = False):
    print("Working on Analogies...")
    DF = pd.read_csv(Analogy_csv)
    A = list(DF["a"])
    B = list(DF["b"])
    C = list(DF["c"])
    D = list(DF["d"])
    
    # Pick a sample of data and calculate answers
    # answers = D
    # pool = ThreadPool(10)
    # predicted = pool.starmap(mini_evaluator, zip(itertools.repeat(emb), A, B, C, D))
    # pool.close()
    # pool.join()
     # Pick a sample of data and calculate answers
    _a = []
    _b = []
    _c = []
    predicted = []
    answers = []
    for i in tqdm(range(len(A))):
        a = A[i]
        b = B[i]
        c = C[i]
        d = D[i]
        if a in emb.words and b in emb.words and c in emb.words and d in emb.words:
            w1, w2, w3, w4 = a, b, c, d
            p = emb.nearest_neighbors(emb[w2] - emb[w1] + emb[w3], exclude=[w1, w2, w3])
            _a.append(a)
            _b.append(b)
            _c.append(c)
            answers.append(d)
            predicted.append(p)
    
    print("Correct predictions: {}%".format((np.sum(np.array(answers) == np.array(predicted))/len(answers))*100))
    
    if savefile:
        with open('./Data/Anology_Predictions/'+model+'_'+output, "w") as f:
            f.write("accuracy: {}\n".format(np.sum(np.array(answers) == np.array(predicted))/len(answers)))
            f.write("a b c answer prediction\n")
            for a, b, c, answer, prediction in zip(A, B, C, answers, predicted):
                f.write("{} {} {} {} {}\n".format(a, b, c, answer, prediction[0]))
    
    print("Analogy prediction for the file {} is ready at {}".format(Analogy_csv, './Data/Anology_Predictions/'+model+'_'+output))

def analogy_evaluator_for_semantic(Analogy_csv, emb, output = "Analogies Output", savefile = False):

    print("Working on Analogies for Conjugation Test...")
    DF = pd.read_csv(Analogy_csv)
    A = list(DF["a"])
    B = list(DF["b"])
    C = list(DF["c"])
    D = list(DF["d"])
    
    # Pick a sample of data and calculate answers
    correct = 0
    _a = []
    _b = []
    _c = []
    predicted = []
    answers = []
    for i in tqdm(range(len(A))):
        a = A[i]
        b = B[i]
        c = C[i]
        d = D[i].split(", ")
        if a in emb.words and b in emb.words and c in emb.words:
            w1, w2, w3, w4 = a, b, c, d
            p = emb.nearest_neighbors(emb[w2] - emb[w1] + emb[w3], exclude=[w1, w2, w3])
            _a.append(a)
            _b.append(b)
            _c.append(c)
            answers.append(d)
            predicted.append(p)
            if p in d:
                correct += 1
    
    predition_percent = (correct/len(predicted))*100
    print("Correct predictions: {}%".format(predition_percent))
    
    if savefile:
        timestr = time.strftime("%Y%m%d-%H%M%S")+".txt"
        with open(output+"/"+timestr, "x") as f:
            f.write("accuracy: {}\n".format(predition_percent))
            f.write("a : b : c : answers : prediction\n")
            for a, b, c, answer, prediction in zip(_a, _b, _c, answers, predicted):
                f.write("{} : {} : {} : {} : {}\n".format(a, b, c, answer, prediction[0]))
    
    print("Analogy prediction for the file {} is ready at {}".format(Analogy_csv, output+"/"+timestr))