import os
import argparse
import networkx as nx
import sentencepiece as spm
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
import pandas as pd
import random
import pickle
from get_word2vec_glove import get_word2vec_glove_embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Intrinsic Tests')
    parser.add_argument('--WD',  help='Working Directory of Model like Node2Vec, fasttext, etc.', default="./")
    parser.add_argument('--model', choices=['word2vec','glove','FastText','CharLM','Albert','ELMO'], help='model used for evaluation')
    parser.add_argument('--emb_size', type=int, default=300, help='dimension of embedding')
    args = parser.parse_args()
    
    WD = args.WD
    model = args.model
    emb_size = args.emb_size
    
    WD_Intrinsic = '../../evaluations/Intrinsic'
    os.chdir(WD_Intrinsic)
    
    from Analogies import analogy_evaluator
    from Similarities import evaluate_similarity_MCQs
    
    os.chdir("word_embeddings_benchmarks")
    from word_embeddings_benchmarks.web.vocabulary import CountedVocabulary, count, Vocabulary
    from word_embeddings_benchmarks.web.embedding import Embedding
    from word_embeddings_benchmarks.web.evaluate import calculate_purity, evaluate_categorization,evaluate_relatedness,evaluate_relatedness_classification
    os.chdir("..")
    
    emb_dict = {}
    word = None
    if model == 'FastText':
        with open('./vec_files/FastText.300.vec', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                emb_dict[word] = np.fromstring(vec, sep=' ')
     
        vocab = Vocabulary(words = emb_dict.keys())
        vectors = []
        for word in emb_dict.keys():
            vectors.append(emb_dict[word])
        emb = Embedding(vocab, vectors)

    elif model == 'CharLM':
        with open('./vec_files/CHARLM.525.vec', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                emb_dict[word] = np.fromstring(vec, sep=' ')
        vocab = Vocabulary(words = emb_dict.keys())
        vectors = []
        for word in emb_dict.keys():
            vectors.append(emb_dict[word])
        emb = Embedding(vocab, vectors)

    elif model == 'word2vec' or model == 'glove':
        emb = get_word2vec_glove_embedding(model, emb_size)

    elif model == 'Albert':
        with open('./vec_files/ALBERT.768.vec', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                emb_dict[word] = np.fromstring(vec, sep=' ')
        vocab = Vocabulary(words = emb_dict.keys())
        vectors = []
        for word in emb_dict.keys():
            vectors.append(emb_dict[word])
        emb = Embedding(vocab, vectors)

    elif model == 'ELMO':
        with open('./vec_files/ELMo.1024.vec', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                emb_dict[word] = np.fromstring(vec, sep=' ')
        vocab = Vocabulary(words = emb_dict.keys())
        vectors = []
        for word in emb_dict.keys():
            vectors.append(emb_dict[word])
        emb = Embedding(vocab, vectors)

    else:
        pass

    print("######################################")
    print("                "+model+"   "+"        ")
    print("######################################")

    f = open('./Data/automated_relatedness_AK_test.csv','r')
    X = []
    Y = []
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.split(',')
        X.append([line[0],line[1]])
        Y.append(int(line[2].replace('\n','').strip()))
    X = np.asarray(X)
    Y = np.asarray(Y)

    if model in ['word2vec','glove','FastText','CharLM','ELMO']:
        lower_threshold = 0.25
        upper_threshold = 0.5

    else:
        lower_threshold = 0.5
        upper_threshold = 0.7
    acc,f_score = evaluate_relatedness_classification(emb, model, X, Y, lower_threshold, upper_threshold)
    print("Relatedness scores:: accuracy {} and fscore {}".format(acc,f_score))
    
    evaluate_similarity_MCQs(emb,model)

    X = []
    Y = []
    DF = pd.read_csv("Data/final_syntactic_categorization.csv",header=None,sep=',')
    for i in range(len(DF)):
        for word in DF.iloc[i,:]:
            X.append(word)
            Y.append(i)
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("Categorization (Syntactic): {}".format(evaluate_categorization(emb, X, Y)))


    X = []
    Y = []
    f = open('Data/final_semantic_categorization.csv','r')
    lines = f.readlines()
    f.close()
    for line in lines:
        line = line.split(',')
        line[-1] = line[-1].replace('\n','')
        for l in line[1:]:
            X.append(l)
            Y.append(line[0])
    X = np.asarray(X)
    Y = np.asarray(Y)
    print("Categorization (semantic): {}".format(evaluate_categorization(emb, X, Y)))
    print('Semantic Analogy')
    analogy_evaluator("Data/Final_semantic_analogies.csv", emb, model,'Predicted_semantic_analogies.csv', savefile = True)
    print('Syntactic Analogy')
    analogy_evaluator("Data/final_syntactic_analogies.csv", emb, model,'Predicted_syntactic_analogies.csv', savefile = True)