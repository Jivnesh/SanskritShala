import os
import sentencepiece as spm
import pdb
import gensim
from gensim.models import KeyedVectors
import pickle
from tqdm import tqdm
import numpy as np
def get_word2vec_glove_embedding(model,emb_size):
    sp_model = spm.SentencePieceProcessor()
    sp_model.load('../..//models/corpus_variants/Sentencepiece_Model'+'/model_vocab_32000.model')
    vectors = []
    f =  open('./Data/embedding_space_or_vocab.txt','r')
    lines = f.readlines()
    words = set()
    for line in lines:
        line = line.replace('\n','')
        words.add(line)
    f.close()
    words.remove('')
    if model == 'word2vec':
        oov = 0
        saved_model = gensim.models.Word2Vec.load('../../models/word2vec/saved_models/word2vec.model')
        for word in tqdm(words, desc = "Generating vectors"):
            enc = sp_model.encode_as_pieces(word)
            vector = np.zeros(emb_size)
            piece_size = 0
            for piece in enc:
                if piece in saved_model.wv.vocab:
                    vector += saved_model.wv[piece]
                    piece_size += 1

            if piece_size != 0:
                vector = vector/piece_size
            else:
                vector =  np.random.rand(emb_size)
                oov += 1
            vectors.append(vector)
    elif model == 'glove':
        emb_dict = {}
        oov = 0
        with open('../../models/GloVe/saved_models/vectors.txt', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                emb_dict[word] = np.fromstring(vec, sep=' ')
        for word in tqdm(words, desc = "Generating vectors"):
            enc = sp_model.encode_as_pieces(word)
            vector = np.zeros(emb_size)
            piece_size = 0
            for piece in enc:
                if piece in emb_dict.keys():
                    vector += emb_dict[piece]
                    piece_size += 1

            if piece_size != 0:
                vector = vector/piece_size
            else:
                oov += 1
                vector =  np.random.rand(emb_size)
            vectors.append(vector)
    else:
        pass
    print('Total number of OOV words are {}'.format(oov))
    os.chdir("word_embeddings_benchmarks")
    from word_embeddings_benchmarks.web.vocabulary import CountedVocabulary, count, Vocabulary
    from word_embeddings_benchmarks.web.embedding import Embedding
    from word_embeddings_benchmarks.web.evaluate import calculate_purity, evaluate_categorization
    os.chdir("..")
    vocab = Vocabulary(words = words)
    emb = Embedding(vocab, vectors)
    return emb