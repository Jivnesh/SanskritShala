class Sentence(object):
    def __init__(self, words, word_ids, char_seqs, char_id_seqs):
        self.words = words
        self.word_ids = word_ids
        self.char_seqs = char_seqs
        self.char_id_seqs = char_id_seqs

    def length(self):
        return len(self.words)

class NER_DependencyInstance(object):
    def __init__(self, sentence, tokens_dict, ids_dict, heads):
        self.sentence = sentence
        self.tokens = tokens_dict
        self.ids = ids_dict
        self.heads = heads

    def length(self):
        return self.sentence.length()