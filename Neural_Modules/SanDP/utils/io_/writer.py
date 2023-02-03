
class Writer(object):
    def __init__(self, alphabets):
        self.__source_file = None
        self.alphabets = alphabets

    def start(self, file_path):
        self.__source_file = open(file_path, 'w')

    def close(self):
        self.__source_file.close()

    def write(self, word, pos, ner, head, arc, lengths, auto_label=None, symbolic_root=False, symbolic_end=False):
        batch_size, _ = word.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        for i in range(batch_size):
            for j in range(start, lengths[i] - end):
                w = self.alphabets['word_alphabet'].get_instance(word[i, j])
                p = self.alphabets['pos_alphabet'].get_instance(pos[i, j])
                n = self.alphabets['ner_alphabet'].get_instance(ner[i, j])
                t = self.alphabets['arc_alphabet'].get_instance(arc[i, j])
                h = head[i, j]
                if auto_label is not None:
                    m = self.alphabets['auto_label_alphabet'].get_instance(auto_label[i, j])
                    self.__source_file.write('%d\t%s\t%s\t%s\t%d\t%s\t%s\n' % (j, w, p, n, h, t, m))
                else:
                    self.__source_file.write('%d\t%s\t%s\t%s\t%d\t%s\n' % (j, w, p, n, h, t))
            self.__source_file.write('\n')

class Index2Instance(object):
    def __init__(self, alphabet):
        self.__alphabet = alphabet

    def index2instance(self, indices, lengths, symbolic_root=False, symbolic_end=False):
        batch_size, _ = indices.shape
        start = 1 if symbolic_root else 0
        end = 1 if symbolic_end else 0
        instnaces = []
        for i in range(batch_size):
            tmp_instances = []
            for j in range(start, lengths[i] - end):
                instamce = self.__alphabet.get_instance(indices[i, j])
                tmp_instances.append(instamce)
            instnaces.append(tmp_instances)
        return instnaces