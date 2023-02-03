from .instance import NER_DependencyInstance
from .instance import Sentence
from .prepare_data import ROOT, END, MAX_CHAR_LENGTH

class Reader(object):
    def __init__(self, file_path, alphabets):
        self.__source_file = open(file_path, 'r')
        self.alphabets = alphabets

    def close(self):
        self.__source_file.close()

    def getNext(self, lower_case=False, symbolic_root=False, symbolic_end=False):
        line = self.__source_file.readline()
        # skip multiple blank lines.
        while len(line) > 0 and len(line.strip()) == 0:
            line = self.__source_file.readline()
        if len(line) == 0:
            return None

        lines = []
        while len(line.strip()) > 0:
            line = line.strip()
            lines.append(line.split('\t'))
            line = self.__source_file.readline()

        length = len(lines)
        if length == 0:
            return None

        heads = []
        tokens_dict = {}
        ids_dict = {}
        for alphabet_name in self.alphabets.keys():
            tokens_dict[alphabet_name] = []
            ids_dict[alphabet_name] = []
        if symbolic_root:
            for alphabet_name, alphabet in self.alphabets.items():
                if alphabet_name.startswith('char'):
                    tokens_dict[alphabet_name].append([ROOT, ])
                    ids_dict[alphabet_name].append([alphabet.get_index(ROOT), ])
                else:
                    tokens_dict[alphabet_name].append(ROOT)
                    ids_dict[alphabet_name].append(alphabet.get_index(ROOT))
            heads.append(0)

        for tokens in lines:
            chars = []
            char_ids = []
            if lower_case:
                tokens[1] = tokens[1].lower()
            for char in tokens[1]:
                chars.append(char)
                char_ids.append(self.alphabets['char_alphabet'].get_index(char))
            if len(chars) > MAX_CHAR_LENGTH:
                chars = chars[:MAX_CHAR_LENGTH]
                char_ids = char_ids[:MAX_CHAR_LENGTH]
            tokens_dict['char_alphabet'].append(chars)
            ids_dict['char_alphabet'].append(char_ids)

            word = tokens[1]
            # print(word+ ' ')
            pos = tokens[2]
            ner = tokens[3]
            head = int(tokens[4])
            arc_tag = tokens[5]
            if len(tokens) > 6:
                auto_label = tokens[6]
                tokens_dict['auto_label_alphabet'].append(auto_label)
                ids_dict['auto_label_alphabet'].append(self.alphabets['auto_label_alphabet'].get_index(auto_label))
            tokens_dict['word_alphabet'].append(word)
            ids_dict['word_alphabet'].append(self.alphabets['word_alphabet'].get_index(word))
            tokens_dict['pos_alphabet'].append(pos)
            ids_dict['pos_alphabet'].append(self.alphabets['pos_alphabet'].get_index(pos))
            tokens_dict['ner_alphabet'].append(ner)
            ids_dict['ner_alphabet'].append(self.alphabets['ner_alphabet'].get_index(ner))
            tokens_dict['arc_alphabet'].append(arc_tag)
            ids_dict['arc_alphabet'].append(self.alphabets['arc_alphabet'].get_index(arc_tag))
            heads.append(head)

        if symbolic_end:
            for alphabet_name, alphabet in self.alphabets.items():
                if alphabet_name.startswith('char'):
                    tokens_dict[alphabet_name].append([END, ])
                    ids_dict[alphabet_name].append([alphabet.get_index(END), ])
                else:
                    tokens_dict[alphabet_name] = [END]
                    ids_dict[alphabet_name] = [alphabet.get_index(END)]
            heads.append(0)

        return NER_DependencyInstance(Sentence(tokens_dict['word_alphabet'], ids_dict['word_alphabet'],
                                               tokens_dict['char_alphabet'], ids_dict['char_alphabet']),
                                      tokens_dict, ids_dict, heads)