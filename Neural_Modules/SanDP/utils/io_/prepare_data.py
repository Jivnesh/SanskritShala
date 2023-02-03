import os.path
import numpy as np
from .alphabet import Alphabet
from .logger import get_logger
import torch

# Special vocabulary symbols - we always put them at the end.
PAD = "_<PAD>_"
ROOT = "_<ROOT>_"
END = "_<END>_"
_START_VOCAB = [PAD, ROOT, END]

MAX_CHAR_LENGTH = 45
NUM_CHAR_PAD = 2

UNK_ID = 0
PAD_ID_WORD = 1
PAD_ID_CHAR = 1
PAD_ID_TAG = 0

NUM_SYMBOLIC_TAGS = 3

_buckets = [10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 140]

from .reader import Reader

def create_alphabets(alphabet_directory, train_paths, extra_paths=None, max_vocabulary_size=100000, embedd_dict=None,
                     min_occurence=1, lower_case=False):
    def expand_vocab(vocab_list, char_alphabet, pos_alphabet, ner_alphabet, arc_alphabet):
        vocab_set = set(vocab_list)
        for data_path in extra_paths:
            with open(data_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split('\t')
                    if lower_case:
                        tokens[1] = tokens[1].lower()
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = tokens[1]
                    pos = tokens[2]
                    ner = tokens[3]
                    arc_tag = tokens[5]

                    pos_alphabet.add(pos)
                    ner_alphabet.add(ner)
                    arc_alphabet.add(arc_tag)
                    if embedd_dict is not None:
                        if word not in vocab_set and (word in embedd_dict or word.lower() in embedd_dict):
                            vocab_set.add(word)
                            vocab_list.append(word)
                    else:
                        if word not in vocab_set:
                            vocab_set.add(word)
                            vocab_list.append(word)
        return vocab_list, char_alphabet, pos_alphabet, ner_alphabet, arc_alphabet

    logger = get_logger("Create Alphabets")
    word_alphabet = Alphabet('word', defualt_value=True, singleton=True)
    char_alphabet = Alphabet('character', defualt_value=True)
    pos_alphabet = Alphabet('pos', defualt_value=True)
    ner_alphabet = Alphabet('ner', defualt_value=True)
    arc_alphabet = Alphabet('arc', defualt_value=True)
    auto_label_alphabet = Alphabet('auto_labeler', defualt_value=True)
    if not os.path.isdir(alphabet_directory):
        logger.info("Creating Alphabets: %s" % alphabet_directory)

        char_alphabet.add(PAD)
        pos_alphabet.add(PAD)
        ner_alphabet.add(PAD)
        arc_alphabet.add(PAD)
        auto_label_alphabet.add(PAD)

        char_alphabet.add(ROOT)
        pos_alphabet.add(ROOT)
        ner_alphabet.add(ROOT)
        arc_alphabet.add(ROOT)
        auto_label_alphabet.add(ROOT)

        char_alphabet.add(END)
        pos_alphabet.add(END)
        ner_alphabet.add(END)
        arc_alphabet.add(END)
        auto_label_alphabet.add(END)

        vocab = dict()
        if isinstance(train_paths, str):
            train_paths = [train_paths]
        for train_path in train_paths:
            with open(train_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue

                    tokens = line.split('\t')
                    if lower_case:
                        tokens[1] = tokens[1].lower()
                    for char in tokens[1]:
                        char_alphabet.add(char)

                    word = tokens[1]
                    # print(word)
                    pos = tokens[2]
                    ner = tokens[3]
                    arc_tag = tokens[5]

                    pos_alphabet.add(pos)
                    ner_alphabet.add(ner)
                    arc_alphabet.add(arc_tag)

                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

        # collect singletons
        singletons = set([word for word, count in vocab.items() if count <= min_occurence])

        # if a singleton is in pretrained embedding dict, set the count to min_occur + c
        if embedd_dict is not None:
            for word in vocab.keys():
                if word in embedd_dict or word.lower() in embedd_dict:
                    vocab[word] += min_occurence

        vocab_list = sorted(vocab, key=vocab.get, reverse=True)
        vocab_list = [word for word in vocab_list if vocab[word] > min_occurence]
        vocab_list = _START_VOCAB + vocab_list

        if extra_paths is not None:
            vocab_list, char_alphabet, pos_alphabet, ner_alphabet, arc_alphabet = \
                expand_vocab(vocab_list, char_alphabet, pos_alphabet, ner_alphabet, arc_alphabet)

        if len(vocab_list) > max_vocabulary_size:
            vocab_list = vocab_list[:max_vocabulary_size]

        for word in vocab_list:
            word_alphabet.add(word)
            if word in singletons:
                word_alphabet.add_singleton(word_alphabet.get_index(word))

        word_alphabet.save(alphabet_directory)
        char_alphabet.save(alphabet_directory)
        pos_alphabet.save(alphabet_directory)
        ner_alphabet.save(alphabet_directory)
        arc_alphabet.save(alphabet_directory)
        auto_label_alphabet.save(alphabet_directory)

    else:
        print('loading saved alphabet from %s' % alphabet_directory)
        word_alphabet.load(alphabet_directory)
        char_alphabet.load(alphabet_directory)
        pos_alphabet.load(alphabet_directory)
        ner_alphabet.load(alphabet_directory)
        arc_alphabet.load(alphabet_directory)
        auto_label_alphabet.load(alphabet_directory)

    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    ner_alphabet.close()
    arc_alphabet.close()
    auto_label_alphabet.close()

    alphabet_dict = {'word_alphabet': word_alphabet, 'char_alphabet': char_alphabet, 'pos_alphabet': pos_alphabet,
                     'ner_alphabet': ner_alphabet, 'arc_alphabet': arc_alphabet, 'auto_label_alphabet': auto_label_alphabet}
    return alphabet_dict

def create_alphabets_for_sequence_tagger(alphabet_directory, parser_alphabet_directory, paths):
    logger = get_logger("Create Alphabets")
    print('loading saved alphabet from %s' % parser_alphabet_directory)
    word_alphabet = Alphabet('word', defualt_value=True, singleton=True)
    char_alphabet = Alphabet('character', defualt_value=True)
    pos_alphabet = Alphabet('pos', defualt_value=True)
    ner_alphabet = Alphabet('ner', defualt_value=True)
    arc_alphabet = Alphabet('arc', defualt_value=True)
    auto_label_alphabet = Alphabet('auto_labeler', defualt_value=True)

    word_alphabet.load(parser_alphabet_directory)
    char_alphabet.load(parser_alphabet_directory)
    pos_alphabet.load(parser_alphabet_directory)
    ner_alphabet.load(parser_alphabet_directory)
    arc_alphabet.load(parser_alphabet_directory)
    try:
        auto_label_alphabet.load(alphabet_directory)
    except:
        print('Creating auto labeler alphabet')
        auto_label_alphabet.add(PAD)
        auto_label_alphabet.add(ROOT)
        auto_label_alphabet.add(END)
        for path in paths:
            with open(path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    tokens = line.split('\t')
                    if len(tokens) > 6:
                        auto_label = tokens[6]
                        auto_label_alphabet.add(auto_label)

    word_alphabet.save(alphabet_directory)
    char_alphabet.save(alphabet_directory)
    pos_alphabet.save(alphabet_directory)
    ner_alphabet.save(alphabet_directory)
    arc_alphabet.save(alphabet_directory)
    auto_label_alphabet.save(alphabet_directory)
    word_alphabet.close()
    char_alphabet.close()
    pos_alphabet.close()
    ner_alphabet.close()
    arc_alphabet.close()
    auto_label_alphabet.close()
    alphabet_dict = {'word_alphabet': word_alphabet, 'char_alphabet': char_alphabet, 'pos_alphabet': pos_alphabet,
                     'ner_alphabet': ner_alphabet, 'arc_alphabet': arc_alphabet, 'auto_label_alphabet': auto_label_alphabet}
    return alphabet_dict

def read_data(source_path, alphabets, max_size=None,
              lower_case=False, symbolic_root=False, symbolic_end=False):
    data = [[] for _ in _buckets]
    max_char_length = [0 for _ in _buckets]
    print('Reading data from %s' % ', '.join(source_path) if type(source_path) is list else source_path)
    counter = 0
    if type(source_path) is not list:
        source_path = [source_path]
    for path in source_path:
        reader = Reader(path, alphabets)
        inst = reader.getNext(lower_case=lower_case, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
        while inst is not None and (not max_size or counter < max_size):
            counter += 1
            inst_size = inst.length()
            sent = inst.sentence
            for bucket_id, bucket_size in enumerate(_buckets):
                if inst_size < bucket_size:
                    data[bucket_id].append([sent.word_ids, sent.char_id_seqs, inst.ids['pos_alphabet'], inst.ids['ner_alphabet'],
                                            inst.heads, inst.ids['arc_alphabet'], inst.ids['auto_label_alphabet']])
                    max_len = max([len(char_seq) for char_seq in sent.char_seqs])
                    if max_char_length[bucket_id] < max_len:
                        max_char_length[bucket_id] = max_len
                    break

            inst = reader.getNext(lower_case=lower_case, symbolic_root=symbolic_root, symbolic_end=symbolic_end)
        reader.close()
    print("Total number of data: %d" % counter)
    return data, max_char_length

def read_data_to_variable(source_path, alphabets, device, max_size=None,
                          lower_case=False, symbolic_root=False, symbolic_end=False):
    data, max_char_length = read_data(source_path, alphabets,
                                      max_size=max_size, lower_case=lower_case,
                                      symbolic_root=symbolic_root, symbolic_end=symbolic_end)
    # print(data)
    bucket_sizes = [len(data[b]) for b in range(len(_buckets))]

    data_variable = []

    for bucket_id in range(len(_buckets)):
        bucket_size = bucket_sizes[bucket_id]
        if bucket_size <= 0:
            data_variable.append((1, 1))
            continue

        bucket_length = _buckets[bucket_id]
        char_length = min(MAX_CHAR_LENGTH, max_char_length[bucket_id] + NUM_CHAR_PAD)
        wid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        cid_inputs = np.empty([bucket_size, bucket_length, char_length], dtype=np.int64)
        pid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        nid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        hid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        aid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)
        mid_inputs = np.empty([bucket_size, bucket_length], dtype=np.int64)

        masks = np.zeros([bucket_size, bucket_length], dtype=np.float32)
        single = np.zeros([bucket_size, bucket_length], dtype=np.int64)

        lengths = np.empty(bucket_size, dtype=np.int64)

        for i, inst in enumerate(data[bucket_id]):

            wids, cid_seqs, pids, nids, hids, aids, mids = inst
            # print(wids)
            inst_size = len(wids)
            lengths[i] = inst_size
            # word ids
            wid_inputs[i, :inst_size] = wids
            wid_inputs[i, inst_size:] = PAD_ID_WORD
            for c, cids in enumerate(cid_seqs):
                cid_inputs[i, c, :len(cids)] = cids
                cid_inputs[i, c, len(cids):] = PAD_ID_CHAR
            cid_inputs[i, inst_size:, :] = PAD_ID_CHAR
            # pos ids
            pid_inputs[i, :inst_size] = pids
            pid_inputs[i, inst_size:] = PAD_ID_TAG
            # ner ids
            nid_inputs[i, :inst_size] = nids
            nid_inputs[i, inst_size:] = PAD_ID_TAG
            # arc ids
            aid_inputs[i, :inst_size] = aids
            aid_inputs[i, inst_size:] = PAD_ID_TAG
            # auto_label ids
            mid_inputs[i, :inst_size] = mids
            mid_inputs[i, inst_size:] = PAD_ID_TAG
            # heads
            hid_inputs[i, :inst_size] = hids
            hid_inputs[i, inst_size:] = PAD_ID_TAG
            # masks
            masks[i, :inst_size] = 1.0
            for j, wid in enumerate(wids):
                if alphabets['word_alphabet'].is_singleton(wid):
                    single[i, j] = 1

        words = torch.LongTensor(wid_inputs)
        chars = torch.LongTensor(cid_inputs)
        pos = torch.LongTensor(pid_inputs)
        ner = torch.LongTensor(nid_inputs)
        heads = torch.LongTensor(hid_inputs)
        arc = torch.LongTensor(aid_inputs)
        auto_label = torch.LongTensor(mid_inputs)
        masks = torch.FloatTensor(masks)
        single = torch.LongTensor(single)
        lengths = torch.LongTensor(lengths)
        words = words.to(device)
        chars = chars.to(device)
        pos = pos.to(device)
        ner = ner.to(device)
        heads = heads.to(device)
        arc = arc.to(device)
        auto_label = auto_label.to(device)
        masks = masks.to(device)
        single = single.to(device)
        lengths = lengths.to(device)
        # print(ner)

        data_variable.append((words, chars, pos, ner, heads, arc, auto_label, masks, single, lengths))

    return data_variable, bucket_sizes

def iterate_batch(data, batch_size, device, unk_replace=0.0, shuffle=False):
    data_variable, bucket_sizes = data

    bucket_indices = np.arange(len(_buckets))
    if shuffle:
        np.random.shuffle((bucket_indices))

    for bucket_id in bucket_indices:
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]
        if bucket_size <= 0:
            continue

        words, chars, pos, ner, heads, arc, auto_label, masks, single, lengths = data_variable[bucket_id]
        if unk_replace:
            ones = single.data.new(bucket_size, bucket_length).fill_(1)
            noise = masks.data.new(bucket_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single * noise)

        indices = None
        if shuffle:
            indices = torch.randperm(bucket_size).long()
            indices = indices.to(device)
        for start_idx in range(0, bucket_size, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield words[excerpt], chars[excerpt], pos[excerpt], ner[excerpt], heads[excerpt], arc[excerpt], auto_label[excerpt], \
                  masks[excerpt], lengths[excerpt]

def iterate_batch_rand_bucket_choosing(data, batch_size, device, unk_replace=0.0):
    data_variable, bucket_sizes = data
    indices_left = [set(np.arange(bucket_size)) for bucket_size in bucket_sizes]
    while sum(bucket_sizes) > 0:
        non_empty_buckets = [i for i, bucket_size in enumerate(bucket_sizes) if bucket_size > 0]
        bucket_id = np.random.choice(non_empty_buckets)
        bucket_size = bucket_sizes[bucket_id]
        bucket_length = _buckets[bucket_id]

        words, chars, pos, ner, heads, arc, auto_label, masks, single, lengths = data_variable[bucket_id]
        min_batch_size = min(bucket_size, batch_size)
        indices = torch.LongTensor(np.random.choice(list(indices_left[bucket_id]), min_batch_size, replace=False))
        set_indices = set(indices.numpy())
        indices_left[bucket_id] = indices_left[bucket_id].difference(set_indices)
        indices = indices.to(device)
        words = words[indices]
        if unk_replace:
            ones = single.data.new(min_batch_size, bucket_length).fill_(1)
            noise = masks.data.new(min_batch_size, bucket_length).bernoulli_(unk_replace).long()
            words = words * (ones - single[indices] * noise)
        bucket_sizes = [len(s) for s in indices_left]
        yield words, chars[indices], pos[indices], ner[indices], heads[indices], arc[indices], auto_label[indices], masks[indices], lengths[indices]


def calc_num_batches(data, batch_size):
    _, bucket_sizes = data
    bucket_sizes_mod_batch_size = [int(bucket_size / batch_size) + 1 if bucket_size > 0 else 0 for bucket_size in bucket_sizes]
    num_batches = sum(bucket_sizes_mod_batch_size)
    return num_batches