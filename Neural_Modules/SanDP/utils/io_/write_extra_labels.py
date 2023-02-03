import os
from .prepare_data import ROOT, END
import pdb
def get_split(path):
    if 'train' in path:
        if 'extra_train' in path:
            split = 'extra_train'
        else:
            split = 'train'
    elif 'dev' in path:
        if 'extra_dev' in path:
            split = 'extra_dev'
        else:
            split = 'dev'
    else:
        split = 'test'
    return split

def add_number_of_children(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and 'poetry' not in file and 'prose' not in file and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if file.endswith("gold.txt") and 'poetry' not in file and 'prose' not in file  and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if file.endswith("gold.txt") and 'poetry' not in file and 'prose' not in file and 'extra' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]

        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        number_of_children = {}
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:
            for line in file:
                # line = line.decode('utf-8')
                line = line.strip()
                # print(line)
                if len(line) == 0:
                    for idx in range(len_sent):
                        node = str(idx + 1)
                        if node not in number_of_children:
                            lines[idx].append('0')
                        else:
                            lines[idx].append(str(number_of_children[node]))
                    if len(lines) > 0:
                        tmp_root_line = root_line + [str(number_of_children['0'])]
                        sentences_list.append(tmp_root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    number_of_children = {}
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                if head not in number_of_children:
                    number_of_children[head] = 1
                else:
                    number_of_children[head] += 1
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths


def add_distance_from_the_root(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'poetry' not in file and 'prose' not in file  and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'poetry' not in file and 'prose' not in file  and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and 'poetry' not in file and 'prose' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]

        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root', '0']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        tree_dict = {'0': '0'}
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:
            for line in file:

                # line = line.decode('utf-8')
                line = line.strip()
                # print(line)
                if len(line) == 0:
                    for idx in range(len_sent):
                        depth = 1
                        node = str(idx + 1)
                        # print(tree_dict)
                        ct = 0
                        while tree_dict[node] != '0':
                            node = tree_dict[node]
                            # print(node,tree_dict[node])
                            depth += 1
                            ct+=1
                            if ct > 10:
                                depth=2
                                break
                            # print(depth)
                        lines[idx].append(str(depth))
                    if len(lines) > 0:
                        sentences_list.append(root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    tree_dict = {'0': '0'}
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                tree_dict[idx] = head
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths

def add_relative_pos_based(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    # most of the code for this function is taken from:
    # https://github.com/mstrise/dep2label/blob/master/encoding.py
    def pos_cluster(pos):
        # clustering the parts of speech
        if pos[0] == 'V':
            pos = 'VB'
        elif pos == 'NNS':
            pos = 'NN'
        elif pos == 'NNPS':
            pos = 'NNP'
        elif 'JJ' in pos:
            pos = 'JJ'
        elif pos[:2] == 'RB' or pos == 'WRB' or pos == 'RP':
            pos = 'RB'
        elif pos[:3] == 'PRP':
            pos = 'PRP'
        elif pos in ['.', ':', ',', "''", '``']:
            pos = '.'
        elif pos[0] == '-':
            pos = '-RB-'
        elif pos[:2] == 'WP':
            pos = 'WP'
        return pos

    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'poetry' not in file and 'prose' not in file  and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'poetry' not in file and 'prose' not in file  and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and 'poetry' not in file and 'prose' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]
        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root', '+0_XX']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        tree_dict = {'0': '0'}
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:
            for line in file:
                # line = line.decode('utf-8')
                line = line.strip()
                # print(line)
                if len(line) == 0:
                    for idx in range(len_sent):
                        info_of_a_word = lines[idx]
                        # head is on the right side from the word
                        head = int(info_of_a_word[4]) - 1
                        if head == -1:
                            info_about_head = root_line
                        else:
                            info_about_head = lines[head]
                        if idx < head:
                            relative_position_head = 1
                            postag_head = pos_cluster(info_about_head[2])

                            for x in range(idx + 1, head):
                                another_word = lines[x]
                                postag_word_before_head = pos_cluster(another_word[2])
                                if postag_word_before_head == postag_head:
                                    relative_position_head += 1
                            label = str(
                                "+" +
                                repr(relative_position_head) +
                                "_" +
                                postag_head)
                            lines[idx].append(label)

                        # head is on the left side from the word
                        elif idx > head:
                            relative_position_head = 1
                            postag_head = pos_cluster(info_about_head[2])
                            for x in range(head + 1, idx):
                                another_word = lines[x]
                                postag_word_before_head = pos_cluster(another_word[2])
                                if postag_word_before_head == postag_head:
                                    relative_position_head += 1
                            label = str(
                                "-" +
                                repr(relative_position_head) +
                                "_" +
                                postag_head)
                            lines[idx].append(label)
                    if len(lines) > 0:
                        sentences_list.append(root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    tree_dict = {'0': '0'}
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                tree_dict[idx] = head
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths

def add_language_model(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]

        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:
            for line in file:
                # line = line.decode('utf-8')
                line = line.strip()
                if len(line) == 0:
                    for idx in range(len_sent):
                        if idx < len_sent - 1:
                            lines[idx].append(lines[idx+1][1])
                        else:
                            lines[idx].append(END)
                    if len(lines) > 0:
                        tmp_root_line = root_line + [lines[0][1]]
                        sentences_list.append(tmp_root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths

def add_relative_TAG(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    # most of the code for this function is taken from:
    # https://github.com/mstrise/dep2label/blob/master/encoding.py
    def pos_cluster(pos):
        # clustering the parts of speech
        if pos[0] == 'V':
            pos = 'VB'
        elif pos == 'NNS':
            pos = 'NN'
        elif pos == 'NNPS':
            pos = 'NNP'
        elif 'JJ' in pos:
            pos = 'JJ'
        elif pos[:2] == 'RB' or pos == 'WRB' or pos == 'RP':
            pos = 'RB'
        elif pos[:3] == 'PRP':
            pos = 'PRP'
        elif pos in ['.', ':', ',', "''", '``']:
            pos = '.'
        elif pos[0] == '-':
            pos = '-RB-'
        elif pos[:2] == 'WP':
            pos = 'WP'
        return pos

    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]
        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root', '+0_XX']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        tree_dict = {'0': '0'}
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:

            for line in file:
                # print(line)
                # print(reading_path)
                # line = line.decode('utf-8')
                line = line.strip()
                if len(line) == 0:
                    for idx in range(len_sent):
                        info_of_a_word = lines[idx]
                        # head is on the right side from the word
                        head = int(info_of_a_word[4]) - 1
                        if head == -1:
                            info_about_head = root_line
                        else:
                            # print(len(lines), head)
                            info_about_head = lines[head]
                                
                        if idx < head:
                            relative_position_head = 1
                            tag_head = info_about_head[5]

                            for x in range(idx + 1, head):
                                another_word = lines[x]
                                tag_word_before_head = another_word[5]
                                if tag_word_before_head == tag_head:
                                    relative_position_head += 1
                            label = str(
                                "+" +
                                repr(relative_position_head) +
                                "_" +
                                tag_head)
                            lines[idx].append(label)

                        # head is on the left side from the word
                        elif idx > head:
                            relative_position_head = 1
                            tag_head = info_about_head[5]
                            for x in range(head + 1, idx):
                                another_word = lines[x]
                                tag_word_before_head = another_word[5]
                                if tag_word_before_head == tag_head:
                                    relative_position_head += 1
                            label = str(
                                "-" +
                                repr(relative_position_head) +
                                "_" +
                                tag_head)
                            lines[idx].append(label)
                    if len(lines) > 0:
                        sentences_list.append(root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    tree_dict = {'0': '0'}
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                tree_dict[idx] = head
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths


def add_head(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    # most of the code for this function is taken from:
    # https://github.com/mstrise/dep2label/blob/master/encoding.py
    def pos_cluster(pos):
        # clustering the parts of speech
        if pos[0] == 'V':
            pos = 'VB'
        elif pos == 'NNS':
            pos = 'NN'
        elif pos == 'NNPS':
            pos = 'NNP'
        elif 'JJ' in pos:
            pos = 'JJ'
        elif pos[:2] == 'RB' or pos == 'WRB' or pos == 'RP':
            pos = 'RB'
        elif pos[:3] == 'PRP':
            pos = 'PRP'
        elif pos in ['.', ':', ',', "''", '``']:
            pos = '.'
        elif pos[0] == '-':
            pos = '-RB-'
        elif pos[:2] == 'WP':
            pos = 'WP'
        return pos

    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]
        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root', '+0_XX']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        tree_dict = {'0': '0'}
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:

            for line in file:
                # print(line)
                # print(reading_path)
                # line = line.decode('utf-8')
                line = line.strip()
                if len(line) == 0:
                    for idx in range(len_sent):
                        info_of_a_word = lines[idx]
                        # head is on the right side from the word
                        head = int(info_of_a_word[4]) - 1
                        if head == -1:
                            info_about_head = root_line
                        else:
                            # print(len(lines), head)
                            info_about_head = lines[head]
                        head_word = info_about_head[1]     
                        lines[idx].append(head_word)  
                        # if idx < head:
                        #     relative_position_head = 1
                            

                        #     for x in range(idx + 1, head):
                        #         another_word = lines[x]
                        #         postag_word_before_head = pos_cluster(another_word[2])
                        #         if postag_word_before_head == postag_head:
                        #             relative_position_head += 1
                        #     label = str(
                        #         "+" +
                        #         repr(relative_position_head) +
                        #         "_" +
                        #         postag_head)
                            

                        # # head is on the left side from the word
                        # elif idx > head:
                        #     relative_position_head = 1
                        #     postag_head = pos_cluster(info_about_head[2])
                        #     for x in range(head + 1, idx):
                        #         another_word = lines[x]
                        #         postag_word_before_head = pos_cluster(another_word[2])
                        #         if postag_word_before_head == postag_head:
                        #             relative_position_head += 1
                        #     label = str(
                        #         "-" +
                        #         repr(relative_position_head) +
                        #         "_" +
                        #         postag_head)
                        #     lines[idx].append(label)
                    if len(lines) > 0:
                        sentences_list.append(root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    tree_dict = {'0': '0'}
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                tree_dict[idx] = head
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths
import json
def get_modified_coarse(ma):
    ma = ma.replace('sgpl','sg').replace('sgdu','sg')
    with open('/home/jivnesh/DCST_scratch/utils/io_/coarse_to_ma_dict.json', 'r') as fh:
        coarse_dict = json.load(fh)
    for key in coarse_dict.keys():
        if ma in coarse_dict[key]:
            return key
def add_head_coarse_pos(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    # most of the code for this function is taken from:
    # https://github.com/mstrise/dep2label/blob/master/encoding.py
    def pos_cluster(pos):
        # clustering the parts of speech
        if pos[0] == 'V':
            pos = 'VB'
        elif pos == 'NNS':
            pos = 'NN'
        elif pos == 'NNPS':
            pos = 'NNP'
        elif 'JJ' in pos:
            pos = 'JJ'
        elif pos[:2] == 'RB' or pos == 'WRB' or pos == 'RP':
            pos = 'RB'
        elif pos[:3] == 'PRP':
            pos = 'PRP'
        elif pos in ['.', ':', ',', "''", '``']:
            pos = '.'
        elif pos[0] == '-':
            pos = '-RB-'
        elif pos[:2] == 'WP':
            pos = 'WP'
        return pos

    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]
        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root', 'O']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        tree_dict = {'0': '0'}
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:

            for line in file:
                # print(line)
                # print(reading_path)
                # line = line.decode('utf-8')
                line = line.strip()
                if len(line) == 0:
                    for idx in range(len_sent):
                        info_of_a_word = lines[idx]
                        # head is on the right side from the word
                        head = int(info_of_a_word[4]) - 1
                        if head == -1:
                            info_about_head = root_line
                        else:
                            # print(len(lines), head)
                            info_about_head = lines[head]
                        postag_head = info_about_head[3]
                        lines[idx].append(postag_head)    
                    if len(lines) > 0:
                        sentences_list.append(root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    tree_dict = {'0': '0'}
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                tree_dict[idx] = head
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths

def add_head_ma(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    # most of the code for this function is taken from:
    # https://github.com/mstrise/dep2label/blob/master/encoding.py
    def pos_cluster(pos):
        # clustering the parts of speech
        if pos[0] == 'V':
            pos = 'VB'
        elif pos == 'NNS':
            pos = 'NN'
        elif pos == 'NNPS':
            pos = 'NNP'
        elif 'JJ' in pos:
            pos = 'JJ'
        elif pos[:2] == 'RB' or pos == 'WRB' or pos == 'RP':
            pos = 'RB'
        elif pos[:3] == 'PRP':
            pos = 'PRP'
        elif pos in ['.', ':', ',', "''", '``']:
            pos = '.'
        elif pos[0] == '-':
            pos = '-RB-'
        elif pos[:2] == 'WP':
            pos = 'WP'
        return pos

    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]
        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root', 'XX']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        tree_dict = {'0': '0'}
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:

            for line in file:
                # print(line)
                # print(reading_path)
                # line = line.decode('utf-8')
                line = line.strip()
                if len(line) == 0:
                    for idx in range(len_sent):
                        info_of_a_word = lines[idx]
                        # head is on the right side from the word
                        head = int(info_of_a_word[4]) - 1
                        if head == -1:
                            info_about_head = root_line
                        else:
                            # print(len(lines), head)
                            info_about_head = lines[head]
                        postag_head = pos_cluster(info_about_head[2])
                        lines[idx].append(postag_head)    
                    if len(lines) > 0:
                        sentences_list.append(root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    tree_dict = {'0': '0'}
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                tree_dict[idx] = head
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths

def add_label(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]

        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print('############ Add Label Task #################')
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:
            for line in file:
                # line = line.decode('utf-8')
                line = line.strip()
                # Now blank space got detected
                if len(line) == 0:
                    # Append next word to last column
                    for idx in range(len_sent):
                            lines[idx].append(lines[idx][5])
                    # Add root line first
                    if len(lines) > 0:
                        tmp_root_line = root_line + [root_line[5]]
                        sentences_list.append(tmp_root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths

def predict_ma_tag_of_modifier(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]

        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print('############ Add Label Task #################')
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:
            for line in file:
                # line = line.decode('utf-8')
                line = line.strip()
                # Now blank space got detected
                if len(line) == 0:
                    # Append next word to last column
                    for idx in range(len_sent):
                            lines[idx].append(clean_ma(lines[idx][3]))
                    # Add root line first
                    if len(lines) > 0:
                        tmp_root_line = root_line + [root_line[3]]
                        sentences_list.append(tmp_root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths

def predict_coarse_of_modifier(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]

        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print('############ Add Label Task #################')
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:
            for line in file:
                # line = line.decode('utf-8')
                line = line.strip()
                # Now blank space got detected
                if len(line) == 0:
                    # Append next word to last column
                    for idx in range(len_sent):
                            lines[idx].append(lines[idx][3])
                    # Add root line first
                    if len(lines) > 0:
                        tmp_root_line = root_line + [root_line[3]]
                        sentences_list.append(tmp_root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths
import re
def get_case(ma):
        indeclinable = ['ind','prep','interj','prep','conj','part']
        case_list = ['nom','voc','acc','i','inst','dat','abl','g','loc']
        gender_list = ['n','f','m','*']
        person_list = ['1','2','3']
        no_list = ['du','sg','pl']
        pops = [' ac',' ps']
        ma=ma.replace('sgpl','sg').replace('sgdu','sg')
        temp = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", ma).replace('[] ','').strip(' []')
        temp = temp.split('.')
        if temp[-1] == '':
            temp.pop(-1)
        # Remove active passive
        case=''
        no=''
        person=''
        gender=''
        tense=''
        coarse=''
        for a,b in enumerate(temp):
            if b in pops:
                temp.pop(a)
        # Get gender
        for a,b in enumerate(temp):
            if b.strip() in gender_list:
                gender = b.strip()
                temp.pop(a)
        # Get case
        for a,b in enumerate(temp):
            if b.strip() in case_list:
                case = b.strip()
                temp.pop(a)
        if case!= '':
            coarse ='Noun'
        # Get person
        for a,b in enumerate(temp):
            if b.strip() in person_list:
                person = b.strip()
                temp.pop(a)
        # Get no
        for a,b in enumerate(temp):
            if b.strip() in no_list:
                no = b.strip()
                temp.pop(a)
        # Get Tense
        for b in temp:
            tense=tense+ ' '+b.strip()
        tense=tense.strip()
        
#         print(tense)
        if tense == 'adv':
            coarse = 'adv'
        for ind in indeclinable:
            if tense == ind:
                coarse = 'Ind'
        if tense == 'abs' or tense == 'ca abs':
            coarse = 'IV'
        if tense!='' and coarse=='':
            if person !='' or no!='':
                coarse= 'FV'
            else:
                coarse = 'IV'
        if case == 'i':
            return 'inst'
        
        if case !='':
            return case
        else:
            return coarse
def clean_ma(ma):
    ma = re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", ma).replace('[] ','').strip(' []').replace(' ac','').replace(' ps','').replace('sgpl','sg').replace('sgdu','sg')
    ma = ma.replace('i.','inst.').replace('.','').replace(' ','')
    return ma
def predict_case_of_modifier(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    if src_domain == tgt_domain:
        pred_paths = []
        if use_unlabeled_data:
            pred_paths = [file for file in os.listdir(parser_path) if
                          file.endswith("pred.txt") and 'extra' in file and tgt_domain in file]

        gold_paths = [file for file in os.listdir(parser_path) if
                      file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' not in file]
        if use_labeled_data:
            gold_paths += [file for file in os.listdir(parser_path) if
                           file.endswith("gold.txt") and 'extra' not in file and tgt_domain in file and 'train' in file]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError
    else:
        pred_paths = [file for file in os.listdir(parser_path) if file.endswith("pred.txt") and tgt_domain in file]

        gold_paths = []
        if use_labeled_data:
            gold_paths = ['data/onto_pos_ner_dp_train_' + src_domain]

        if not use_unlabeled_data and not use_labeled_data:
            raise ValueError

    paths = pred_paths + gold_paths
    print('############ Add Label Task #################')
    print("Adding labels to paths: %s" % ', '.join(paths))
    root_line = ['0', ROOT, 'XX', 'O', '0', 'root']
    writing_paths = {}
    sentences = {}
    for path in paths:
        if tgt_domain in path:
            reading_path = parser_path + path
            writing_path = model_path + 'parser_' + path
            split = get_split(writing_path)
        else:
            reading_path = path
            writing_path = model_path + 'parser_' + 'domain_' + src_domain + '_train_model_domain_' + src_domain + '_data_domain_' + src_domain + '_gold.txt'
            split = 'extra_train'
        writing_paths[split] = writing_path
        len_sent = 0
        lines = []
        sentences_list = []
        with open(reading_path, 'r') as file:
            for line in file:
                # line = line.decode('utf-8')
                line = line.strip()
                # Now blank space got detected
                if len(line) == 0:
                    # Append next word to last column
                    for idx in range(len_sent):
                            lines[idx].append(get_case(lines[idx][3]))
                    # Add root line first
                    if len(lines) > 0:
                        tmp_root_line = root_line + [root_line[3]]
                        sentences_list.append(tmp_root_line)
                    for line_ in lines:
                        sentences_list.append(line_)
                    sentences_list.append([])
                    lines = []
                    len_sent = 0
                    continue
                tokens = line.split('\t')
                idx = tokens[0]
                word = tokens[1]
                pos = tokens[2]
                ner = tokens[3]
                head = tokens[4]
                arc_tag = tokens[5]
                lines.append([idx, word, pos, ner, head, arc_tag])
                len_sent += 1
        sentences[split] = sentences_list

    train_sentences = []
    if 'train' in sentences:
        train_sentences = sentences['train']
    else:
        writing_paths['train'] = writing_paths['extra_train'].replace('extra_train', 'train')
    if 'extra_train' in sentences:
        train_sentences += sentences['extra_train']
        del writing_paths['extra_train']
    if 'extra_dev' in sentences:
        train_sentences += sentences['extra_dev']
        del writing_paths['extra_dev']
    with open(writing_paths['train'], 'w') as f:
        for sent in train_sentences:
            f.write('\t'.join(sent) + '\n')
    for split in ['dev', 'test']:
        if split in sentences:
            split_sentences = sentences[split]
            with open(writing_paths[split], 'w') as f:
                for sent in split_sentences:
                    f.write('\t'.join(sent) + '\n')
    return writing_paths

def Multitask_case_predict(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    writing_paths = {}
    # multitask_silver_20ktrain_san
    writing_paths['train'] = 'data/Multitask_case_train_'+src_domain
    writing_paths['dev'] = 'data/Multitask_case_dev_'+src_domain
    writing_paths['test'] = 'data/Multitask_case_test_'+src_domain
    return writing_paths

def Multitask_POS_predict(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    writing_paths = {}
    # multitask_silver_20ktrain_san
    # writing_paths['train'] = 'data/Multitask_POS_predict_train_san'
    # writing_paths['dev'] = 'data/Multitask_POS_predict_dev_san'
    # writing_paths['test'] = 'data/Multitask_POS_predict_test_san'
    writing_paths['train'] = 'data/Multitask_morph_train_'+src_domain
    writing_paths['dev'] = 'data/Multitask_morph_dev_'+src_domain
    writing_paths['test'] = 'data/Multitask_morph_test_'+src_domain
    return writing_paths

def Multitask_coarse_predict(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    writing_paths = {}
    # multitask_silver_20ktrain_san
    writing_paths['train'] = 'data/Multitask_coarse_predict_train_'+src_domain
    writing_paths['dev'] = 'data/Multitask_coarse_predict_dev_'+src_domain
    writing_paths['test'] = 'data/Multitask_coarse_predict_test_'+src_domain
    return writing_paths

def Multitask_label_predict(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    writing_paths = {}
    # multitask_silver_20ktrain_san
    writing_paths['train'] = 'data/Multitask_label_train_'+src_domain
    writing_paths['dev'] = 'data/Multitask_label_dev_'+src_domain
    writing_paths['test'] = 'data/Multitask_label_test_'+src_domain
    return writing_paths
###################################

def MRL_case(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    writing_paths = {}
    # multitask_silver_20ktrain_san
    writing_paths['train'] = 'data/Prep_MRL/ud_pos_ner_dp_train_'+src_domain+'_case'
    writing_paths['dev'] = 'data/Prep_MRL/ud_pos_ner_dp_dev_'+src_domain+'_case'
    writing_paths['test'] = 'data/Prep_MRL/ud_pos_ner_dp_test_'+src_domain+'_case'
    return writing_paths

def MRL_POS(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    writing_paths = {}
    # multitask_silver_20ktrain_san
    writing_paths['train'] = '/media/guest/rathin_workspace/TransLamp-Jivnesh/wsmp_dataset/mytask2_train_morph'
    writing_paths['dev'] = '/media/guest/rathin_workspace/TransLamp-Jivnesh/wsmp_dataset/mytask2_dev_morph'
    writing_paths['test'] = '/media/guest/rathin_workspace/TransLamp-Jivnesh/wsmp_dataset/mytask2_test'
    return writing_paths

def MRL_label(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    writing_paths = {}
    # multitask_silver_20ktrain_san
    writing_paths['train'] = '/media/guest/rathin_workspace/TransLamp-Jivnesh/wsmp_dataset/mytask2_train_lemma'
    writing_paths['dev'] = '/media/guest/rathin_workspace/TransLamp-Jivnesh/wsmp_dataset/mytask2_dev_lemma'
    writing_paths['test'] = '/media/guest/rathin_workspace/TransLamp-Jivnesh/wsmp_dataset/mytask2_test'
    return writing_paths

def MRL_no(model_path, parser_path, src_domain, tgt_domain, use_unlabeled_data=True, use_labeled_data=True):
    writing_paths = {}
    # multitask_silver_20ktrain_san
    writing_paths['train'] = 'data/Prep_MRL/ud_pos_ner_dp_train_'+src_domain+'_no'
    writing_paths['dev'] = 'data/Prep_MRL/ud_pos_ner_dp_dev_'+src_domain+'_no'
    writing_paths['test'] = 'data/Prep_MRL/ud_pos_ner_dp_test_'+src_domain+'_no'
    return writing_paths