import numpy as np

def read_file(filename):
    sentences = []
    sentence = []
    lengths = []
    num_sentneces_to_remove = 0
    num_sentences = 0
    num_tokens_to_remove = 0
    num_tokens = 0
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                xx_count = 0
                for row in sentence:
                    if row[2] == 'XX':
                        xx_count += 1
                if xx_count / len(sentence) >= 0.5:
                    num_sentneces_to_remove += 1
                    num_tokens_to_remove += len(sentence)
                else:
                    sentences.append(sentence)
                    lengths.append(len(sentence))
                num_sentences += 1
                num_tokens += len(sentence)
                sentence = []
                continue
            tokens = line.split('\t')
            idx = tokens[0]
            word = tokens[1]
            pos = tokens[2]
            ner = tokens[3]
            arc = tokens[4]
            arc_tag = tokens[5]
            sentence.append((idx, word, pos, ner, arc, arc_tag))
    print("removed %d sentences out of %d sentences" % (num_sentneces_to_remove, num_sentences))
    print("removed %d tokens out of %d tokens" % (num_tokens_to_remove, num_tokens))
    return sentences

def write_file(filename, sentences):
    with open(filename, 'w') as file:
        for sentence in sentences:
            for row in sentence:
                file.write('\t'.join([token for token in row]) + '\n')
            file.write('\n')

dataset_dict = {'ontonotes': 'onto'}
datasets = ['ontonotes']
splits = ['test']
domains = ['all', 'wb']

for dataset in datasets:
    for domain in domains:
        for split in splits:
            print('dataset: %s, domain: %s, split: %s' % (dataset, domain, split))
            filemame = 'data/'+ dataset_dict[dataset] + '_pos_ner_dp_' + split + '_' + domain
            sentences = read_file(filemame)
            write_filename = filemame + '_without_xx'
            write_file(write_filename, sentences)
