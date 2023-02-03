from __future__ import print_function
import sys
from os import path, makedirs, system, remove

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import uuid
import json
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD
from utils.io_ import seeds, Writer, get_logger, Index2Instance, prepare_data, write_extra_labels
from utils.models.sequence_tagger import Sequence_Tagger
from utils import load_word_embeddings
from utils.tasks.seqeval import accuracy_score, f1_score, precision_score, recall_score,classification_report

uid = uuid.uuid4().hex[:6]

logger = get_logger('SequenceTagger')

def read_arguments():
    args_ = argparse.ArgumentParser(description='Sovling SequenceTagger')
    args_.add_argument('--dataset', choices=['ontonotes', 'ud'], help='Dataset', required=True)
    args_.add_argument('--domain', help='domain', required=True)
    args_.add_argument('--rnn_mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn',
                       required=True)
    args_.add_argument('--task', default='distance_from_the_root', choices=['distance_from_the_root', 'number_of_children',\
     'relative_pos_based', 'language_model','add_label','add_head_coarse_pos','Multitask_POS_predict','Multitask_case_predict',\
     'Multitask_label_predict','Multitask_coarse_predict','MRL_case','MRL_POS','MRL_no','MRL_label',\
     'predict_coarse_of_modifier','predict_ma_tag_of_modifier','add_head_ma','predict_case_of_modifier'], help='sequence_tagger task')
    args_.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_.add_argument('--tag_space', type=int, default=128, help='Dimension of tag space')
    args_.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    args_.add_argument('--num_filters', type=int, default=50, help='Number of filters in CNN')
    args_.add_argument('--kernel_size', type=int, default=3, help='Size of Kernel for CNN')
    args_.add_argument('--use_pos', action='store_true', help='use part-of-speech embedding.')
    args_.add_argument('--use_char', action='store_true', help='use character embedding and CNN.')
    args_.add_argument('--word_dim', type=int, default=300, help='Dimension of word embeddings')
    args_.add_argument('--pos_dim', type=int, default=50, help='Dimension of POS embeddings')
    args_.add_argument('--char_dim', type=int, default=50, help='Dimension of Character embeddings')
    args_.add_argument('--initializer', choices=['xavier'], help='initialize model parameters')
    args_.add_argument('--opt', choices=['adam', 'sgd'], help='optimization algorithm')
    args_.add_argument('--momentum', type=float, default=0.9, help='momentum of optimizer')
    args_.add_argument('--betas', nargs=2, type=float, default=[0.9, 0.9], help='betas of optimizer')
    args_.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    args_.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
    args_.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    args_.add_argument('--clip', type=float, default=5.0, help='gradient clipping')
    args_.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    args_.add_argument('--epsilon', type=float, default=1e-8, help='epsilon for adam')
    args_.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    args_.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    args_.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    args_.add_argument('--unk_replace', type=float, default=0.,
                       help='The rate to replace a singleton word with UNK')
    args_.add_argument('--punct_set', nargs='+', type=str, help='List of punctuations')
    args_.add_argument('--word_embedding', choices=['random', 'glove', 'fasttext', 'word2vec'],
                       help='Embedding for words')
    args_.add_argument('--word_path', help='path for word embedding dict - in case word_embedding is not random')
    args_.add_argument('--freeze_word_embeddings', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_.add_argument('--char_embedding', choices=['random','hellwig'], help='Embedding for characters',
                       required=True)
    args_.add_argument('--pos_embedding', choices=['random','one_hot'], help='Embedding for pos',
                       required=True)
    args_.add_argument('--char_path', help='path for character embedding dict')
    args_.add_argument('--pos_path', help='path for pos embedding dict')
    args_.add_argument('--use_unlabeled_data', action='store_true', help='flag to use unlabeled data.')
    args_.add_argument('--use_labeled_data', action='store_true', help='flag to use labeled data.')
    args_.add_argument('--model_path', help='path for saving model file.', required=True)
    args_.add_argument('--parser_path', help='path for loading parser files.', default=None)
    args_.add_argument('--load_path', help='path for loading saved source model file.', default=None)
    args_.add_argument('--strict',action='store_true', help='if True loaded model state should contain '
                                                            'exactly the same keys as current model')
    args_.add_argument('--eval_mode', action='store_true', help='evaluating model without training it')
    args = args_.parse_args()
    args_dict = {}
    args_dict['dataset'] = args.dataset
    args_dict['domain'] = args.domain
    args_dict['task'] = args.task
    args_dict['rnn_mode'] = args.rnn_mode
    args_dict['load_path'] = args.load_path
    args_dict['strict'] = args.strict
    args_dict['model_path'] = args.model_path
    if not path.exists(args_dict['model_path']):
        makedirs(args_dict['model_path'])
    args_dict['parser_path'] = args.parser_path
    args_dict['model_name'] = 'domain_' + args_dict['domain']
    args_dict['full_model_name'] = path.join(args_dict['model_path'],args_dict['model_name'])
    args_dict['use_unlabeled_data'] = args.use_unlabeled_data
    args_dict['use_labeled_data'] = args.use_labeled_data
    print(args_dict['parser_path'])
    if args_dict['task'] == 'number_of_children':
        args_dict['data_paths'] = write_extra_labels.add_number_of_children(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                            use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                            use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'distance_from_the_root':
        args_dict['data_paths'] = write_extra_labels.add_distance_from_the_root(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'Multitask_label_predict':
        args_dict['data_paths'] = write_extra_labels.Multitask_label_predict(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'Multitask_coarse_predict':
        args_dict['data_paths'] = write_extra_labels.Multitask_coarse_predict(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'Multitask_POS_predict':
        args_dict['data_paths'] = write_extra_labels.Multitask_POS_predict(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'relative_pos_based':
        args_dict['data_paths'] = write_extra_labels.add_relative_pos_based(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'add_label':
        args_dict['data_paths'] = write_extra_labels.add_label(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'add_relative_TAG':
        args_dict['data_paths'] = write_extra_labels.add_relative_TAG(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'add_head_coarse_pos':
        args_dict['data_paths'] = write_extra_labels.add_head_coarse_pos(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'predict_ma_tag_of_modifier':
        args_dict['data_paths'] = write_extra_labels.predict_ma_tag_of_modifier(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'Multitask_case_predict':
        args_dict['data_paths'] = write_extra_labels.Multitask_case_predict(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'predict_coarse_of_modifier':
        args_dict['data_paths'] = write_extra_labels.predict_coarse_of_modifier(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'predict_case_of_modifier':
        args_dict['data_paths'] = write_extra_labels.predict_case_of_modifier(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'add_head_ma':
        args_dict['data_paths'] = write_extra_labels.add_head_ma(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'MRL_case':
        args_dict['data_paths'] = write_extra_labels.MRL_case(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'MRL_POS':
        args_dict['data_paths'] = write_extra_labels.MRL_POS(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'MRL_no':
        args_dict['data_paths'] = write_extra_labels.MRL_no(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    elif args_dict['task'] == 'MRL_label':
        args_dict['data_paths'] = write_extra_labels.MRL_label(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                                                 use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                                                 use_labeled_data=args_dict['use_labeled_data'])
    else: #args_dict['task'] == 'language_model':
        args_dict['data_paths'] = write_extra_labels.add_language_model(args_dict['model_path'], args_dict['parser_path'], args_dict['domain'], args_dict['domain'],
                                                            use_unlabeled_data=args_dict['use_unlabeled_data'],
                                                            use_labeled_data=args_dict['use_labeled_data'])
    args_dict['splits'] = args_dict['data_paths'].keys()
    alphabet_data_paths = deepcopy(args_dict['data_paths'])
    if args_dict['dataset'] == 'ontonotes':
        data_path = 'data/onto_pos_ner_dp'
    else:
        data_path = 'data/ud_pos_ner_dp'
    # Adding more resources to make sure equal alphabet size for all domains
    for split in args_dict['splits']:
        if args_dict['dataset'] == 'ontonotes':
            alphabet_data_paths['additional_' + split] = data_path + '_' + split + '_' + 'all'
        else:
            if '_' in args_dict['domain']:
                alphabet_data_paths[split] = data_path + '_' + split + '_' + args_dict['domain'].split('_')[0]
            else:
                alphabet_data_paths[split] = args_dict['data_paths'][split]
    args_dict['alphabet_data_paths'] = alphabet_data_paths
    args_dict['num_epochs'] = args.num_epochs
    args_dict['batch_size'] = args.batch_size
    args_dict['hidden_size'] = args.hidden_size
    args_dict['tag_space'] = args.tag_space
    args_dict['num_layers'] = args.num_layers
    args_dict['num_filters'] = args.num_filters
    args_dict['kernel_size'] = args.kernel_size
    args_dict['learning_rate'] = args.learning_rate
    args_dict['initializer'] = nn.init.xavier_uniform_ if args.initializer == 'xavier' else None
    args_dict['opt'] = args.opt
    args_dict['momentum'] = args.momentum
    args_dict['betas'] = tuple(args.betas)
    args_dict['epsilon'] = args.epsilon
    args_dict['decay_rate'] = args.decay_rate
    args_dict['clip'] = args.clip
    args_dict['gamma'] = args.gamma
    args_dict['schedule'] = args.schedule
    args_dict['p_rnn'] = tuple(args.p_rnn)
    args_dict['p_in'] = args.p_in
    args_dict['p_out'] = args.p_out
    args_dict['unk_replace'] = args.unk_replace
    args_dict['punct_set'] = None
    if args.punct_set is not None:
        args_dict['punct_set'] = set(args.punct_set)
        logger.info("punctuations(%d): %s" % (len(args_dict['punct_set']), ' '.join(args_dict['punct_set'])))
    args_dict['freeze_word_embeddings'] = args.freeze_word_embeddings
    args_dict['word_embedding'] = args.word_embedding
    args_dict['word_path'] = args.word_path
    args_dict['use_char'] = args.use_char
    args_dict['char_embedding'] = args.char_embedding
    args_dict['pos_embedding'] = args.pos_embedding
    args_dict['char_path'] = args.char_path
    args_dict['pos_path'] = args.pos_path
    args_dict['use_pos'] = args.use_pos
    args_dict['pos_dim'] = args.pos_dim
    args_dict['word_dict'] = None
    args_dict['word_dim'] = args.word_dim
    if args_dict['word_embedding'] != 'random' and args_dict['word_path']:
        args_dict['word_dict'], args_dict['word_dim'] = load_word_embeddings.load_embedding_dict(args_dict['word_embedding'],
                                                                                                 args_dict['word_path'])
    args_dict['char_dict'] = None
    args_dict['char_dim'] = args.char_dim
    if args_dict['char_embedding'] != 'random':
        args_dict['char_dict'], args_dict['char_dim'] = load_word_embeddings.load_embedding_dict(args_dict['char_embedding'],
                                                                                                 args_dict['char_path'])
    args_dict['pos_dict'] = None
    if args_dict['pos_embedding'] != 'random':
        args_dict['pos_dict'], args_dict['pos_dim'] = load_word_embeddings.load_embedding_dict(args_dict['pos_embedding'],
                                                                                                 args_dict['pos_path'])
    args_dict['alphabet_path'] = path.join(args_dict['model_path'], 'alphabets' + '_src_domain_' + args_dict['domain'] + '/')
    args_dict['alphabet_parser_path'] = path.join(args_dict['parser_path'], 'alphabets' + '_src_domain_' + args_dict['domain'] + '/')
    args_dict['model_name'] = path.join(args_dict['model_path'], args_dict['model_name'])
    args_dict['eval_mode'] = args.eval_mode
    args_dict['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args_dict['word_status'] = 'frozen' if args.freeze_word_embeddings else 'fine tune'
    args_dict['char_status'] = 'enabled' if args.use_char else 'disabled'
    args_dict['pos_status'] = 'enabled' if args.use_pos else 'disabled'
    logger.info("Saving arguments to file")
    save_args(args, args_dict['full_model_name'])
    logger.info("Creating Alphabets")
    alphabet_dict = creating_alphabets(args_dict['alphabet_path'], args_dict['alphabet_parser_path'], args_dict['alphabet_data_paths'])
    args_dict = {**args_dict, **alphabet_dict}
    ARGS = namedtuple('ARGS', args_dict.keys())
    my_args = ARGS(**args_dict)
    return my_args


def creating_alphabets(alphabet_path, alphabet_parser_path, alphabet_data_paths):
    data_paths_list = alphabet_data_paths.values()
    alphabet_dict = {}
    alphabet_dict['alphabets'] = prepare_data.create_alphabets_for_sequence_tagger(alphabet_path, alphabet_parser_path, data_paths_list)
    for k, v in alphabet_dict['alphabets'].items():
        num_key = 'num_' + k.split('_alphabet')[0]
        alphabet_dict[num_key] = v.size()
        logger.info("%s : %d" % (num_key, alphabet_dict[num_key]))
    return alphabet_dict

def construct_embedding_table(alphabet, tokens_dict, dim, token_type='word'):
    if tokens_dict is None:
        return None
    scale = np.sqrt(3.0 / dim)
    table = np.empty([alphabet.size(), dim], dtype=np.float32)
    table[prepare_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, dim]).astype(np.float32)
    oov_tokens = 0
    for token, index in alphabet.items():
        if token in tokens_dict:
            embedding = tokens_dict[token]
        elif token.lower() in tokens_dict:
            embedding = tokens_dict[token.lower()]
        else:
            embedding = np.random.uniform(-scale, scale, [1, dim]).astype(np.float32)
            oov_tokens += 1
        table[index, :] = embedding
    print('token type : %s, number of oov: %d' % (token_type, oov_tokens))
    table = torch.from_numpy(table)
    return table

def get_free_gpu():
    system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp.txt')
    memory_available = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
    remove("tmp.txt")
    free_device = 'cuda:' + str(np.argmax(memory_available))
    return free_device

def save_args(args, full_model_name):
    arg_path = full_model_name + '.arg.json'
    argparse_dict = vars(args)
    with open(arg_path, 'w') as f:
        json.dump(argparse_dict, f)

def generate_optimizer(args, lr, params):
    params = filter(lambda param: param.requires_grad, params)
    if args.opt == 'adam':
        return Adam(params, lr=lr, betas=args.betas, weight_decay=args.gamma, eps=args.epsilon)
    elif args.opt == 'sgd':
        return SGD(params, lr=lr, momentum=args.momentum, weight_decay=args.gamma, nesterov=True)
    else:
        raise ValueError('Unknown optimization algorithm: %s' % args.opt)


def save_checkpoint(model, optimizer, opt, dev_eval_dict, test_eval_dict, full_model_name):
    path_name = full_model_name + '.pt'
    print('Saving model to: %s' % path_name)
    state = {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'opt': opt, 'dev_eval_dict': dev_eval_dict, 'test_eval_dict': test_eval_dict}
    torch.save(state, path_name)


def load_checkpoint(args, model, optimizer, dev_eval_dict, test_eval_dict, start_epoch, load_path, strict=True):
    print('Loading saved model from: %s' % load_path)
    checkpoint = torch.load(load_path, map_location=args.device)
    if checkpoint['opt'] != args.opt:
        raise ValueError('loaded optimizer type is: %s instead of: %s' % (checkpoint['opt'], args.opt))
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    if strict:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        dev_eval_dict = checkpoint['dev_eval_dict']
        test_eval_dict = checkpoint['test_eval_dict']
        start_epoch = dev_eval_dict['in_domain']['epoch']
    return model, optimizer, dev_eval_dict, test_eval_dict, start_epoch


def build_model_and_optimizer(args):
    word_table = construct_embedding_table(args.alphabets['word_alphabet'], args.word_dict, args.word_dim, token_type='word')
    char_table = construct_embedding_table(args.alphabets['char_alphabet'], args.char_dict, args.char_dim, token_type='char')
    pos_table = construct_embedding_table(args.alphabets['pos_alphabet'], args.pos_dict, args.pos_dim, token_type='pos')
    model = Sequence_Tagger(args.word_dim, args.num_word, args.char_dim, args.num_char,
                            args.use_pos, args.use_char, args.pos_dim, args.num_pos,
                            args.num_filters, args.kernel_size, args.rnn_mode,
                            args.hidden_size, args.num_layers, args.tag_space, args.num_auto_label,
                            embedd_word=word_table, embedd_char=char_table, embedd_pos=pos_table,
                            p_in=args.p_in, p_out=args.p_out, p_rnn=args.p_rnn,
                            initializer=args.initializer)
    optimizer = generate_optimizer(args, args.learning_rate, model.parameters())
    start_epoch = 0
    dev_eval_dict = {'in_domain': initialize_eval_dict()}
    test_eval_dict = {'in_domain': initialize_eval_dict()}
    if args.load_path:
        model, optimizer, dev_eval_dict, test_eval_dict, start_epoch = \
            load_checkpoint(args, model, optimizer,
                            dev_eval_dict, test_eval_dict,
                            start_epoch, args.load_path, strict=args.strict)
    if args.freeze_word_embeddings:
        model.rnn_encoder.word_embedd.weight.requires_grad = False
        # model.rnn_encoder.char_embedd.weight.requires_grad = False
        # model.rnn_encoder.pos_embedd.weight.requires_grad = False
    device = args.device
    model.to(device)
    return model, optimizer, dev_eval_dict, test_eval_dict, start_epoch


def initialize_eval_dict():
    eval_dict = {}
    eval_dict['auto_label_accuracy'] = 0.0
    eval_dict['auto_label_precision'] = 0.0
    eval_dict['auto_label_recall'] = 0.0
    eval_dict['auto_label_f1'] = 0.0
    return eval_dict

def in_domain_evaluation(args, datasets, model, optimizer, dev_eval_dict, test_eval_dict, epoch,
                         best_model, best_optimizer, patient):
    # In-domain evaluation
    curr_dev_eval_dict = evaluation(args, datasets['dev'], 'dev', model, args.domain, epoch, 'current_results')
    is_best_in_domain = dev_eval_dict['in_domain']['auto_label_f1'] <= curr_dev_eval_dict['auto_label_f1']

    if is_best_in_domain:
        for key, value in curr_dev_eval_dict.items():
            dev_eval_dict['in_domain'][key] = value
        curr_test_eval_dict = evaluation(args, datasets['test'], 'test', model, args.domain, epoch, 'current_results')
        for key, value in curr_test_eval_dict.items():
            test_eval_dict['in_domain'][key] = value
        best_model = deepcopy(model)
        best_optimizer = deepcopy(optimizer)
        patient = 0
    else:
        patient += 1
    if epoch == args.num_epochs:
        # save in-domain checkpoint
        for split in ['dev', 'test']:
            eval_dict = dev_eval_dict['in_domain'] if split == 'dev' else test_eval_dict['in_domain']
            write_results(args, datasets[split], args.domain, split, best_model, args.domain, eval_dict)
        save_checkpoint(best_model, best_optimizer, args.opt, dev_eval_dict, test_eval_dict, args.full_model_name)

    print('\n')
    return dev_eval_dict, test_eval_dict, best_model, best_optimizer, patient, curr_dev_eval_dict


def evaluation(args, data, split, model, domain, epoch, str_res='results'):
    # evaluate performance on data
    model.eval()
    auto_label_idx2inst = Index2Instance(args.alphabets['auto_label_alphabet'])
    eval_dict = initialize_eval_dict()
    eval_dict['epoch'] = epoch
    pred_labels = []
    gold_labels = []
    for batch in prepare_data.iterate_batch(data, args.batch_size, args.device):
        word, char, pos, ner, heads, arc_tags, auto_label, masks, lengths = batch
        output, masks, lengths = model.forward(word, char, pos, mask=masks, length=lengths)
        auto_label_preds = model.decode(output, mask=masks, length=lengths, leading_symbolic=prepare_data.NUM_SYMBOLIC_TAGS)
        lengths = lengths.cpu().numpy()
        word = word.data.cpu().numpy()
        pos = pos.data.cpu().numpy()
        ner = ner.data.cpu().numpy()
        heads = heads.data.cpu().numpy()
        arc_tags = arc_tags.data.cpu().numpy()
        auto_label = auto_label.data.cpu().numpy()
        auto_label_preds = auto_label_preds.data.cpu().numpy()
        gold_labels += auto_label_idx2inst.index2instance(auto_label, lengths, symbolic_root=True)
        pred_labels += auto_label_idx2inst.index2instance(auto_label_preds, lengths, symbolic_root=True)

    eval_dict['auto_label_accuracy'] = accuracy_score(gold_labels, pred_labels) * 100
    eval_dict['auto_label_precision'] = precision_score(gold_labels, pred_labels) * 100
    eval_dict['auto_label_recall'] = recall_score(gold_labels, pred_labels) * 100
    eval_dict['auto_label_f1'] = f1_score(gold_labels, pred_labels) * 100
    eval_dict['classification_report'] = classification_report(gold_labels, pred_labels)
    print_results(eval_dict, split, domain, str_res)
    return eval_dict


def print_results(eval_dict, split, domain, str_res='results'):
    print('----------------------------------------------------------------------------------------------------------------------------')
    print('Testing model on domain %s' % domain)
    print('--------------- sequence_tagger - %s ---------------' % split)
    print(
        str_res + ' on ' + split + ' accuracy: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)'
        % (eval_dict['auto_label_accuracy'], eval_dict['auto_label_precision'], eval_dict['auto_label_recall'], eval_dict['auto_label_f1'],
           eval_dict['epoch']))
    print(eval_dict['classification_report'])


def write_results(args, data, data_domain, split, model, model_domain, eval_dict):
    str_file = args.full_model_name + '_' + split + '_model_domain_' + model_domain + '_data_domain_' + data_domain
    res_filename = str_file + '_res.txt'
    pred_filename = str_file + '_pred.txt'
    gold_filename = str_file + '_gold.txt'

    # save results dictionary into a file
    with open(res_filename, 'w') as f:
        json.dump(eval_dict, f)

    # save predictions and gold labels into files
    pred_writer = Writer(args.alphabets)
    gold_writer = Writer(args.alphabets)
    pred_writer.start(pred_filename)
    gold_writer.start(gold_filename)
    for batch in prepare_data.iterate_batch(data, args.batch_size, args.device):
        word, char, pos, ner, heads, arc_tags, auto_label, masks, lengths = batch
        output, masks, lengths = model.forward(word, char, pos, mask=masks, length=lengths)
        auto_label_preds = model.decode(output, mask=masks, length=lengths,
                                        leading_symbolic=prepare_data.NUM_SYMBOLIC_TAGS)
        lengths = lengths.cpu().numpy()
        word = word.data.cpu().numpy()
        pos = pos.data.cpu().numpy()
        ner = ner.data.cpu().numpy()
        heads = heads.data.cpu().numpy()
        arc_tags = arc_tags.data.cpu().numpy()
        auto_label_preds = auto_label_preds.data.cpu().numpy()
        # writing predictions
        pred_writer.write(word, pos, ner, heads, arc_tags, lengths, auto_label=auto_label_preds, symbolic_root=True)
        # writing gold labels
        gold_writer.write(word, pos, ner, heads, arc_tags, lengths, auto_label=auto_label, symbolic_root=True)

    pred_writer.close()
    gold_writer.close()

def main():
    logger.info("Reading and creating arguments")
    args = read_arguments()
    logger.info("Reading Data")
    datasets = {}
    for split in args.splits:
        dataset = prepare_data.read_data_to_variable(args.data_paths[split], args.alphabets, args.device,
                                                     symbolic_root=True)
        datasets[split] = dataset

    logger.info("Creating Networks")
    num_data = sum(datasets['train'][1])
    model, optimizer, dev_eval_dict, test_eval_dict, start_epoch = build_model_and_optimizer(args)
    best_model = deepcopy(model)
    best_optimizer = deepcopy(optimizer)
    logger.info('Training INFO of in domain %s' % args.domain)
    logger.info('Training on Dependecy Parsing')
    print(model)
    logger.info("train: gamma: %f, batch: %d, clip: %.2f, unk replace: %.2f" % (args.gamma, args.batch_size, args.clip, args.unk_replace))
    logger.info('number of training samples for %s is: %d' % (args.domain, num_data))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (args.p_in, args.p_out, args.p_rnn))
    logger.info("num_epochs: %d" % (args.num_epochs))
    print('\n')

    if not args.eval_mode:
        logger.info("Training")
        num_batches = prepare_data.calc_num_batches(datasets['train'], args.batch_size)
        lr = args.learning_rate
        patient = 0
        terminal_patient = 0
        decay = 0
        for epoch in range(start_epoch + 1, args.num_epochs + 1):
            print('Epoch %d (Training: rnn mode: %s, optimizer: %s, learning rate=%.6f, eps=%.1e, decay rate=%.2f (schedule=%d, decay=%d)): ' % (
                epoch, args.rnn_mode, args.opt, lr, args.epsilon, args.decay_rate, args.schedule, decay))
            model.train()
            total_loss = 0.0
            total_train_inst = 0.0

            iter = prepare_data.iterate_batch_rand_bucket_choosing(
                    datasets['train'], args.batch_size, args.device, unk_replace=args.unk_replace)
            start_time = time.time()
            batch_num = 0
            for batch_num, batch in enumerate(iter):
                batch_num = batch_num + 1
                optimizer.zero_grad()
                # compute loss of main task
                word, char, pos, ner_tags, heads, arc_tags, auto_label, masks, lengths = batch
                output, masks, lengths = model.forward(word, char, pos, mask=masks, length=lengths)
                loss = model.loss(output, auto_label, mask=masks, length=lengths)

                # update losses
                num_insts = masks.data.sum() - word.size(0)
                total_loss += loss.item() * num_insts
                total_train_inst += num_insts
                # optimize parameters
                loss.backward()
                clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()

                time_ave = (time.time() - start_time) / batch_num
                time_left = (num_batches - batch_num) * time_ave

                # update log
                if batch_num % 50 == 0:
                    log_info = 'train: %d/%d, domain: %s, total loss: %.2f, time left: %.2fs' % \
                               (batch_num, num_batches, args.domain, total_loss / total_train_inst, time_left)
                    sys.stdout.write(log_info)
                    sys.stdout.write('\n')
                    sys.stdout.flush()
            print('\n')
            print('train: %d/%d, domain: %s, total_loss: %.2f, time: %.2fs' %
                  (batch_num, num_batches, args.domain, total_loss / total_train_inst, time.time() - start_time))

            dev_eval_dict, test_eval_dict, best_model, best_optimizer, patient,curr_dev_eval_dict = in_domain_evaluation(args, datasets, model, optimizer, dev_eval_dict, test_eval_dict, epoch, best_model, best_optimizer, patient)
            store ={'dev_eval_dict':curr_dev_eval_dict }
            ############################################# 
            str_file = args.full_model_name + '_' +'all_epochs'
            with open(str_file,'a') as f:
                f.write(str(store)+'\n')
            if patient == 0:
                terminal_patient = 0
            else:
                terminal_patient += 1
            if terminal_patient >= 4 * args.schedule:
                # Save best model and terminate learning
                cur_epoch = epoch
                epoch = args.num_epochs
                in_domain_evaluation(args, datasets, model, optimizer, dev_eval_dict, test_eval_dict, epoch, best_model,
                                     best_optimizer, patient)
                log_info = 'Terminating training in epoch %d' % (cur_epoch)
                sys.stdout.write(log_info)
                sys.stdout.write('\n')
                sys.stdout.flush()
                return
            if patient >= args.schedule:
                lr = args.learning_rate / (1.0 + epoch * args.decay_rate)
                optimizer = generate_optimizer(args, lr, model.parameters())
                print('updated learning rate to %.6f' % lr)
                patient = 0
            print_results(test_eval_dict['in_domain'], 'test', args.domain, 'best_results')
            print('\n')

    else:
        logger.info("Evaluating")
        epoch = start_epoch
        for split in ['train', 'dev', 'test']:
            evaluation(args, datasets[split], split, model, args.domain, epoch, 'best_results')


if __name__ == '__main__':
    main()
