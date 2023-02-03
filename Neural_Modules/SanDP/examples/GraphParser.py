from __future__ import print_function
import sys
from os import path, makedirs

sys.path.append(".")
sys.path.append("..")

import argparse
from copy import deepcopy
import json
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from utils.io_ import seeds, Writer, get_logger, prepare_data, rearrange_splits
from utils.models.parsing_gating import BiAffine_Parser_Gated
from utils import load_word_embeddings
from utils.tasks import parse
import time
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, SGD
import uuid

uid = uuid.uuid4().hex[:6]

logger = get_logger('GraphParser')

def read_arguments():
    args_ = argparse.ArgumentParser(description='Sovling GraphParser')
    args_.add_argument('--dataset', choices=['ontonotes', 'ud'], help='Dataset', required=True)
    args_.add_argument('--domain', help='domain/language', required=True)
    args_.add_argument('--rnn_mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn',
                       required=True)
    args_.add_argument('--gating',action='store_true', help='use gated mechanism')
    args_.add_argument('--num_gates', type=int, default=0, help='number of gates for gating mechanism')
    args_.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_.add_argument('--batch_size', type=int, default=64, help='Number of sentences in each batch')
    args_.add_argument('--hidden_size', type=int, default=256, help='Number of hidden units in RNN')
    args_.add_argument('--arc_space', type=int, default=128, help='Dimension of tag space')
    args_.add_argument('--arc_tag_space', type=int, default=128, help='Dimension of tag space')
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
    args_.add_argument('--arc_decode', choices=['mst', 'greedy'], help='arc decoding algorithm', required=True)
    args_.add_argument('--unk_replace', type=float, default=0.,
                       help='The rate to replace a singleton word with UNK')
    args_.add_argument('--punct_set', nargs='+', type=str, help='List of punctuations')
    args_.add_argument('--word_embedding', choices=['random', 'glove', 'fasttext', 'word2vec'],
                       help='Embedding for words')
    args_.add_argument('--word_path', help='path for word embedding dict - in case word_embedding is not random')
    args_.add_argument('--freeze_word_embeddings', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_.add_argument('--freeze_sequence_taggers', action='store_true', help='frozen the BiLSTMs of the pre-trained taggers.')
    args_.add_argument('--char_embedding', choices=['random','hellwig'], help='Embedding for characters',
                       required=True)
    args_.add_argument('--pos_embedding', choices=['random','one_hot'], help='Embedding for pos',
                       required=True)
    args_.add_argument('--char_path', help='path for character embedding dict')
    args_.add_argument('--pos_path', help='path for pos embedding dict')
    args_.add_argument('--set_num_training_samples', type=int, help='downsampling training set to a fixed number of samples')
    args_.add_argument('--model_path', help='path for saving model file.', required=True)
    args_.add_argument('--load_path', help='path for loading saved source model file.', default=None)
    args_.add_argument('--load_sequence_taggers_paths', nargs='+', help='path for loading saved sequence_tagger saved_models files.', default=None)
    args_.add_argument('--strict',action='store_true', help='if True loaded model state should contin '
                                                            'exactly the same keys as current model')
    args_.add_argument('--eval_mode', action='store_true', help='evaluating model without training it')
    args = args_.parse_args()
    args_dict = {}
    args_dict['dataset'] = args.dataset
    args_dict['domain'] = args.domain
    args_dict['rnn_mode'] = args.rnn_mode
    args_dict['gating'] = args.gating
    args_dict['num_gates'] = args.num_gates
    args_dict['arc_decode'] = args.arc_decode
    # args_dict['splits'] = ['train', 'dev', 'test']
    args_dict['splits'] = ['train', 'dev', 'test','poetry','prose']
    args_dict['model_path'] = args.model_path
    if not path.exists(args_dict['model_path']):
        makedirs(args_dict['model_path'])
    args_dict['data_paths'] = {}
    if args_dict['dataset'] == 'ontonotes':
        data_path = 'data/onto_pos_ner_dp'
    else:
        data_path = 'data/ud_pos_ner_dp'
    for split in args_dict['splits']:
        args_dict['data_paths'][split] = data_path + '_' + split + '_' + args_dict['domain']
    ###################################    
    args_dict['data_paths']['poetry'] = 'data/ud_pos_ner_dp' + '_' + 'poetry' + '_' + args_dict['domain']
    args_dict['data_paths']['prose'] = 'data/ud_pos_ner_dp' + '_' + 'prose' + '_' + args_dict['domain']
    ###################################
    args_dict['alphabet_data_paths'] = {}
    for split in args_dict['splits']:
        if args_dict['dataset'] == 'ontonotes':
            args_dict['alphabet_data_paths'][split] = data_path + '_' + split + '_' + 'all'
        else:
            if '_' in args_dict['domain']:
                args_dict['alphabet_data_paths'][split] = data_path + '_' + split + '_' + args_dict['domain'].split('_')[0]
            else:
                args_dict['alphabet_data_paths'][split] = args_dict['data_paths'][split]
    args_dict['model_name'] = 'domain_' + args_dict['domain']
    args_dict['full_model_name'] = path.join(args_dict['model_path'],args_dict['model_name'])
    args_dict['load_path'] = args.load_path
    args_dict['load_sequence_taggers_paths'] = args.load_sequence_taggers_paths
    if args_dict['load_sequence_taggers_paths'] is not None:
        args_dict['gating'] = True
        args_dict['num_gates'] = len(args_dict['load_sequence_taggers_paths']) + 1
    else:
        if not args_dict['gating']:
            args_dict['num_gates'] = 0
    args_dict['strict'] = args.strict
    args_dict['num_epochs'] = args.num_epochs
    args_dict['batch_size'] = args.batch_size
    args_dict['hidden_size'] = args.hidden_size
    args_dict['arc_space'] = args.arc_space
    args_dict['arc_tag_space'] = args.arc_tag_space
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
    args_dict['set_num_training_samples'] = args.set_num_training_samples
    args_dict['punct_set'] = None
    if args.punct_set is not None:
        args_dict['punct_set'] = set(args.punct_set)
        logger.info("punctuations(%d): %s" % (len(args_dict['punct_set']), ' '.join(args_dict['punct_set'])))
    args_dict['freeze_word_embeddings'] = args.freeze_word_embeddings
    args_dict['freeze_sequence_taggers'] = args.freeze_sequence_taggers
    args_dict['word_embedding'] = args.word_embedding
    args_dict['word_path'] = args.word_path
    args_dict['use_char'] = args.use_char
    args_dict['char_embedding'] = args.char_embedding
    args_dict['char_path'] = args.char_path
    args_dict['pos_embedding'] = args.pos_embedding
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
    args_dict['model_name'] = path.join(args_dict['model_path'], args_dict['model_name'])
    args_dict['eval_mode'] = args.eval_mode
    args_dict['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args_dict['word_status'] = 'frozen' if args.freeze_word_embeddings else 'fine tune'
    args_dict['char_status'] = 'enabled' if args.use_char else 'disabled'
    args_dict['pos_status'] = 'enabled' if args.use_pos else 'disabled'
    logger.info("Saving arguments to file")
    save_args(args, args_dict['full_model_name'])
    logger.info("Creating Alphabets")
    alphabet_dict = creating_alphabets(args_dict['alphabet_path'], args_dict['alphabet_data_paths'], args_dict['word_dict'])
    args_dict = {**args_dict, **alphabet_dict}
    ARGS = namedtuple('ARGS', args_dict.keys())
    my_args = ARGS(**args_dict)
    return my_args


def creating_alphabets(alphabet_path, alphabet_data_paths, word_dict):
    train_paths = alphabet_data_paths['train']
    extra_paths = [v for k,v in alphabet_data_paths.items() if k != 'train']
    alphabet_dict = {}
    alphabet_dict['alphabets'] = prepare_data.create_alphabets(alphabet_path,
                                                               train_paths,
                                                               extra_paths=extra_paths,
                                                               max_vocabulary_size=100000,
                                                               embedd_dict=word_dict)
    for k, v in alphabet_dict['alphabets'].items():
        num_key = 'num_' + k.split('_')[0]
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
             'opt': opt,
             'dev_eval_dict': dev_eval_dict,
             'test_eval_dict': test_eval_dict}
    torch.save(state, path_name)


def load_checkpoint(args, model, optimizer, dev_eval_dict, test_eval_dict, start_epoch, load_path, strict=True):
    print('Loading saved model from: %s' % load_path)
    checkpoint = torch.load(load_path, map_location=args.device)
    if checkpoint['opt'] != args.opt:
        raise ValueError('loaded optimizer type is: %s instead of: %s' % (checkpoint['opt'], args.opt))
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)

    if strict:
        generate_optimizer(args, args.learning_rate, model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
        dev_eval_dict = checkpoint['dev_eval_dict']
        test_eval_dict = checkpoint['test_eval_dict']
        start_epoch = dev_eval_dict['in_domain']['epoch']
    return model, optimizer, dev_eval_dict, test_eval_dict, start_epoch


def build_model_and_optimizer(args):
    word_table = construct_embedding_table(args.alphabets['word_alphabet'], args.word_dict, args.word_dim, token_type='word')
    char_table = construct_embedding_table(args.alphabets['char_alphabet'], args.char_dict, args.char_dim, token_type='char')
    pos_table = construct_embedding_table(args.alphabets['pos_alphabet'], args.pos_dict, args.pos_dim, token_type='pos')
    model = BiAffine_Parser_Gated(args.word_dim, args.num_word, args.char_dim, args.num_char,
                            args.use_pos, args.use_char, args.pos_dim, args.num_pos,
                            args.num_filters, args.kernel_size, args.rnn_mode,
                            args.hidden_size, args.num_layers, args.num_arc,
                            args.arc_space, args.arc_tag_space, args.num_gates,
                            embedd_word=word_table, embedd_char=char_table, embedd_pos=pos_table,
                            p_in=args.p_in, p_out=args.p_out,  p_rnn=args.p_rnn,
                            biaffine=True, arc_decode=args.arc_decode, initializer=args.initializer)
    print(model)
    optimizer = generate_optimizer(args, args.learning_rate, model.parameters())
    start_epoch = 0
    dev_eval_dict = {'in_domain': initialize_eval_dict()}
    test_eval_dict = {'in_domain': initialize_eval_dict()}
    if args.load_path:
        model, optimizer, dev_eval_dict, test_eval_dict, start_epoch = \
            load_checkpoint(args, model, optimizer,
                            dev_eval_dict, test_eval_dict,
                            start_epoch, args.load_path, strict=args.strict)
    if args.load_sequence_taggers_paths:
        pretrained_dict = {}
        model_dict = model.state_dict()
        for idx, path in enumerate(args.load_sequence_taggers_paths):
            print('Loading saved sequence_tagger from: %s' % path)
            checkpoint = torch.load(path, map_location=args.device)
            for k, v in checkpoint['model_state_dict'].items():
                if 'rnn_encoder.' in k:
                    pretrained_dict['extra_rnn_encoders.' + str(idx) + '.' + k.replace('rnn_encoder.', '')] = v
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    if args.freeze_sequence_taggers:
        print('Freezing Classifiers')
        for name, parameter in model.named_parameters():
            if 'extra_rnn_encoders' in name:
                parameter.requires_grad = False
    if args.freeze_word_embeddings:
        model.rnn_encoder.word_embedd.weight.requires_grad = False
        # model.rnn_encoder.char_embedd.weight.requires_grad = False
        # model.rnn_encoder.pos_embedd.weight.requires_grad = False
    device = args.device
    model.to(device)
    return model, optimizer, dev_eval_dict, test_eval_dict, start_epoch


def initialize_eval_dict():
    eval_dict = {}
    eval_dict['dp_uas'] = 0.0
    eval_dict['dp_las'] = 0.0
    eval_dict['epoch'] = 0
    eval_dict['dp_ucorrect'] = 0.0
    eval_dict['dp_lcorrect'] = 0.0
    eval_dict['dp_total'] = 0.0
    eval_dict['dp_ucomplete_match'] = 0.0
    eval_dict['dp_lcomplete_match'] = 0.0
    eval_dict['dp_ucorrect_nopunc'] = 0.0
    eval_dict['dp_lcorrect_nopunc'] = 0.0
    eval_dict['dp_total_nopunc'] = 0.0
    eval_dict['dp_ucomplete_match_nopunc'] = 0.0
    eval_dict['dp_lcomplete_match_nopunc'] = 0.0
    eval_dict['dp_root_correct'] = 0.0
    eval_dict['dp_total_root'] = 0.0
    eval_dict['dp_total_inst'] = 0.0
    eval_dict['dp_total'] = 0.0
    eval_dict['dp_total_inst'] = 0.0
    eval_dict['dp_total_nopunc'] = 0.0
    eval_dict['dp_total_root'] = 0.0
    return eval_dict

def in_domain_evaluation(args, datasets, model, optimizer, dev_eval_dict, test_eval_dict, epoch,
                         best_model, best_optimizer, patient):
    # In-domain evaluation
    curr_dev_eval_dict = evaluation(args, datasets['dev'], 'dev', model, args.domain, epoch, 'current_results')
    is_best_in_domain = dev_eval_dict['in_domain']['dp_lcorrect_nopunc'] <= curr_dev_eval_dict['dp_lcorrect_nopunc'] or \
              (dev_eval_dict['in_domain']['dp_lcorrect_nopunc'] == curr_dev_eval_dict['dp_lcorrect_nopunc'] and
               dev_eval_dict['in_domain']['dp_ucorrect_nopunc'] <= curr_dev_eval_dict['dp_ucorrect_nopunc'])

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
        if args.set_num_training_samples is not None:
            splits_to_write = datasets.keys()
        else:
            splits_to_write = ['dev', 'test']
        for split in splits_to_write:
            if split == 'dev':
                eval_dict = dev_eval_dict['in_domain']
            elif split == 'test':
                eval_dict = test_eval_dict['in_domain']
            else:
                eval_dict = None
            write_results(args, datasets[split], args.domain, split, best_model, args.domain, eval_dict)
        print("Saving best model")
        save_checkpoint(best_model, best_optimizer, args.opt, dev_eval_dict, test_eval_dict, args.full_model_name)

    print('\n')
    return dev_eval_dict, test_eval_dict, best_model, best_optimizer, patient


def evaluation(args, data, split, model, domain, epoch, str_res='results'):
    # evaluate performance on data
    model.eval()

    eval_dict = initialize_eval_dict()
    eval_dict['epoch'] = epoch
    for batch in prepare_data.iterate_batch(data, args.batch_size, args.device):
        word, char, pos, ner, heads, arc_tags, auto_label, masks, lengths = batch
        out_arc, out_arc_tag, masks, lengths = model.forward(word, char, pos, mask=masks, length=lengths)
        heads_pred, arc_tags_pred, _ = model.decode(out_arc, out_arc_tag, mask=masks, length=lengths,
                                                    leading_symbolic=prepare_data.NUM_SYMBOLIC_TAGS)
        lengths = lengths.cpu().numpy()
        word = word.data.cpu().numpy()
        pos = pos.data.cpu().numpy()
        ner = ner.data.cpu().numpy()
        heads = heads.data.cpu().numpy()
        arc_tags = arc_tags.data.cpu().numpy()
        heads_pred = heads_pred.data.cpu().numpy()
        arc_tags_pred = arc_tags_pred.data.cpu().numpy()
        stats, stats_nopunc, stats_root, num_inst = parse.eval_(word, pos, heads_pred, arc_tags_pred, heads,
                                                                arc_tags, args.alphabets['word_alphabet'], args.alphabets['pos_alphabet'],
                                                                lengths, punct_set=args.punct_set, symbolic_root=True)
        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        corr_root, total_root = stats_root
        eval_dict['dp_ucorrect'] += ucorr
        eval_dict['dp_lcorrect'] += lcorr
        eval_dict['dp_total'] += total
        eval_dict['dp_ucomplete_match'] += ucm
        eval_dict['dp_lcomplete_match'] += lcm
        eval_dict['dp_ucorrect_nopunc'] += ucorr_nopunc
        eval_dict['dp_lcorrect_nopunc'] += lcorr_nopunc
        eval_dict['dp_total_nopunc'] += total_nopunc
        eval_dict['dp_ucomplete_match_nopunc'] += ucm_nopunc
        eval_dict['dp_lcomplete_match_nopunc'] += lcm_nopunc
        eval_dict['dp_root_correct'] += corr_root
        eval_dict['dp_total_root'] += total_root
        eval_dict['dp_total_inst'] += num_inst

    eval_dict['dp_uas'] = eval_dict['dp_ucorrect'] * 100 / eval_dict['dp_total']  # considering w. punctuation
    eval_dict['dp_las'] = eval_dict['dp_lcorrect'] * 100 / eval_dict['dp_total']  # considering w. punctuation
    print_results(eval_dict, split, domain, str_res)
    return eval_dict


def print_results(eval_dict, split, domain, str_res='results'):
    print('----------------------------------------------------------------------------------------------------------------------------')
    print('Testing model on domain %s' % domain)
    print('--------------- Dependency Parsing - %s ---------------' % split)
    print(
        str_res + ' on ' + split + '  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            eval_dict['dp_ucorrect'], eval_dict['dp_lcorrect'], eval_dict['dp_total'],
            eval_dict['dp_ucorrect'] * 100 / eval_dict['dp_total'],
            eval_dict['dp_lcorrect'] * 100 / eval_dict['dp_total'],
            eval_dict['dp_ucomplete_match'] * 100 / eval_dict['dp_total_inst'],
            eval_dict['dp_lcomplete_match'] * 100 / eval_dict['dp_total_inst'],
            eval_dict['epoch']))
    print(
        str_res + ' on ' + split + '  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
            eval_dict['dp_ucorrect_nopunc'], eval_dict['dp_lcorrect_nopunc'], eval_dict['dp_total_nopunc'],
            eval_dict['dp_ucorrect_nopunc'] * 100 / eval_dict['dp_total_nopunc'],
            eval_dict['dp_lcorrect_nopunc'] * 100 / eval_dict['dp_total_nopunc'],
            eval_dict['dp_ucomplete_match_nopunc'] * 100 / eval_dict['dp_total_inst'],
            eval_dict['dp_lcomplete_match_nopunc'] * 100 / eval_dict['dp_total_inst'],
            eval_dict['epoch']))
    print(str_res + ' on ' + split + '  Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
        eval_dict['dp_root_correct'], eval_dict['dp_total_root'],
        eval_dict['dp_root_correct'] * 100 / eval_dict['dp_total_root'], eval_dict['epoch']))
    print('\n')

def write_results(args, data, data_domain, split, model, model_domain, eval_dict):
    str_file = args.full_model_name + '_' + split + '_model_domain_' + model_domain + '_data_domain_' + data_domain
    res_filename = str_file + '_res.txt'
    pred_filename = str_file + '_pred.txt'
    gold_filename = str_file + '_gold.txt'
    if eval_dict is not None:
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
        out_arc, out_arc_tag, masks, lengths = model.forward(word, char, pos, mask=masks, length=lengths)
        heads_pred, arc_tags_pred, _ = model.decode(out_arc, out_arc_tag, mask=masks, length=lengths,
                                                    leading_symbolic=prepare_data.NUM_SYMBOLIC_TAGS)
        lengths = lengths.cpu().numpy()
        word = word.data.cpu().numpy()
        pos = pos.data.cpu().numpy()
        ner = ner.data.cpu().numpy()
        heads = heads.data.cpu().numpy()
        arc_tags = arc_tags.data.cpu().numpy()
        heads_pred = heads_pred.data.cpu().numpy()
        arc_tags_pred = arc_tags_pred.data.cpu().numpy()
        # writing predictions
        pred_writer.write(word, pos, ner, heads_pred, arc_tags_pred, lengths, symbolic_root=True)
        # writing gold labels
        gold_writer.write(word, pos, ner, heads, arc_tags, lengths, symbolic_root=True)

    pred_writer.close()
    gold_writer.close()

def main():
    logger.info("Reading and creating arguments")
    args = read_arguments()
    logger.info("Reading Data")
    datasets = {}
    for split in args.splits:
        print("Splits are:",split)
        dataset = prepare_data.read_data_to_variable(args.data_paths[split], args.alphabets, args.device,
                                                     symbolic_root=True)
        datasets[split] = dataset
    if args.set_num_training_samples is not None:
        print('Setting train and dev to %d samples' % args.set_num_training_samples)
        datasets = rearrange_splits.rearranging_splits(datasets, args.set_num_training_samples)
    logger.info("Creating Networks")
    num_data = sum(datasets['train'][1])
    model, optimizer, dev_eval_dict, test_eval_dict, start_epoch = build_model_and_optimizer(args)
    best_model = deepcopy(model)
    best_optimizer = deepcopy(optimizer)

    logger.info('Training INFO of in domain %s' % args.domain)
    logger.info('Training on Dependecy Parsing')
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
        decay = 0
        for epoch in range(start_epoch + 1, args.num_epochs + 1):
            print('Epoch %d (Training: rnn mode: %s, optimizer: %s, learning rate=%.6f, eps=%.1e, decay rate=%.2f (schedule=%d, decay=%d)): ' % (
                epoch, args.rnn_mode, args.opt, lr, args.epsilon, args.decay_rate, args.schedule, decay))
            model.train()
            total_loss = 0.0
            total_arc_loss = 0.0
            total_arc_tag_loss = 0.0
            total_train_inst = 0.0

            train_iter = prepare_data.iterate_batch_rand_bucket_choosing(
                    datasets['train'], args.batch_size, args.device, unk_replace=args.unk_replace)
            start_time = time.time()
            batch_num = 0
            for batch_num, batch in enumerate(train_iter):
                batch_num = batch_num + 1
                optimizer.zero_grad()
                # compute loss of main task
                word, char, pos, ner_tags, heads, arc_tags, auto_label, masks, lengths = batch
                out_arc, out_arc_tag, masks, lengths = model.forward(word, char, pos, mask=masks, length=lengths)
                loss_arc, loss_arc_tag = model.loss(out_arc, out_arc_tag, heads, arc_tags, mask=masks, length=lengths)
                loss = loss_arc + loss_arc_tag

                # update losses
                num_insts = masks.data.sum() - word.size(0)
                total_arc_loss += loss_arc.item() * num_insts
                total_arc_tag_loss += loss_arc_tag.item() * num_insts
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
                    log_info = 'train: %d/%d, domain: %s, total loss: %.2f, arc_loss: %.2f, arc_tag_loss: %.2f, time left: %.2fs' % \
                               (batch_num, num_batches, args.domain, total_loss / total_train_inst, total_arc_loss / total_train_inst,
                                total_arc_tag_loss / total_train_inst, time_left)
                    sys.stdout.write(log_info)
                    sys.stdout.write('\n')
                    sys.stdout.flush()
            print('\n')
            print('train: %d/%d, domain: %s, total_loss: %.2f, arc_loss: %.2f, arc_tag_loss: %.2f, time: %.2fs' %
                  (batch_num, num_batches, args.domain, total_loss / total_train_inst, total_arc_loss / total_train_inst,
                   total_arc_tag_loss / total_train_inst, time.time() - start_time))

            dev_eval_dict, test_eval_dict, best_model, best_optimizer, patient = in_domain_evaluation(args, datasets, model, optimizer, dev_eval_dict, test_eval_dict, epoch, best_model, best_optimizer, patient)
            if patient >= args.schedule:
                lr = args.learning_rate / (1.0 + epoch * args.decay_rate)
                optimizer = generate_optimizer(args, lr, model.parameters())
                print('updated learning rate to %.6f' % lr)
                patient = 0
            print_results(test_eval_dict['in_domain'], 'test', args.domain, 'best_results')
            print('\n')
        for split in datasets.keys():
            eval_dict = evaluation(args, datasets[split], split, best_model, args.domain, epoch, 'best_results')
            write_results(args, datasets[split], args.domain, split, model, args.domain, eval_dict)

    else:
        logger.info("Evaluating")
        epoch = start_epoch
        for split in ['train', 'dev', 'test','poetry','prose']:
            eval_dict = evaluation(args, datasets[split], split, model, args.domain, epoch, 'best_results')
            write_results(args, datasets[split], args.domain, split, model, args.domain, eval_dict)


if __name__ == '__main__':
    main()
