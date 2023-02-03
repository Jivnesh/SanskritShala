from unicodedata import decimal
import torch
from sklearn.metrics import classification_report
from data_config import get_path,choices
from tpipeline import TPipeline,TaggerDataset
import time
import argparse

def acc(path,test_d_path,exp_type):
    f = open(test_d_path,'r')
    gold =  f.readlines()
    f.close()
    f = open(path)
    pred =  f.readlines()
    f.close()
    w = open('combine.pks.conll','w')
    w.write('word_id	word	postag	lemma	gold_head	gold_label	pred_head	pred_label\n')
    for i in range(len(gold)):
        if gold[i] == '\n':
            w.write('\n')
            continue
        gold[i] = gold[i].split('\t')
        gold[i][-1] = gold[i][-1].replace('\n','')
        pred[i] = pred[i].split('\t')
        pred[i][-1] = pred[i][-1].replace('\n','')
        temp = [gold[i][0],gold[i][1],gold[i][3],gold[i][3],gold[i][6],gold[i][7],pred[i][6],pred[i][7]]
        w.write('\t'.join(temp)+'\n')
    w.close()
    targs = []
    preds= []
    pr,tg=[],[]
    # print(pred)
    for i in range(len(pred)):
        if gold[i] == '\n':
            continue
        preds.append(pred[i][7])
        targs.append(gold[i][7])
    target_names = list(set(targs))
    print(classification_report(preds, targs, target_names=target_names,digits=4))
    f = open(exp_type+'eval_matrix.txt','w')
    f.write(str(classification_report(preds, targs, target_names=target_names,digits=4)))
    f.close()

def run(panelty,model_path,train_path,dev_path,test_d_path,epochs,btch_size,exp_type,training):
    torch.cuda.empty_cache()
    print(train_boolean)
    trainer = TPipeline(
            training_config={
            'category': 'customized-mwt-ner', # pipeline category
            'task': 'posdep', # task name
            'save_dir': model_path, # directory for saving trained model
            'train_conllu_fpath': train_path, # annotations file in CONLLU format  for training
            'dev_conllu_fpath': dev_path, # annotations file in CONLLU format for development
            'max_epoch': epochs,
            "batch_size":btch_size,
            'panelty':panelty,
            "training":training
        })

    if training:
        trainer.train()
    # test_d_path = dev_path
    test_set = TaggerDataset(
        config=trainer._config,
        input_conllu=test_d_path,
        gold_conllu=test_d_path,
        evaluate=True
    )
    test_set.numberize()
    test_batch_num = len(test_set) // trainer._config.batch_size + (len(test_set) % trainer._config.batch_size != 0)
    result,path = trainer._eval_posdep(data_set=test_set, batch_num=test_batch_num,
                            name='testfaL', epoch=-1,task='test')
    print("Path of test preds ",path)
    del trainer
    torch.cuda.empty_cache()
    acc(path,test_d_path,exp_type)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/models1', help='Model path')
    parser.add_argument('--experiment', type=str, default='saCTI-base coarse', help='Experiment type',choices=choices)
    parser.add_argument('--epochs', type=int, default=80, help='epochs')
    parser.add_argument('--batch_size', type=int, default=55, help='batch size')
    parser.add_argument('--training', type=str, default='False', help='True if traning and False if Test',choices=['False','True'])


    args = parser.parse_args()
    exp_type = args.experiment

    train_path,dev_path,test_path = get_path(exp_type)
    model_path = args.model_path #'./models/'
    train_boolean = True if args.training=='True' else False 
    print(train_boolean)
    
    
    panelty = 0.01
    run(panelty,model_path,train_path,dev_path,test_path,args.epochs,args.batch_size,exp_type,train_boolean)



