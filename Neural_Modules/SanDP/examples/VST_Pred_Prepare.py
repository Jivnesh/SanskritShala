import sys

def write_combined(dirs):
    path = "./saved_models/"+dirs+'/final_ensembled_TranSeq/'
    f = open(path+'domain_VST_test_model_domain_VST_data_domain_VST_gold.txt','r')
    gold = f.readlines()
    f.close()
    f = open(path+'domain_VST_test_model_domain_VST_data_domain_VST_pred.txt','r')
    pred = f.readlines()
    f.close()

    for i in range(len(gold)):
        if gold[i] == '\n':
            continue
        if gold[i].split('\t')[0] == pred[i].split('\t')[0]:
            gold[i] = gold[i].replace('\n','\t')
            gold[i] = gold[i]+'\t'.join(pred[i].split('\t')[-2:])


    gold.insert(0,'word_id\tword\tpostag\tlemma\tgold_head\tgold_label\tpred_head\tpred_label\n\n')


    f = open(path+'VST_test.txt','w')
    for line in gold:
        f.write(line)
    f.close()


if __name__=="__main__":

    dir_path = sys.argv[1]

    write_combined(dir_path)

