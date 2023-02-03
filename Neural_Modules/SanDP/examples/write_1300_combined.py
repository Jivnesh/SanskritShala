import sys

def write_combined(dirs):
    path = "./saved_models/"+dirs+"/final_ensembled/"
    f = open(path+'domain_san_test_model_domain_san_data_domain_san_gold.txt','r')
    gold = f.readlines()
    f.close()
    f = open(path+'domain_san_test_model_domain_san_data_domain_san_pred.txt','r')
    pred = f.readlines()
    f.close()

    for i in range(len(gold)):
        if gold[i] == '\n':
            continue
        if gold[i].split('\t')[0] == pred[i].split('\t')[0]:
            gold[i] = gold[i].replace('\n','\t')
            gold[i] = gold[i]+'\t'.join(pred[i].split('\t')[-2:])

    f = open(path+'domain_san_prose_model_domain_san_data_domain_san_gold.txt','r')
    prose_gold = f.readlines()
    f.close()
    f = open(path+'domain_san_prose_model_domain_san_data_domain_san_pred.txt','r')
    prose_pred = f.readlines()
    f.close()

    for i in range(len(prose_gold)):
        if prose_gold[i] == '\n':
            gold.append('\n')
            continue
        if prose_gold[i].split('\t')[0] == prose_pred[i].split('\t')[0]:
            line = prose_gold[i].replace('\n','\t')
            line =line+'\t'.join(prose_pred[i].split('\t')[-2:])
            gold.append(line)
    gold.insert(0,'word_id\tword\tpostag\tlemma\tgold_head\tgold_label\tpred_head\tpred_label\n\n')


    f = open(path+'combined_1300_test.txt','w')
    for line in gold:
        f.write(line)
    f.close()


if __name__=="__main__":

    dir_path = sys.argv[1]

    write_combined(dir_path)

