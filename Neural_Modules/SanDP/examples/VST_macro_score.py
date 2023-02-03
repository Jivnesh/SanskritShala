import sys


def load_results(filename):

    results = []
    sent = []
    with open(filename, 'r') as fp:
        for i, line in enumerate(fp):
            if i == 0:
                continue
            splits = line.strip().split('\t')
            if len(line.strip()) == 0:
                if len(sent) != 0:
                    results.append(sent)
                    sent = []
                continue
            gold_head = splits[-4]
            gold_label = splits[-3]
            pred_head = splits[-2]
            pred_label = splits[-1]
            sent.append((gold_head, gold_label, pred_head, pred_label))
    print('Total Number of sentences ' + str(len(results)))
    return results

def calculate_las_uas(gold_heads, gold_labels, pred_heads, pred_labels):

    u_correct = 0
    l_correct = 0
    u_total = 0
    l_total = 0

    for i in range(len(gold_heads)):
        if gold_heads[i] == pred_heads[i]:
            u_correct +=1
        u_total +=1
        l_total +=1
        if gold_heads[i] == pred_heads[i] and gold_labels[i] == pred_labels[i]:
            l_correct +=1
    return u_correct, u_total, l_correct, l_total


def calculate_stats(results,path):
    u_correct = 0
    l_correct = 0
    u_total = 0
    l_total = 0

    sent_uas = []
    sent_las = []

    for i in range(len(results)):
        gold_heads, gold_labels, pred_heads, pred_labels = zip(*results[i])
        u_c, u_t, l_c, l_t = calculate_las_uas(gold_heads, gold_labels, pred_heads, pred_labels)
        if u_t >0:
            uas = float(u_c)/u_t
            las = float(l_c)/l_t
            sent_uas.append(uas)
            sent_las.append(las)
        u_correct += u_c
        l_correct += l_c
        u_total += u_t
        l_total += l_t

    UAS = float(u_correct)/u_total
    LAS = float(l_correct)/l_total
    path = path.replace('VST_test.txt','Macro-UAS-LAS-score.txt')
    f = open(path,'w')
    f.write('Word level UAS : ' + str(UAS) +'\n')
    f.write('Word level LAS : ' + str(LAS)+'\n')
    f.write('Sentence level UAS : ' + str(float(sum(sent_uas))/len(sent_uas))+'\n')
    f.write('Sentence level LAS : ' + str(float(sum(sent_las))/len(sent_las))+'\n')
    f.close()
    print('Word level UAS : ' + str(UAS))
    print('Word level LAS : ' + str(LAS))
    print('Sentence level UAS : ' + str(float(sum(sent_uas))/len(sent_uas)))
    print('Sentence level LAS : ' + str(float(sum(sent_las))/len(sent_las)))

    return sent_uas, sent_las, UAS, LAS

def write_results(sent_uas, sent_las, filename_uas, filename_las):

    fp_uas = open(filename_uas, 'w')
    fp_las = open(filename_las, 'w')

    for i in range(len(sent_uas)):
        fp_uas.write(str(sent_uas[i]) + '\n')
        fp_las.write(str(sent_las[i]) + '\n')

    fp_uas.close()
    fp_las.close()


if __name__=="__main__":
    dirs = sys.argv[1]
    # results_2 = load_results(sys.argv[2])
    ##path = "Predictions/Yap/"+dirs
    path = "./saved_models/"+dirs+"/final_ensembled_TranSeq/VST_test.txt"
    result = load_results(path)


    sent_uas1, sent_las1, UAS1, LAS1 = calculate_stats(result,path)
    # sent_uas2, sent_las2, UAS2, LAS2 = calculate_stats(results_2)


    write_results(sent_uas1, sent_las1, 'results1_uas.txt', 'results1_las.txt')
    # write_results(sent_uas2, sent_las2, 'results2_uas.txt', 'results2_las.txt')
