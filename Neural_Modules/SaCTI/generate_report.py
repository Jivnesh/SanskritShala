from cgi import test
import os
def make_report(path_origdata,csv,task):
    #test_grammer_swap0.1_eval_matrix
    # report_which = path_origdata.replace('15t','t')
    # report_which = path_origdata.replace('15d','d')
    if task=='test':
        report_which = path_origdata.split('/')[-1].split('test')[0]
    else:
        report_which = path_origdata.split('/')[-1].split('dev')[0]
    

    # print(report_which)
    # print(report_which.replace('compounds','').replace('context',''))
    model= path_origdata.split('/')[-1].replace('swap','').replace('_eval_matrix.txt','')
    model = model.split('_')[-1]
    print("t",model)

    #report_which.replace('compounds','').replace('context','')

    dev1 = open(path_origdata,'r')
    dev = dev1.readlines()
    for i in range(len(dev)):
        dev[i] = dev[i].replace('\n','').split('  ')
    print()
    print()
    print()
    print("yesh")
    print(dev7])

    print(dev[10][2])
    # exit()
    accuracy = dev[7][13]
    f1 = dev[8][4]
    recall = dev[8][3]
    prec = dev[8][2]
    writer.writerow([model,accuracy,recall,prec,f1])

    # print([model,context,batch,accuracy,f1,recall,prec])
import csv
pth = 'results english/'
files = os.listdir(pth)
files.sort()
task = 'dev'

f = open('results_task2'+task+'.csv', 'w')
writer = csv.writer(f)
writer.writerow(['model','accuracy','recall','prec','f1'])
# task = 'dev
# print(files)
#exit()
# if 'dev' in 'indicbertcontext15015dev_report.txt':
#     print("dell")
# exit()
for file in files:
    print(file)
    if task in file:
        # if '150' in file:
        #     print('150')
        make_report(pth+file,writer,task)
    
f.close()


