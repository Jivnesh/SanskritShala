from django.http.response import Http404, HttpResponseRedirect
from django.shortcuts import render, get_object_or_404, redirect, HttpResponse
from . import models, forms, codeforline
from .models import Sentences, WordOptions, Wordsinsentence, User, Noun, Indeclinables, Verbs, Exsentences
from .tables import WordOptionsTable, SentencesTable, WordsinsentenceTable
import json
from django_datatables_view.base_datatable_view import BaseDatatableView
import random
import os
import time
from subprocess import call
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

import requests
from bs4 import BeautifulSoup
import csv

# renders response for index page
def index(request):
    return render(request, 'annotatorapp/index.html', {})


# returns an HttpResponse object with that rendered text.
def lineview(request):
    return render(request, 'annotatorapp/index.html', {})


# Combines annotatorapp/tables.html with the given context tabledata and returns an HttpResponse object with that rendered text.
# data is collected from the tables structure specified in tables.py
def wordtableview(request):
    tabledata = WordOptionsTable(WordOptions.objects.all())
    return render(request, 'annotatorapp/tables.html', {'tabledata': tabledata})


# Combines annotatorapp/tables.html with the given context tabledata and returns an HttpResponse object with that rendered text.
# data is collected from the tables structure specified in tables.py
def sentenceview(request):
    tabledata = SentencesTable(Sentences.objects.all())
    return render(request, 'annotatorapp/tables.html', {'tabledata': tabledata})


# Combines the template with the context and returns an HttpResponse object with that rendered text.
# data is collected from the tables structure specified in tables.py
def wordsinsentenceview(request):
    tabledata = WordsinsentenceTable(Wordsinsentence.objects.all())
    return render(request, 'annotatorapp/tables.html', {'tabledata': tabledata})

#renders a list consisting of lines and ids
def xsentenceview(request,batch_id):
    l = [0,20,40,60,80,100,120,140,160,180]
    r = l[batch_id]
    ids = Exsentences.objects.values('xsent_id')[r:r+20]
    line = Exsentences.objects.values('line')[r:r+20]
    chunks = Exsentences.objects.values('chunks')[r:r+20]
    lemmas = Exsentences.objects.values('lemmas')[r:r+20]
    morph_cng = Exsentences.objects.values('morph_cng')[r:r+20]
    lists = zip(ids,line,chunks,lemmas,morph_cng)
    return render(request, 'annotatorapp/exsent.html', {'lists':lists,'batch_id':batch_id})

# for rendering response  upon obtaining saved word data (present as Draggable operators) from the database
def get_dragdata(request):
    if request.is_ajax():
        if request.method == 'POST':
            sent_id = json.loads(request.POST['sentid'])
            Sentence1 = Sentences.objects.get(id=sent_id)
            wordsdata = WordOptions.objects.all().filter(sentence=Sentence1)
            data = codeforline.getsentwordtree(sent_id);
            print(data)
            return HttpResponse(data)
    else:
        raise Http404


# for rendering response upon saving the current selected data to database
def save_dragdata(request):
    if request.is_ajax():
        if request.method == 'POST':
            wp = json.loads(request.POST['wp'])
            wc = json.loads(request.POST['wc'])
            wr = json.loads(request.POST['wr'])
            sent_id = json.loads(request.POST['sentid'])
            Sentence1 = Sentences.objects.get(id=sent_id)
            wordsdata = WordOptions.objects.all().filter(sentence=Sentence1)
            for w in wordsdata:
                try:
                    w.isSelected = False
                    w.isEliminated = True
                    w.parent = -1
                    w.relation = ''
                    w.children = ''
                    w.save()
                except Exception as e:
                    print("wordsdata updated in ajex save_dragdata:selection elimination ")
                    print(e)
            for i in wp:
                try:
                    w = WordOptions.objects.get(id=i)
                    w.parent = int(wp[i])
                    w.isSelected = True
                    w.isEliminated = False
                    w.save()
                except Exception as e:
                    print("Wordsinsentencenot updated in ajex save_dragdata:wp ")
                    print(e)
            for i in wr:
                try:
                    w = WordOptions.objects.get(id=i)
                    w.relation = wr[i]
                    w.isSelected = True
                    w.isEliminated = False
                    w.save()
                except Exception as e:
                    print("Wordsinsentencenot updated in ajex save_dragdata:wr ")
                    print(e)
            for i in wc:
                try:
                    w = WordOptions.objects.get(id=i)
                    w.children = w.children + wc[i]
                    w.isSelected = True
                    w.isEliminated = False
                    w.save()
                except Exception as e:
                    print("Wordsinsentencenot updated in ajex save_dragdata:wc ")
                    print(e)
            return HttpResponse("Success!")
    else:
        raise Http404

sentencetype = ""
sentence = ""

#function that checks if input sentence is present in database otherwise sends request to SHR for data scrap.
#returns a dictionary and pandas dataframe with the data
def presentdataview(request):
    saveline = True
    if request.method == 'GET':
        try:
            Sentence = Sentences(
                line=request.GET.get('line',''),
                linetype=request.GET.get('linetype',''),
            )

            # global variables to use in other functions
            global sentencetype
            global sentence
            sentencetype = request.GET['linetype']
            sentence = request.GET['line']

            if not codeforline.checksent(Sentence):  # if new sentence appears
                dict_ = codeforline.getdatafromsite(Sentence)
                df = dict_['t']
                line_header = dict_['line_header']
                # print("hello "+ line_header)
                if saveline:
                    Sentence.line_header = line_header
                    Sentence.save()
                    codeforline.savedatafromsite(df, Sentence)
                    print("Adding Sentences data to Database \n\n")
            if codeforline.checksent(Sentence):
                Sentence1 = Sentences.objects.get(line=Sentence.line, linetype=Sentence.linetype)
                wordsdata = WordOptions.objects.all().filter(sentence=Sentence1)
                words = Sentence1.line.split(' ')
                chunknum = {}
                c = 0
                for word in words:
                    c = c + 1
                    chunknum[word] = c
                sent_id = Sentence1.id
                pos = 0
                # _dict = {'sent_id':sent_id,'line_header':line_header} 
                context = codeforline.contestofwordsdata(sent_id)
                return render(request, 'annotatorapp/presentdata.html',context)
            else:
                wordsdata = codeforline.worddataofsentence(df, Sentence)
                return render(request, 'annotatorapp/presentdata.html',
                              {'wordsdata': wordsdata, 'words': Sentence.line.split(' ')})
        except Exception as e:
            print("Sentence not inserted : ")
            print(e)
        Sentences1 = Sentences.objects.all()
        for s in Sentences1:
            sent_id = s.id
            break
        return render(request, 'annotatorapp/presentdata.html', {'sentid': sent_id})
    if request.method == "POST":
        Inputlineform = forms.inputlineform(request.POST)
        if Inputlineform.is_valid():
            print('form is valid')
            try:
                Sentence = Sentences(
                    line=Inputlineform.cleaned_data['line'],
                    linetype=Inputlineform.cleaned_data['linetype'],
                )
                sentencetype = Inputlineform.cleaned_data['linetype']

                if not codeforline.checksent(Sentence):  # if new sentence appears
                    dict_ = codeforline.getdatafromsite(Sentence)
                    df = dict_['t']
                    line_header = dict_['line_header']
                    # print("hello "+ line_header)
                    if saveline:
                        Sentence.line_header = line_header
                        Sentence.save()
                        codeforline.savedatafromsite(df, Sentence)
                        print("Adding Sentences data to Database \n\n")
                if codeforline.checksent(Sentence):
                    Sentence1 = Sentences.objects.get(line=Sentence.line, linetype=Sentence.linetype)
                    wordsdata = WordOptions.objects.all().filter(sentence=Sentence1)
                    words = Sentence1.line.split(' ')
                    chunknum = {}
                    c = 0
                    for word in words:
                        c = c + 1
                        chunknum[word] = c
                    sent_id = Sentence1.id
                    pos = 0
                    # _dict = {'sent_id':sent_id,'line_header':line_header} 
                    context = codeforline.contestofwordsdata(sent_id)
                    return render(request, 'annotatorapp/presentdata.html',context)
                else:
                    wordsdata = codeforline.worddataofsentence(df, Sentence)
                    return render(request, 'annotatorapp/presentdata.html',
                                    {'wordsdata': wordsdata, 'words': Sentence.line.split(' ')})
            except Exception as e:
                print("Sentence not inserted : ")
                print(e)
        Sentences1 = Sentences.objects.all()
        for s in Sentences1:
            sent_id = s.id
            break
        return render(request, 'annotatorapp/presentdata.html', {'sentid': sent_id})
    else:
        Sentence1 = Sentences.objects.get(id=request.session.get('sent_id'))
        wordsdata = WordOptions.objects.all().filter(sentence=Sentence1)
        words = Sentence1.line.split(' ')
        chunknum = {}
        c = 0
        for word in words:
            c = c + 1
            chunknum[word] = c
        sent_id = Sentence1.id
        pos = 0
        context = codeforline.contestofwordsdata(sent_id)
        return render(request, 'annotatorapp/presentdata.html', context)

def select_wordoptionview(request, sent_id, wordoption_id):
    wo = WordOptions.objects.get(id=wordoption_id)
    wo.isSelected = True
    request.session['sent_id'] = sent_id
    wo.save()
    return redirect('annotatorapp:presentdataview')


# for eliminating the conflicting segments
def eliminate_wordoptionview(request, sent_id, wordoption_id):
    wo = WordOptions.objects.get(id=wordoption_id)
    wo.isEliminated = True
    wo.save()
    request.session['sent_id'] = sent_id
    return redirect('annotatorapp:presentdataview')


# for resetting every selected segment back to the initial position
def reset_allselectionview(request, sent_id):
    # collecting required values
    Sentence1 = Sentences.objects.get(id=sent_id)
    wordsdata = WordOptions.objects.all().filter(sentence=Sentence1)
    # iterating through the collected values and initializing them
    for wo in wordsdata:
        wo.isSelected = False
        wo.isEliminated = False
        wo.parent = -1
        wo.relation = ''
        wo.children = ''
        wo.save()
    request.session['sent_id'] = sent_id
    return redirect('annotatorapp:presentdataview')


# rendering response for saving details of each data segment(flowchart data) clicked by user
def save_data_to_db(request):
    if request.is_ajax():
        if request.method == 'POST':
            # load the data to be saved into model
            it = json.loads(request.POST['it'])
            et = json.loads(request.POST['et'])
            cs = json.loads(request.POST['cs'])
            ss = json.loads(request.POST['ss'])
            user = User(savedSentence=ss, clickSequence=cs, init_time=it, end_time=et)
            user.save()
            return HttpResponse('Success')
    else:
        raise Http404

#used to retrieve autocomplete noun/verbs/indeclinables options
def get_form_data(request):
    if request.is_ajax():
        if request.method == 'POST':
            table_id = json.loads(request.POST['table_id'])
            if table_id == 'noun':
                data = Noun.objects.values_list('sh');
            elif table_id == 'verb':
                data = Verbs.objects.values_list('sh');
            elif table_id == 'ind':
                data = Indeclinables.objects.values_list('sh');
            return HttpResponse(data)
    else:
        raise Http404

def get_sol_data(request):
    if request.is_ajax():
        if request.method == 'POST':
            try:
                line = json.loads(request.POST['line'])
            except Exception as e:
                print(e)
            data = {}
            xsent = Exsentences.objects.filter(line=line)
            data['id'] = xsent[0].xsent_id
            data['chunks'] = xsent[0].chunks
            data['lemmas'] = xsent[0].lemmas
            data['morph_cng'] = xsent[0].morph_cng
            return HttpResponse(json.dumps(data))
    else:
        raise Http404

def go_to_prodigy(request):
    print("Hello")
    if request.is_ajax():
        if request.method == 'POST':
            from indic_transliteration import sanscript
            from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
            
            data = request.POST['savefile']
            # print(data.split('\n'))
            content = data.split('\n')
            words = []
            tags = []
            con = ""
            for i in range(len(content)-1):
                words.append(content[i].split(',')[0])
                tags.append(content[i].split(',')[1])
                text = transliterate(words[-1], sanscript.IAST, sanscript.SLP1)
                con += str(i+1)+'\t'+text+'\t'+tags[-1]+'\t'+'_'+'\t'+'0'+'\t'+'_'
                if i!=len(content)-2:
                    con+='\n'
            file = open('/home/jivnesh/SCL_Platform/San-SOTA/data/ud_pos_ner_dp_prose_san', 'w')
            print(con)
            
            # Writing a string to file
            file.write(con)
            
            # Closing file
            file.close()

            from subprocess import call
            with open('./prodigy.sh', 'rb') as file:
                script = file.read()
            rc = call(script,shell=True)
            for i in range(9):
                time.sleep(1)
            stat = {}
            stat['status'] = "right"
            return HttpResponse(json.dumps(stat))
    else:
        raise Http404

# def predict_morph_tag(request):
#     print("Python->In file views.py->running code for Extracting Tags")
#     if request.is_ajax():
#         if request.method == 'POST':
#             from indic_transliteration import sanscript
#             from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
#             print(request.POST)
#             data = request.POST['words[]']
#             final = ""
#             for word in data:
#                 if sentencetype!="Slp":
#                     word = transliterate(word, sentencetype.lower(), sanscript.SLP1)
#                 final += word+'\t'+'_'+'\t'+'_'+'\n'

#             # file = open('/home/jivnesh/SCL_Platform/San-SOTA/data/ud_pos_ner_dp_prose_san', 'w')
#             file = open('/home/jivnesh/SanShala-Models/LemmaTag/data/test.txt', 'w')

#             # Writing a string to file
#             file.write(final)
            
#             # Closing file
#             file.close()

#             # For Web Scraping 
#             f = open("/home/guest/Documents/SanShala-Models/LemmaTag/Morph_Scrapper/1.txt", "w")
#             st = ""
#             st = ' '.join(data)+'\n'
#             f.write(st)
#             f.close()

#             process = subprocess.Popen(["/bin/sh", "./extract_tag.sh"])
#             process.wait()

#             data1 = pd.read_csv('/home/guest/Documents/SanShala-Models/LemmaTag/Morph_Scrapper/details/0.csv',sep=',')
#             line = st
#             text = transliterate(line, sanscript.SLP1,sanscript.IAST)
#             candidate_space = []
#             for word in text.split():
#                 candidate_space.append(list(set(list(data1[data1['word']==word]['morph']))))
            
#             tags_word = candidate_space

#             # recommendation by first model
#             other_tags_recommendation = {}
#             for i in range(len(tags_word)):
#                 temp = []
#                 for j,tag in enumerate(tags_word[i]):
#                     temp.append([j,tag])
#                 other_tags_recommendation[i] = temp

#             # Running second model for single prediction
#             process = subprocess.Popen(["/bin/sh", "./extract_tag2.sh"])
#             process.wait()

#             file = open('/home/jivnesh/SanShala-Models/LemmaTag/logs/prediction_morph_tag/taglem_test_ep0.txt', 'r')
#             predicted_result = file.read()
#             predicted_result = predicted_result.split('\n')
#             predicted_words = []
#             predicted_tags = []
#             for i in range(len(predicted_result)-1):
#                 temp = predicted_result.split('\t')
#                 predicted_words.append(temp[0])
#                 predicted_tags.append(temp[2])

#             stat = {}
#             stat['word'] = predicted_words
#             stat['tags'] = predicted_tags
#             return HttpResponse(json.dumps(stat))
#     else:
#         raise Http404

def predict_segmentation(request):
    if request.is_ajax():
        if request.method == 'POST':
            try:
                line = json.loads(request.POST['line'])
            except Exception as e:
                print(e)
            data = {}
            
            print("Sentence type is : ",sentencetype)
            xsent = Exsentences.objects.filter(line=line)

            dic = {"WX":"wx", "SLP":"slp1", "Velthuis":"velthuis"}
            line = transliterate(line, dic[sentencetype], sanscript.SLP1)
            
            import subprocess

            p = subprocess.Popen(("/bin/sh", "./extract_segmentation.sh", line))
            p.wait()
            filename = "/home/jivneshs/SanShala-Models/TransLIST/V0/output.txt"
            f = open(filename, "r")
            final_segmentation = f.read()
            final_segmentation = transliterate(final_segmentation, sanscript.SLP1, sanscript.IAST)
            data['lemmas'] = final_segmentation.split(" ")
            return HttpResponse(json.dumps(data))
    else:
        raise Http404

def go_to_lemma(request):
    if request.is_ajax():
        if request.method == 'POST':
            data = {}
            line = request.POST['line']
            website = "http://10.14.84.23:4000/predict?sen="+line+"&in=IAST&out=Devanagari"
            # import webbrowser
            # webbrowser.open(website)
            return HttpResponseRedirect(website)
            # response = redirect(website)
            # return response
            # return redirect(website)
            # return HttpResponse(json.dumps(data))
    else:
        raise Http404
