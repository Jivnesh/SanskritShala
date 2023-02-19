import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import codecs
import time
import re
import os
from . import models
from .models import Sentences, WordOptions, Wordsinsentence
import json


def getdatafromsite(inputsent):  # Scrapping data from site
    inputline = inputsent.line
    inputtype = inputsent.linetype
    problem = []
    pbwords = []
    s_type = {}
    s_type['WX'] = 'WX'
    s_type['SLP'] = 'SL'
    s_type['Velthuis'] = 'VH'
    s_type['KH'] = 'KH'

    s_d = inputline

    s_c = s_d.replace(" ", "+")
    # for utilising the sanskrit heritage app, the url has been specified
    urlname = ("http://sanskrit.inria.fr/cgi-bin/SKT/sktgraph.cgi?lex=SH&st=t&us=f&cp=t&text=" +
               s_c + "&t=" + s_type[inputtype] + "&topic=&mode=g&corpmode=&corpdir=&sentno=")

    print(urlname)
    page = requests.get(urlname)
    # parsing using beautifulsoup
    soup = bs(page.text, 'html.parser')
    print(soup)
    table = soup.table
    tablebody = table.find('table', {'class': 'center'})
    t = pd.DataFrame(
        columns=['id', 'level', 'color_class', 'position', 'chunk_no', 'word', 'lemma', 'pre_verb', 'morph', 'colspan',
                 'wordlenth', 'aux_inf'])

    i = 0
    id_ = 0
    if not (tablebody):  #### wronginputs
        print('no table body of given inputline')

    # for valid entries corresponding to Wordsinsentence
    for child in tablebody.children:
        if (child.name == 'tr'):
            if i < 1:
                linechar = []
                c = 0
                for char in child.children:
                    linechar.append(char.string)
                    c += 1
                i += 1
                line_header = "".join(linechar)
                linechunks = line_header.split("\xa0")
                continue
            position_ = 0
            j = 0
            for wordtable in child.children:
                c = 0
                for ch in linechar[0:position_]:
                    if (re.match('\xa0', ch)):  # or (re.match('_',ch))
                        c += 1
                    # if the contents exist in wordtable
                    # following assignings are carried out.
                if (wordtable.contents):
                    color_ = wordtable.table.get('class')[0]
                    colspan_ = wordtable.get('colspan')
                    word_ = wordtable.table.tr.td.string
                    onclickdatas_ = wordtable.table.tr.td.get('onclick')
                    show_box_data = str(re.search(r'showBox\(\'(.*?)\'', onclickdatas_).group(1))
                    for onclickdata_ in show_box_data.split("<br>"):  # required splits carried out at positions stated
                        filter_data_ = str(re.sub(r'</?a.*?>|</?i>| ✘', "", onclickdata_))
                        morphslist_ = re.findall(r'{\s?(.*?)\s?}', filter_data_)  # .split(' | ')
                        lsearch = re.search(r'\[(.*)\]\{|\}\[(.*)\]', filter_data_)
                        ldata = ""
                        if (not (lsearch == None)):
                            ldata = str(lsearch.group(1)) if (not (lsearch.group(1) == None)) else str(lsearch.group(2))
                        lemmas_ = str(re.sub(r'\[(.*)\]|{\s?(.*?)\s?}|\s', "", ldata))
                            
                        lemmadata = ldata.split(" ")
                        if len(lemmadata) > 1:
                            auxi_ = " ".join(lemmadata[1:])
                        else:
                            auxi_ = ""
                        lemmas_ = "".join(lemmadata[0])
                        lemmalists_ = lemmas_.split("-")
                        if (len(lemmalists_) > 1):
                            preverb_ = ",".join(lemmalists_[0:(len(lemmalists_) - 1)])
                            lemmalist_ = "".join(lemmalists_[-1:]).split("_")
                        else:
                            preverb_ = ""
                            lemmalist_ = "".join(lemmalists_[0]).split("_")
                        if (len(lemmalist_) > 1):
                            auxi_ = auxi_ + " sence of lemma = " + "".join(lemmalist_[1:(len(lemmalist_))])
                            lemma_ = "".join(lemmalist_[0])
                        else:
                            lemma_ = "".join(lemmalist_[0])
                        morphs_ = str(morphslist_[0])
                        for morph_ in morphs_.split(" | "):
                            t.loc[id_] = [id_, i, color_, position_, c + 1, word_, lemma_, preverb_, morph_,
                                          int(colspan_), len(word_), auxi_]
                            if (re.match(r'grey_back', color_)):
                                if not (word_ == 'pop'):
                                    problem.append(id_)  # filling entries to problem list
                                else:
                                    id_ = id_ - 1
                            id_ += 1

                    position_ += int(colspan_)
                else:
                    position_ += 1
            i = i + 1
            dict_ = {'t':t,'line_header':line_header}
    return dict_


def savedatafromsite(df, sent):
    ##captures sentences from the site and ssaves them to WordOptions model
    for i in range(df.shape[0]):
        row = df.iloc[i]  # indexing based on postion
        try:
            word_option = WordOptions(sentence=sent,
                                      level=row["level"],
                                      color_class=row["color_class"],
                                      position=row["position"],
                                      chunk_no=row["chunk_no"],
                                      lemma=row["lemma"],
                                      pre_verb=row["pre_verb"],
                                      morph=row["morph"],
                                      colspan=row["colspan"],
                                      wordlength=row["wordlenth"],
                                      aux_info=row["aux_inf"],
                                      word=row["word"],
                                      )
            word_option.save()  # saving the class object

        except Exception as e:
            print("WordOption not inserted : ", sent_id, "/", word)
            print(e)

    line = sent.line
    cno = 0
    for wd in line.split(' '):
        cno = cno + 1
        try:
            wordsinsentence = Wordsinsentence(sentence=sent,
                                              word=wd,
                                              parent=-1,
                                              children='',
                                              relation='',
                                              wordoptions='',
                                              chunkno=cno
                                              )
            wordsinsentence.save()
        except Exception as e:
            print("Wordsinsentence not inserted : ")
            print(e)


def worddataofsentence(df, sent):
    words = []
    ##related data for WordOptions model to handle values
    for i in range(df.shape[0]):
        row = df.iloc[i]
        word_option = WordOptions(sentence=sent,
                                  level=row["level"],
                                  color_class=row["color_class"],
                                  position=row["position"],
                                  chunk_no=row["chunk_no"],
                                  lemma=row["lemma"],
                                  pre_verb=row["pre_verb"],
                                  morph=row["morph"],
                                  colspan=row["colspan"],
                                  wordlength=row["wordlenth"],
                                  aux_info=row["aux_inf"],
                                  word=row["word"],
                                  )
        words.append(WordOptions)
    return words


def checksent(sent):
    return Sentences.objects.filter(line=sent.line, linetype=sent.linetype).exists()


def conflicts(df, sent):
    # pass is just a placeholder for functionality to be added later.
    pass


#retruns string converted from df object from WordOptions model and performing required modifications.
def getsentwordtree(sent_id):
    df = pd.DataFrame(columns=['id', 'wordid', 'word', 'morph', 'lemma', 'rel', 'parent'])
    Sentence1 = Sentences.objects.get(id=sent_id)#extracting sentences corresponding to given id
    wordsdata = WordOptions.objects.all().filter(sentence=Sentence1)
    temp = {}
    df = {}

    i = 0
    temp[-1] = 0
    for wd in wordsdata:
        if wd.isSelected:
            temp[wd.id] = i + 1
            i = i + 1

    i = 0
    #adding further details involving lemma for the input .
    for wd in wordsdata:
        if wd.isSelected:
            lemma = wd.lemma
            if not str(wd.pre_verb) == '':
                lemma = wd.pre_verb + '-' + lemma
            if not str(wd.aux_info) == '':
                if wd.aux_info[-18:-1] == 'sence of lemma = ':
                    lemma = lemma + '-' + wd.aux_info[-1:]
                    if not str(wd.aux_info[:-18]) == ' ':
                        lemma = lemma + ' (' + wd.aux_info[:-18] + ')'
                else:
                    lemma = lemma + ' (' + wd.aux_info + ')'

            df[i + 1] = [i + 1, wd.id, wd.word, wd.morph, lemma, wd.relation, temp[wd.parent]]
            i = i + 1
    return json.dumps(df)

#this function helps decide the conflicts asociated with each segment in the input sentence
def contestofwordsdata(sent_id):
    # print(sent_id['line_header'])
    Sentence1 = Sentences.objects.get(id=sent_id)
    wordsdata = WordOptions.objects.all().filter(sentence=Sentence1)
    wordsinsentence = Wordsinsentence.objects.all().filter(sentence=Sentence1)
    chunkwordids = {}
    for dw in wordsinsentence:
        chunkwordids[dw.chunkno] = dw.id

    df = pd.DataFrame(
        columns=['wordid', 'level', 'color_class', 'position', 'chunk_no', 'lemma', 'pre_verb', 'morph', 'colspan',
                 'wordlength', 'aux_info', 'word', 'subminp', 'maxp', 'endposition'])

    i = 0
    for wd in wordsdata:
        df.loc[i] = [wd.id, wd.level, wd.color_class, wd.position, wd.chunk_no, wd.lemma, wd.pre_verb, wd.morph,
                     wd.colspan, wd.wordlength, wd.aux_info, wd.word, 0, 0, wd.position + wd.colspan]
        i += 1

    wordfromchunk = {}
    colspanofchunk = {}
    for c in df.chunk_no.unique():
        df1 = df.loc[df['chunk_no'] == c]
        minp = min(df1.position)
        for i in df1.index:
            df.loc[i, 'subminp'] = df.loc[i, 'position'] - minp

            df.loc[i, 'maxp'] = df.loc[i, 'subminp'] + df.loc[i, 'colspan']

        wordfromchunk[c] = []
        df1 = df.loc[df['chunk_no'] == c]
        for w in df1.word.unique():
            wordfromchunk[c].append(w)
        colspanofchunk[c] = max(df1['maxp'])

    words = df.word.unique()
    levelofword = {}
    posofword = {}
    idsofword = {}
    colspanofword = {}
    for w in words:
        levelofword[w] = min(df.loc[df['word'] == w].level)
        posofword[w] = min(df.loc[df['word'] == w].subminp)
        colspanofword[w] = max(df.loc[df['word'] == w].colspan)
        idsofword[w] = df.loc[df['word'] == w].wordid.unique()
    sentwords = Sentence1.line.split(' ')
    chunknum = {}
    c = 0
    for sw in sentwords:
        c = c + 1
        chunknum[sw] = c

    maxlevel = max(df.level)
    levelrange = range(1, maxlevel + 1)
    chunkrange = range(1, max(df.chunk_no) + 1)
    positionrange = range(max(df['position']) + 1)
    maxpos = max(df['position'] + df['colspan'])
    levelpos = {}
    levelwordpos = {}
    for l in levelrange:
        levelpos[l] = []
        levelwordpos[l] = []
        df1 = df.loc[df['level'] == l]
        for p in df1.position.unique():
            levelwordpos[l].append(p)
        for p in positionrange:
            check = True
            for i in df1.index:
                if (p == df1.loc[i, 'position']) or ((p > df1.loc[i, 'position']) and (p < df1.loc[i, 'endposition'])):
                    check = False
                    break
            if check:
                levelpos[l].append(p)
    dragdata = {}
    links = {}
    ic = 0

    # for dw in wordsinsentence :
    for dw in wordsdata:
        if dw.isSelected:
            lemma = dw.lemma
            if not str(dw.pre_verb) == '':
                lemma = dw.pre_verb + '-' + lemma
            if not str(dw.aux_info) == '':
                if dw.aux_info[-18:-1] == 'sence of lemma = ':
                    lemma = lemma + '-' + dw.aux_info[-1:]
                    if not str(dw.aux_info[:-18]) == ' ':
                        lemma = lemma + ' (' + dw.aux_info[:-18] + ')'
                else:
                    lemma = lemma + ' (' + dw.aux_info + ')'
            data1 = {

                "properties": {
                    "title": str(dw.id) + ' : ' + dw.word + '<br>[' + lemma + ']',
                    "inputs": {
                        "in-" + str(dw.id): {
                            "label": dw.morph
                        }
                    },
                    "outputs": {
                        "out-" + str(dw.id): {
                            "label": ' '
                        }
                    }
                }
            }
            # print('here')
            dragdata['word_' + str(dw.id)] = data1

            if not dw.parent == -1:
                link1 = {

                    "fromOperator": 'word_' + str(dw.parent),
                    "fromConnector": "out-" + str(dw.parent),
                    "fromSubConnector": '0',
                    "toOperator": 'word_' + str(dw.id),
                    "toConnector": "in-" + str(dw.id),
                    "toSubConnector": "0",
                    "relationame": dw.relation

                }
                links[ic] = link1
                ic = ic + 1

    conflictslp = {};
    conflictslp1 = {};
    conflictslp1color = {}
    for i in df.index:
        conflictslp[
            str(df.loc[i].level) + '-' + str(df.loc[i].position) + '-' + str(df.loc[i].endposition) + '-' + df.loc[
                i].color_class] = []
    for key in conflictslp.keys():
        l = int(key.split('-')[0])
        p = int(key.split('-')[1])
        e = int(key.split('-')[2])

        for i in df.index:
            if (l == df.loc[i].level) and ((df.loc[i].position > p) and df.loc[i].position < e):
                if not str(df.loc[i].level) + '-' + str(df.loc[i].position) in conflictslp[key]:
                    conflictslp[key].append(str(df.loc[i].level) + '-' + str(df.loc[i].position))
            if not (l == df.loc[i].level):
                if ((df.loc[i].position > p - 1) and df.loc[i].position < e - 1):
                    if not str(df.loc[i].level) + '-' + str(df.loc[i].position) in conflictslp[key]:
                        conflictslp[key].append(str(df.loc[i].level) + '-' + str(df.loc[i].position))
            if ((df.loc[i].position < p) and df.loc[i].endposition > p + 1):
                conflictslp[key].append(str(df.loc[i].level) + '-' + str(df.loc[i].position))

    for key in conflictslp:
        conflictslp1[key.split('-')[0] + '-' + key.split('-')[1]] = conflictslp[key]
        conflictslp1color[key.split('-')[0] + '-' + key.split('-')[1]] = key.split('-')[3]

    #context dictionary containing every detailed bit of the word in the sentence
    context = {'line': Sentence1.line,'line_header':Sentence1.line_header, 'wordsdata': wordsdata, 'words': sentwords, 'chunknum': chunknum,
               'sentid': sent_id, 'dragdata': json.dumps(dragdata), 'links': json.dumps(links),
               'conflictslp': json.dumps(conflictslp1), 'colorlp': json.dumps(conflictslp1color),
               'levelofword': levelofword, 'levelrange': levelrange, 'posofword': posofword, 'idsofword': idsofword,
               'wordfromchunk': wordfromchunk, 'chunkrange': chunkrange, 'colspanofchunk': colspanofchunk,
               'colspanofword': colspanofword,
               'allwords': words, 'positionrange': positionrange, 'levelpos': levelpos, 'levelwordpos': levelwordpos,
               'wordsinsentence': wordsinsentence, 'chunkwordids': chunkwordids
               }

    #handling the corner cases in sandhi check from the file named all_sandhi.txt
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'all_sandhi.txt')
    s = pd.read_csv(path, encoding='utf-8', sep=',')
    df_2 = pd.DataFrame(data=s)
    keys = conflictslp1.keys()
    for key in keys:
        value = conflictslp1[key]
        l = int(key.split('-')[0])
        p = int(key.split('-')[1])

        #extracting word from dataframe of words corresponding to key level and positon
        word_df1 = df[(df['level'] == l) & (df['position'] == p)]
        word_df1 = word_df1['word'].values[0]

        if len(value) == 0:
            print("no conflicts")
        elif len(value) != 0:
            for v in value:
                lv = int(v.split('-')[0])
                pv = int(v.split('-')[1])

                # extracting word from dataframe of words corresponding to value of the key level and positon

                word_df2 = df[(df['level'] == lv) & (df['position'] == pv)]
                word_df2 = word_df2['word'].values[0]
                if p > pv:
                    #c1 = word_df2
                    #c2 = word_df1
                    if len(word_df2) < len(word_df1):
                        n = len(word_df2)
                    else:
                        n = len(word_df1)
                    t = word_df2[-n:]
                    d = 0

                    for l1, l2 in zip(t, word_df1):
                        if l1 == l2:
                            d = d + 1

                    if n > 2 :
                        print("full conflict : characters greater than 2")
                    elif d == 2:
                        C2 = word_df1[:2]
                        C1 = word_df2[-2:]
                        k = 0
                        for q in df_2.loc[df_2['c2'] == C2].c1:
                            if q == C1:
                                k = k + 1
                        if k == 0:
                            print("conflict")
                        else:
                            value.remove(str(lv) + '-' + str(pv))
                    else:
                        C2 = word_df1[:1]
                        C1 = word_df2[-1:]
                        k = 0
                        for q in df_2.loc[df_2['c2'] == C2].c1:
                            if q == C1:
                                k = k + 1
                        if k == 0:
                            print("conflict")
                        else:
                            #removing the cases of overlapping and sandhi
                            value.remove(str(lv) + '-' + str(pv))

                elif pv > p:
                    #c1 = word_df1
                    #c2 = word_df2
                    if len(word_df2) < len(word_df1):
                        n = len(word_df2)
                    else:
                        n = len(word_df1)
                    t = word_df1[-n:]
                    d = 0

                    for l1, l2 in zip(t, word_df2):
                        if l1 == l2:
                            d = d + 1

                    if n > 2:
                        print("full conflict : characters greater than 2")
                    elif d == 2:
                        C1 = word_df1[-2:]
                        C2 = word_df2[:2]
                        k = 0
                        for q in df_2.loc[df_2['c2'] == C2].c1:
                            if q == C1:
                                k = k + 1
                        if k == 0:
                            print("conflict")
                        else:
                            value.remove(str(lv) + '-' + str(pv))
                    else:
                        C1 = word_df1[-1:]
                        C2 = word_df2[:1]
                        k = 0
                        for q in df_2.loc[df_2['c2'] == C2].c1:
                            if q == C1:
                                k = k + 1
                        if k == 0:
                            print("conflict")
                        else:
                            value.remove(str(lv) + '-' + str(pv))


                else:
                    l1 = len(word_df1)
                    l2 = len(word_df2)
                    if l1 <= l2:
                        d = 0
                        #c1 = word_df1
                        #c2 = word_df2
                        for letter1, letter2 in zip(word_df1, word_df2):
                            if letter1 == letter2:
                                d = d + 1
                            else:
                                break

                        if l1 > 2:
                            print("full conflict : characters greater than 2")
                        elif d == 2:
                            C1 = word_df1[-2:]
                            C2 = word_df2[:2]
                            k = 0
                            for q in df_2.loc[df_2['c2'] == C2].c1:
                                if q == C1:
                                    k = k + 1
                            if k == 0:
                                print("3 conflict")
                            else:
                                value.remove(str(lv) + '-' + str(pv))
                        else:
                            C1 = word_df1[-1:]
                            C2 = word_df2[:1]
                            k = 0
                            for q in df_2.loc[df_2['c2'] == C2].c1:
                                if q == C1:
                                    k = k + 1
                            if k == 0:
                                print("conflict")
                            else:
                                value.remove(str(lv) + '-' + str(pv))
                    else:
                        #c1 = word_df2
                        #c2 = word_df1
                
                        d = 0
                        for letter1, letter2 in zip(word_df1, word_df2):
                            if letter1 == letter2:
                                d = d + 1
                            else:
                                break

                        if l2 > 2:
                            print("full conflict : characters greater than 2")
                        elif d == 2:
                            C2 = word_df1[:2]
                            C1 = word_df2[-2:]
                            k = 0
                            for q in df_2.loc[df_2['c2'] == C2].c1:
                                if q == C1:
                                    k = k + 1
                            if k == 0:
                                print("conflict")
                            else:
                                value.remove(str(lv) + '-' + str(pv))
                        else:
                            C2 = word_df1[:1]
                            C1 = word_df2[-1:]
                            k = 0
                            for q in df_2.loc[df_2['c2'] == C2].c1:
                                if q == C1:
                                    k = k + 1
                            if k == 0:
                                print("conflict")
                            else:
                                value.remove(str(lv) + '-' + str(pv))

    context['allvar'] = context
    return context
