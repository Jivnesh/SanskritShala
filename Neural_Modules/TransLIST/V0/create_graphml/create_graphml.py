#!/usr/bin/env python
# coding: utf-8

# In[3]:


import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import re
import os
import sys
from requests.exceptions import Timeout
from devconvert import dev2slp, iast2slp, slp2dev, slp2iast, slp2tex, slp2wx, wx2slp, dev2wx
from pathlib import Path
import networkx as nx
import argparse

# In[5]:


def create_graph(data_frame):
#List indices for dataframe
#0-'id', 
#1-'level',
#2-'color_class',
#3-'position',
#4-'chunk_no',
#5-'word',
#6-'lemma',
#7-'sense',
#8-,'cng',
#9-'pre_verb',
#10-'morph',
#11-'colspan',
#12-'wordlenth',
#13-'aux_inf,
#14-'der_pre_verb',
#15-'der_lemma',
#16-'der_sense',
#17-'der_morph',
#18-'der_cng',
#19-'char_pos'

#    dict_ = {'source' : [], 'target' = [], 'weight' = []}
    keys = ['source', 'target', 'key']
    dict_ = dict.fromkeys(keys, [])
    edges = set()
    nodes = data_frame.values
    for node_1 in nodes:
        for node_2 in nodes:
            if (not (node_1[0] == node_2[0])):#for adding edges to only those which are not conflicting
#                if not ((node_1[19] <= node_2[19] <= (node_1[19] + node_1[12])) or (node_2[19] <= node_1[19] <= (node_2[19] + node_2[12]))) :                
                if ((node_1[19] <= (node_1[19] + node_1[12] - 1) <= node_2[19]) or (node_2[19] <= (node_2[19] + node_2[12] - 1) <= node_1[19])):
                    if ((not ((node_1[0], node_2[0], 1) in edges))):
                        edges.add((node_1[0], node_2[0], 1))
                else:
#                    if ((not ((node_1[0], node_2[0], 2) in edges)) and ((not ((node_2[0], node_1[0], 2) in edges)))):
                    if ((not ((node_1[0], node_2[0], 2) in edges))):
                        edges.add((node_1[0], node_2[0], 2))
    
    edges_list = list(edges)
    edges_df = pd.DataFrame(edges_list, columns = ['source', 'target', 'key'])
    
    g = nx.from_pandas_edgelist(edges_df, 'source', 'target', 'key', create_using = nx.DiGraph())
    for i in sorted(g.nodes()):
        nx.set_node_attributes(g, pd.Series(data_frame.level, index=data_frame.id).to_dict(), 'level')
        nx.set_node_attributes(g, pd.Series(data_frame.colspan, index=data_frame.id).to_dict(), 'colspan')
        nx.set_node_attributes(g, pd.Series(data_frame.aux_inf, index=data_frame.id).to_dict(), 'aux_inf')
        nx.set_node_attributes(g, pd.Series(data_frame.color_class, index=data_frame.id).to_dict(), 'color_class')
        nx.set_node_attributes(g, pd.Series(data_frame.chunk_no, index=data_frame.id).to_dict(), 'chunk_no')
        nx.set_node_attributes(g, pd.Series(data_frame.position, index=data_frame.id).to_dict(), 'position')
        nx.set_node_attributes(g, pd.Series(data_frame.length_word, index=data_frame.id).to_dict(), 'length_word')
        nx.set_node_attributes(g, pd.Series(data_frame.word, index=data_frame.id).to_dict(), 'word')
        nx.set_node_attributes(g, pd.Series(data_frame.der_pre_verb, index=data_frame.id).to_dict(), 'der_pre_verb')
        nx.set_node_attributes(g, pd.Series(data_frame.der_lemma, index=data_frame.id).to_dict(), 'der_lemma')
        nx.set_node_attributes(g, pd.Series(data_frame.der_sense, index=data_frame.id).to_dict(), 'der_sense')
        nx.set_node_attributes(g, pd.Series(data_frame.der_morph, index=data_frame.id).to_dict(), 'der_morph')
        nx.set_node_attributes(g, pd.Series(data_frame.der_cng, index=data_frame.id).to_dict(), 'der_cng')
        nx.set_node_attributes(g, pd.Series(data_frame.pre_verb, index=data_frame.id).to_dict(), 'pre_verb')
        nx.set_node_attributes(g, pd.Series(data_frame.lemma, index=data_frame.id).to_dict(), 'lemma')
        nx.set_node_attributes(g, pd.Series(data_frame.sense, index=data_frame.id).to_dict(), 'sense')
        nx.set_node_attributes(g, pd.Series(data_frame.morph, index=data_frame.id).to_dict(), 'morph')
        nx.set_node_attributes(g, pd.Series(data_frame.cng, index=data_frame.id).to_dict(), 'cng')
        nx.set_node_attributes(g, pd.Series(data_frame.char_pos, index=data_frame.id).to_dict(), 'char_pos')
        

    return g

def create_graphml(graph_, graphml_file_path):
    nx.write_graphml_xml(graph_, graphml_file_path)


# In[6]:


def getdatafromsite(inputsent, new_path, coding = 'SLP'):  # Scrapping data from site
#    print("\nInput Sentence : " + inputsent)
    
    inputline = inputsent
    inputtype = coding
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

    urlname = ("http://sanskrit.inria.fr/cgi-bin/SKT/sktgraph.cgi?lex=SH&st=t&us=f&cp=t&text=" + s_c + "&t=" + s_type[inputtype] + "&topic=&mode=g&corpmode=&corpdir=&sentno=")
    #urlname = ("http://172.29.92.176/cgi-bin/Heritage_Platform/sktgraph.cgi?lex=SH&st=t&us=f&cp=t&text=" + s_c + "&t=" + s_type[inputtype] + "&topic=&mode=g&corpmode=&corpdir=&sentno=")

#    urlname = ("http://localhost/cgi-bin/SKT/sktgraph.cgi?lex=SH&st=t&us=f&cp=t&text=" + s_c + "&t=" + s_type[inputtype] + "&topic=&mode=g&corpmode=&corpdir=&sentno=")

    print(urlname)
    try:
        page = requests.get(urlname, timeout = 15.0)
    except Timeout:
        print('Request Timout')
        return {}
    # parsing using beautifulsoup
    soup = bs(page.text, 'html.parser')
    table = soup.table
    tablebody = table.find('table', {'class': 'center'})
    t = pd.DataFrame(columns=['id', 'level', 'color_class', 'position', 'chunk_no', 'word', 'lemma', 'sense', 'cng', 'pre_verb', 'morph', 'colspan', 'length_word', 'aux_inf', 'der_pre_verb', 'der_lemma', 'der_sense', 'der_morph', 'der_cng', 'char_pos'])
#    t = pd.DataFrame(columns=['id', 'level', 'color_class', 'position', 'chunk_no', 'word', 'lemma', 'sense', 'cng', 'pre_verb', 'morph', 'colspan', 'length_word', 'aux_inf'])

    i = 0
    id_ = 0
    if not (tablebody):  #### wronginputs
        print('no table body of given inputline') 
        return {}
    
    cng_dict = {}
    
    with open(new_path, 'r') as f:
        for line in f:
            split_line = re.split(r'\t|\n', line)
            (key, val) = (split_line[0], split_line[1])
            cng_dict[key] = val
    
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
                pos_in_chunk = 0
                for ch in linechar[0:position_]:
                    if (re.match('\xa0', ch) or (re.match('_',ch))):  # or (re.match('_',ch))
                        c += 1
                        pos_in_chunk = 0
                    else:
                        pos_in_chunk += 1
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
                        msearch = re.search(r'\{\s?(.*)\s?\}\[|\]\{\s?(.*)\s?\}', filter_data_)
                        mdata = ""
                        ldata = ""
                        if (not (msearch == None)):
                            mdata = str(msearch.group(1)) if (not (msearch.group(1) == None)) else str(msearch.group(2))
                        if (not (lsearch == None)):
                            ldata = str(lsearch.group(1)) if (not (lsearch.group(1) == None)) else str(lsearch.group(2))
                        der_lemma_string = re.search(r'\[(.*)\]\{|\}\[(.*)\]', ldata)
                        der_morph_string = re.search(r'{\s?(.*?)\s?}', ldata)
                        lemmas_ = str(re.sub(r'\[(.*)\]|{\s?(.*?)\s?}|\s', "", ldata))
                            
                        if der_lemma_string == None:
                            auxi_ = ""
                            der_pre_verb = ""
                            der_lemma = ""
                            der_sense = "0"
                            der_morph = ""
                            der_cng = 1 # Not set - according to DCS's cng mapping
                        else:
                            der_lemma_string_value = der_lemma_string.group(1) if (not (der_lemma_string.group(1) == None)) else der_lemma_string.group(2)
                            der_lemma_lists_ = der_lemma_string_value.split("-")
                            if (len(der_lemma_lists_) > 1):
                                der_pre_verb = ",".join(der_lemma_lists_[0:(len(der_lemma_lists_) - 1)])
                                der_lemma_list = "".join(der_lemma_lists_[-1:]).split("_")
                            else:
                                der_pre_verb = ""
                                der_lemma_list = "".join(der_lemma_lists_[0]).split("_")
                            if (len(der_lemma_list) > 1):
                                der_lemma = "".join(der_lemma_list[0])
                                der_sense = "".join(der_lemma_list[1:(len(der_lemma_list))])
                            else:
                                der_lemma = "".join(der_lemma_list[0])
                                der_sense = "1"
                            if der_morph_string == None:
                                der_morph = ""
                            else:
                                der_morph = str(der_morph_string.group(1))
                                
                            if der_morph in cng_dict.keys():
                                der_cng = cng_dict[der_morph]
                                
                            auxi_ = ""
                                
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
                            sense_ = "".join(lemmalist_[1:(len(lemmalist_))])
                        else:
                            lemma_ = "".join(lemmalist_[0])
                            sense_ = "1"
                        # Temporarily assigning sense as 1 for those which do not have any sense attached explicitly. For those which have, their corresponding sense value is used

                        morphs_ = ldata
                        for morph_units in list(morphslist_):
                            for morph_ in morph_units.split(" | "):
                                if morph_ == der_morph:
                                    continue
                                cng_ = 0
                                if morph_ in cng_dict.keys():
                                    cng_ = cng_dict[morph_]
                                
                                t.loc[id_] = [id_, i, str(color_), pos_in_chunk, c + 1, str(word_), str(lemma_), sense_, int(cng_), str(preverb_), str(morph_), int(colspan_), int(colspan_), str(auxi_), str(der_pre_verb), str(der_lemma), der_sense, str(der_morph), der_cng, position_]

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


# In[11]:


def get_graphml(slp_sent, cng_list, graphml_file_name):
    new_path = os.path.join(sys.path[0], cng_list)
    dict_ = {}
#    try:
    dict_ = getdatafromsite(slp_sent, new_path)
#    except Exception:
#        print("Exception in scrapping. Possible Wrong input -> " + slp_sent)
#        return
    if (dict_ == {}):
        print("Empty dict from Heritage. Possible Wrong input or Timeout-> " + slp_sent)
        return
    graph_ = create_graph(dict_['t'])
    
#    write_graphml_path = os.path.join(graphml_folder, (str(sent_id) + ".graphml"))
    
    create_graphml(graph_, graphml_file_name)
    
def check_availability(file_name):
    try:
        if (Path(file_name).stat().st_size > 0):
            return True
        else:
            return False
    except Exception:
        return False

def open_file(file_name):
    if (not check_availability(file_name)):
        return []
    graph_input = open(file_name, mode='rb')
    graph = nx.read_graphml(graph_input)
    nodes = list(graph.nodes(data = True))
    return nodes

def check_graphml(graphml_file):
# To access the graph details
    graph_abs_file_path = os.path.join(os.getcwd(), graphml_file)
    nodes_ = open_file(graph_abs_file_path)
#    nodes_ = sorted(list(graph_.nodes(data = True)), key = lambda x : x[0])
    #print(len(nodes_))
    for item in nodes_:
        #print(item)
        first = item[1]
        word = first['word']
        lemma = first['lemma']
        pre_verb = first['pre_verb'] if 'pre_verb' in first.keys() else ""
        morph = first['morph']
        sense = first['sense']
        cng = first['cng']
        position = first['position']
        chunk_no = first['chunk_no']
        length_word = first['length_word']
        color_class = first['color_class']
        pre_verb = first['der_pre_verb'] if 'der_pre_verb' in first.keys() else ""
        pre_verb = first['der_lemma'] if 'der_lemma' in first.keys() else ""
        pre_verb = first['der_sense'] if 'der_sense' in first.keys() else ""
        pre_verb = first['der_morph'] if 'der_morph' in first.keys() else ""
        pre_verb = first['der_cng'] if 'der_cng' in first.keys() else ""
        char_pos = first['char_pos']
     
    return nodes_

# In[12]:

parser = argparse.ArgumentParser()
parser.add_argument('--sentence', type=str, required=True)
args = parser.parse_args()

#sentence = "etac cAnyac ca kOravya prasaNgi kawukodayam" #"rAjakfzRA janezwA ca kapikacCuphalopamA" #"yenedaM vyasanaM prAptA Bavanto dyUtakAritam"
sentence = args.sentence
graphml_file_name = "output.graphml"
cng_list = "cng_list_final"
slp_sent = iast2slp.convert(sentence)
get_graphml(slp_sent, cng_list, graphml_file_name)
nodes = check_graphml(graphml_file_name)
#print(nodes)


tpls = []
for node in nodes:
	k,d = node
	w = iast2slp.convert(d['word']) ## for web-based scrapper
	#w = dev2slp.convert(d['word']) ## for local scrapper
	s = d['char_pos']
	e = s + len(w) - 1
	## need to add chunk number also
	ch = d['chunk_no']
	tpls.append((w,s,e,ch))

tpls = sorted(list(set(tpls)))	

with open('created_graphml.txt','w') as g:
	for tpl in tpls:
		w,s,e,ch = tpl
		g.write(f"{w},{s},{e},{ch}\n") 
		



























