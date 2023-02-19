import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
# For changing transliteration scheme
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
# To run bash commands
from subprocess import run, call

from flask_cors import CORS, cross_origin

from flaskext.markdown import Markdown
import Sanskrit
import json
from flask import Flask, url_for, render_template, request
import spacy
from spacy import displacy

from stanza.utils.conll import CoNLL
nlp = spacy.load('en_core_web_sm')

HTML_WRAPPER = """<div style=" border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; font-size: 22px">{}</div>"""


app = Flask(__name__)
Markdown(app)
CORS(app)

output_type = ""

def ConvertInSLP(input_scheme, words):
    if input_scheme=="WX":
        for i in range(len(words)):
            words[i] = transliterate(words[i], sanscript.WX, sanscript.SLP1)
    elif input_scheme=="Devanagari":
        for i in range(len(words)):
            words[i] = transliterate(words[i], sanscript.DEVANAGARI, sanscript.SLP1)
    elif input_scheme=="IAST":
        for i in range(len(words)):
            words[i] = transliterate(words[i], sanscript.IAST, sanscript.SLP1)
    elif input_scheme=="Velthuis":
        for i in range(len(words)):
            words[i] = transliterate(words[i], sanscript.VELTHUIS, sanscript.SLP1)
    return words

def ConvertInOutput(out_scheme,words):
    if out_scheme=="WX":
        for i in range(len(words)):
            words[i] = transliterate(words[i], sanscript.SLP1, sanscript.WX)
    elif out_scheme=="Devanagari":
        for i in range(len(words)):
            words[i] = transliterate(words[i], sanscript.SLP1, sanscript.DEVANAGARI)
    elif out_scheme=="IAST":
        for i in range(len(words)):
            words[i] = transliterate(words[i], sanscript.SLP1, sanscript.IAST)
    elif out_scheme=="Velthuis":
        for i in range(len(words)):
            words[i] = transliterate(words[i], sanscript.SLP1, sanscript.VELTHUIS)
    return words

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/', methods=['POST', 'GET'])
@cross_origin(headers=['Content-Type', 'Authorization','Access-Control-Allow-Origin'])
def predict():
    params = request.args
    print(params, " are params")
    if len(params):
        inptype , outtype , sen = params['in'], params['out'], params['sen']
    text = [x for x in request.form.values()]
    # print(text)
    if len(text)==0:
        text = [inptype.strip(), outtype.strip(), sen.strip()]
    output = text[2].strip()


    global output_type
    output_type = text[1]
    # print(output_type)
    #input scheme
    input_scheme = text[0]
    output_scheme = text[1]
    len_words = len(output)
    words = output.split(" ")
    num_of_words = len(words)
    index = {}
    tem = 0
    words = ConvertInSLP(input_scheme,words)
    line = " ".join(words)
    # print(line)
    w = open('/home/jivneshs/SanShala-Models/trankit/input_data/input.txt','w')
    w.write(line)
    w.close()
    import subprocess
    p = subprocess.Popen(("/bin/sh", "./go_to_dep.sh"))
    p.wait()

    import shutil
    original = '/home/jivneshs/SanShala-Models/trankit/input_data/data.json'
    target = '/home/jivneshs/SanShala-Web/sanskritshala.github.io/src/inputfile/data.json'
    shutil.copyfile(original, target)

    # edited code
    raw_text = output
    docx = nlp(raw_text)
    # print(str(docx))
    # print(type(docx))
    html = displacy.render(docx, style="dep")
    html = Sanskrit.get_sans_html(filename='/home/jivneshs/SanShala-Models/trankit/input_data/output.txt')
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    # edited code


    
    # f = open("/home/jivneshs/SanShala-Models/LemmaTag/data/test.txt", "w")
    # st = ""
    # for i in range(num_of_words):
    #     st += words[i]+'\t'+'_'+'\t'+'_\n'
    # f.write(st)
    # f.close()

    # f = open("/home/jivneshs/SanShala-Models/LemmaTag/Morph_Scrapper/1.txt", "w")
    # st = ""
    # st = ' '.join(words)+'\n'
    # f.write(st)
    # f.close()

    # rc = run('cd /home/jivneshs/SanShala-Models/LemmaTag/Morph_Scrapper/; python final_scrap_.py; cd /home/jivneshs/SanShala-Web/templates/POS-tagging',shell=True)

    # data = pd.read_csv('/home/jivneshs/SanShala-Models/LemmaTag/Morph_Scrapper/details/0.csv',sep=',')
    # line = st
    # text = transliterate(line, sanscript.SLP1,sanscript.IAST)
    # candidate_space = []
    # for word in text.split():
    #     candidate_space.append(list(set(list(data[data['word']==word]['morph']))))

    # tags word wise
    # tags_word = {0: ["m. sg. g.",  "n. sg. g."], 1: ["pfp. [1]", "f. sg. nom.", "ca. pfp. [1]"], 2: ["f. sg. nom."], 3: ["n. sg. g.", "m. sg. g."], 4: [
    #     "m. sg. nom."], 5: ["m. sg. nom.", "n. sg. g.", "n. sg. abl."], 6: ["conj."], 7: ["m. sg. i.", "n. sg. i."], 8: ["adv.", "prep."], 9: ["n. sg. acc.", "n. sg. nom."], 10: ["tasil"]}
    # tags_word = candidate_space

    # colors that we use
    color = ["#B4E8FC", "#FFA9B8", "#D1CAFF", "#9EFFD6", "#F74C4A", "#B4E8FC", "#FFA9B8", "#D1CAFF", "#9EFFD6", "#F74C4A", "#B4E8FC", "#FFA9B8", "#D1CAFF", "#9EFFD6", "#F74C4A", "#B4E8FC", "#FFA9B8", "#D1CAFF", "#9EFFD6", "#F74C4A",
             "#B4E8FC", "#FFA9B8", "#D1CAFF", "#9EFFD6", "#F74C4A", "#B4E8FC", "#FFA9B8", "#D1CAFF", "#9EFFD6", "#F74C4A", "#B4E8FC", "#FFA9B8", "#D1CAFF", "#9EFFD6", "#F74C4A", "#B4E8FC", "#FFA9B8", "#D1CAFF", "#9EFFD6", "#F74C4A"]

    # # recommendation by first model
    # other_tags_recommendation = {}
    # for i in range(len(tags_word)):
    #     temp = []
    #     for j,tag in enumerate(tags_word[i]):
    #         temp.append([j,tag])
    #     other_tags_recommendation[i] = temp

    # # other_tags_recommendation = {0: [[0, "m. sg. g."], [1, "n. sg. g."]], 1: [[0, "pfp. [1]"], [1, "f. sg. nom."], [2, "ca. pfp. [1]"]], 2: [[0, "f. sg. nom."]], 3: [[0, "n. sg. g."], [1, "m. sg. g."]], 4: [
    # #     [0, "m. sg. nom."]], 5: [[0, "m. sg. nom."], [1, "n. sg. g."], [2, "n. sg. abl."]], 6: [[0, "conj."]], 7: [[0, "m. sg. i."], [1, "n. sg. i."]], 8: [[0, "adv."], [1, "prep."]], 9: [[0, "n. sg. acc."], [1, "n. sg. nom."]], 10: [[0, "tasil"]]}

    # # Running second model for single prediction
    # # rc = run('cd /home/jivneshs/SanShala-Models/LemmaTag; python lemmatag.py; cd /home/jivneshs/SanShala-Web/templates/POS-tagging',shell=True)
    # data = pd.read_csv('/home/jivneshs/SanShala-Models/LemmaTag/logs/prediction_morph_tag/taglem_test_ep0.txt',sep='\t',header=None)
    # tag_type = list(data[2])
    # print(tag_type)

    # total tags that we have
    total_tags = ["adv.", "prep.", "tasil", "m. sg. i.", "n. sg. i.", "conj.", "m. sg. nom.",
                  "n. sg. g.", "n. sg. abl.", "m. sg. g.", "f. sg. nom.", "pfp. [1]", "ca. pfp. [1]"]
    
    # flag = []
    # flagtemp = 1
    # for i in range(len(tags_word)):
    #     flagtemp = 1
    #     for j in range(len(tags_word[i])):
    #         if tags_word[i][j]==tag_type[i]:
    #             flagtemp = 0
    #             break
    #     flag.append(flagtemp)

    # words = ConvertInOutput(output_scheme,words)
    # for i, word in enumerate(words):
    #     index[i] = [word, tag_type[i], color[i], other_tags_recommendation[i], total_tags, flag[i]]
    # print(index)
    if len_words == 0:
        index["empty"] = "You have entered empty sentence. Please enter some text in your sentence."

    return render_template('result.html', rawtext=raw_text, result=result)

@app.route("/gotodependency/", methods=['POST', 'GET'])
# @cross_origin()
@cross_origin(headers=['Content-Type', 'Authorization','Access-Control-Allow-Origin'])
def gotodependency():
    #Moving forward code
    if request.method == "POST":
        data = request.get_json()
        print(data)
        # # print("out type is : ", output_type)
        # if output_type!="Devanagari":
        #     final_words = ConvertInOutput(output_type,data[0]['Words'])
        # else:
        #     final_words = data[0]['Words']
        # line = " ".join(final_words)
        # print(line)
        # w = open('/home/jivneshs/SanShala-Models/trankit/input_data/input.txt','w')
        # w.write(line)
        # w.close()
        # import subprocess
        # p = subprocess.Popen(("/bin/sh", "./go_to_dep.sh"))
        # p.wait()

        # import shutil
        # original = '/home/jivneshs/SanShala-Models/trankit/input_data/data.json'
        # target = '/home/jivneshs/SanShala-Web/sanskritshala.github.io/src/inputfile/data.json'
        # shutil.copyfile(original, target)


    url = "http://localhost:3000/dp"
    return jsonify(url)

@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0',port = 4040)