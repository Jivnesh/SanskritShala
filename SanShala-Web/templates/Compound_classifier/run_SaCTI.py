import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
# import pickle
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


# For changing transliteration scheme
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

app = Flask(__name__)
# app.config['SERVER_NAME'] = "http://www.cnerg.iitkgp.in.ac/tramp"
import subprocess
import pandas as pd
import os
# @app.route('/tramp')
# def index():
#     return request.base_url
# app.config['APPLICATION_ROOT'] = '/tramp'
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict/',methods=['POST'])
def predict():
    text = [x for x in request.form.values()]
    # print(text)
    output = text[2]
    output_trans = text[1]
    trans_scheme = text[0]
    x = {2:'Tatpurush',3:'Dvand',1:"Bahubrihi",0:"Avyayibhav"}
    d = {"WX":"wx","IAST":"iast","SLP1":"slp1","Devanagari":"devanagari","Velthuis":"velthuis"}
    # Prediction
    def prediction(inp_str, trans_scheme):
        if trans_scheme=="WX":
            inp_str = transliterate(output, sanscript.WX, sanscript.DEVANAGARI)
    
        elif trans_scheme=="SLP1":
            inp_str = transliterate(output, sanscript.SLP1, sanscript.DEVANAGARI)
            
        elif trans_scheme=="IAST":
            inp_str = transliterate(output, sanscript.IAST, sanscript.DEVANAGARI)
           
        elif trans_scheme=="Velthuis":
            inp_str = transliterate(output, sanscript.VELTHUIS, sanscript.DEVANAGARI)
            
        elif trans_scheme=="Devanagari":
            inp_str = output
        words = inp_str.split()
        compound_list = []
        for word in words:
            if "-" in word:
                compound_list.append(words+[word])
        f = open('/home/jivneshs/SanShala-Models/SaCTI/data/coling/coarse/test_new.conll','w')
        for sent in compound_list:
            for i, w in enumerate(sent):
                temp = ['_']*12
                temp[0] = str(i+1)
                temp[1] = w
                f.write('\t'.join(temp)+'\n')
            f.write('\n')
        f.close()

        process = subprocess.Popen(["/bin/sh", "run_SaCTI.sh"])
        process.wait()
        # subprocess.run(["bash","run_SaCTI.sh"])

        file_path = "/home/jivneshs/SanShala-Models/SaCTI/models1/xlm-roberta-base/customized-mwt-ner/preds/tagger.testfaL.conllu.epoch--1"
        f = open(file_path)
        lines = f.readlines()
        cc = {}
        for j in range(len(lines)):
            if lines[j] == '\n':
                cc[lines[j-1].split('\t')[1]] = lines[j-1].split('\t')[7]
        print(cc)

        return cc, inp_str
    
    def change(s):
        text = transliterate(s, d[trans_scheme], d[output_trans])
        return text
    compound_ls , inp_str = prediction(output, trans_scheme)
    words = inp_str.split()
    index = {}
    tem = 0
    filename = 'compo.png'
    compound_type = "Avyayibhav"
    color= ["#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A"]
    
    for i, word in enumerate(words):
        prob = 0
        HundredMinus=100
        if '-' in word:
            compound_type = compound_ls[word]
            if compound_type == "Dvandva":
                prob = [0, 0, 0, 100]
                HundredMinus =  [100,100,100,0]
            elif compound_type == "Avyayibhava":
                prob = [100, 0, 0, 0]
                HundredMinus =  [0,100,100,100]
            elif compound_type == "Tatpurusha":
                prob = [0, 0, 100, 0]
                HundredMinus =  [100,100,0,100]
            else:
                prob = [0, 100, 0, 0]
                HundredMinus =  [100,0,100,100]
            if output_trans=="Devnagari":
                index[i] = [word, compound_type, filename, 1, color[i], prob, HundredMinus]
            else:
                index[i] = [change(word), compound_type, filename, 1, color[i], prob, HundredMinus]
            tem += 1
        else:
            if output_trans=="Devnagari":
                index[i] = [word, 'no-compound', filename, 0, color[i]]
            else:
                index[i] = [change(word), 'no-compound', filename, 0, color[i]]
    print(index)
    print(inp_str)
    print(len(index),len(inp_str))
    if tem == 0:
        index["empty"] = "No compound word found. Please enter the compund word with '-' in your text."
    
    
    return render_template('index.html', prediction_text = inp_str, index = index)

# @app.route('/display/<filename>')
# def display_image(filename):
#     #print('display_image filename: ' + filename)
#     return redirect(url_for('static', filename= filename), code=301)

# @app.route('/results',methods=['POST'])
# def results():

#     data = request.get_json(force=True)
#     # prediction = model.predict([np.array(list(data.values()))])

#     # output = prediction
#     # return jsonify(output)
#     return 0

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0',port = 5000)