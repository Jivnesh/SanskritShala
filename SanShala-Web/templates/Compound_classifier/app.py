import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
# import pickle
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# For loading model
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import pdb
# For changing transliteration scheme
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model =TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=4)

# Load model
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

model_save_path='NLP_saved_model/NLP'

model = TFBertForSequenceClassification.from_pretrained('bert-base-multilingual-cased',num_labels=4)
model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
model.load_weights(model_save_path)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = [x for x in request.form.values()]
    # print(text)

    output = text[2]
    output_trans = text[1]
    trans_scheme = text[0]
    x = {2:'Tatpurush',3:'Dvand',1:"Bahubrihi",0:"Avyayibhav"}
    d = {"WX":"wx","IAST":"iast","SLP1":"slp1","Devanagari":"devanagari","Velthuis":"velthuis"}
    # Prediction
    def prediction(word1,word2, trans_scheme):
        if trans_scheme=="WX":
            text1 = transliterate(word1, sanscript.WX, sanscript.DEVANAGARI)
            text2 = transliterate(word2, sanscript.WX, sanscript.DEVANAGARI)
        elif trans_scheme=="SLP1":
            text1 = transliterate(word1, sanscript.SLP1, sanscript.DEVANAGARI)
            text2 = transliterate(word2, sanscript.SLP1, sanscript.DEVANAGARI)
        elif trans_scheme=="IAST":
            text1 = transliterate(word1, sanscript.IAST, sanscript.DEVANAGARI)
            text2 = transliterate(word2, sanscript.IAST, sanscript.DEVANAGARI)
        elif trans_scheme=="Velthuis":
            text1 = transliterate(word1, sanscript.VELTHUIS, sanscript.DEVANAGARI)
            text2 = transliterate(word2, sanscript.VELTHUIS, sanscript.DEVANAGARI)
        elif trans_scheme=="Devanagari":
            text1 = word1
            text2 = word2
        text=text1+" "+text2

        input_ids_test=[]
        attention_masks_test=[]
        bert_inp=tokenizer.encode_plus(text,add_special_tokens = True,max_length =64,pad_to_max_length = True,return_attention_mask = True)
        input_ids_test.append(bert_inp['input_ids'])
        attention_masks_test.append(bert_inp['attention_mask'])

        input_ids_test=np.asarray(input_ids_test)
        attention_masks_test=np.array(attention_masks_test)

        preds = model.predict([input_ids_test,attention_masks_test])
        pred= np.array(preds[0])
        y_pred=pred
        lis=[]
        for i in range(0,1):
            if(y_pred[i][0]> y_pred[i][1] and y_pred[i][0] > y_pred[i][2]  and y_pred[i][0] > y_pred[i][3]):
                lis.append(x[0])
            elif(y_pred[i][1]> y_pred[i][0] and y_pred[i][1] > y_pred[i][2]  and y_pred[i][1] > y_pred[i][3]):
                lis.append(x[1])
            elif(y_pred[i][2]> y_pred[i][1] and y_pred[i][2] > y_pred[i][0]  and y_pred[i][2] > y_pred[i][3]):
                lis.append(x[2])
            else:
                lis.append(x[3])
        import math
        n1 = math.exp(y_pred[0][0])
        n2 = math.exp(y_pred[0][1])
        n3 = math.exp(y_pred[0][2])
        n4 = math.exp(y_pred[0][3])
        su = n1+n2+n3+n4
        prob = []
        HundredMinusProb = []
        prob.append(round(n1*100/su,2))
        HundredMinusProb.append(100-prob[-1])
        prob.append(round(n2*100/su,2))
        HundredMinusProb.append(100-prob[-1])
        prob.append(round(n3*100/su,2))
        HundredMinusProb.append(100-prob[-1])
        prob.append(round(n4*100/su,2))
        HundredMinusProb.append(100-prob[-1])
        return (lis[0],prob,HundredMinusProb)
    
    def change(s):
        text = transliterate(s, d[trans_scheme], d[output_trans])
        return text
    
    words = output.split(" ")
    index = {}
    tem = 0
    # probability = [.1,.1,.2,.6]
    
    # plt.bar(x,probability)
    # plt.xlabel('Compound type')
    # plt.ylabel('Probability')
    # plt.title('Graphical analysis')
    # plt.savefig('static/compo.png')
    filename = 'compo.png'
    compound_type = "Avyayibhav"
    color= ["#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A","#B4E8FC","#FFA9B8","#D1CAFF","#9EFFD6","#F74C4A"]
    for i, word in enumerate(words):
        if '-' in word:
            lis = word.split('-')
            compound_type, prob, HundredMinus = prediction(lis[0],lis[1],trans_scheme)
            if output_trans=="Devnagari":
                index[i] = [word, compound_type, filename, 1, color[i], prob, HundredMinus]
            else:
                index[i] = [change(lis[0])+'-'+change(lis[1]), compound_type, filename, 1, color[i], prob, HundredMinus]
            tem += 1
        else:
            if output_trans=="Devnagari":
                index[i] = [word, 'no-compound', filename, 0, color[i]]
            else:
                index[i] = [change(word), 'no-compound', filename, 0, color[i]]
    print(index)
    if tem == 0:
        index["empty"] = "No compound word found. Please enter the compund word with '-' in your text."
    print(output)
    
    return render_template('index.html', prediction_text = output, index = index)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename= filename), code=301)

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    print('*****'*10)
    output = prediction[0]
    print(data)
    print(output)
    pdb.set_trace()
    return jsonify(output)

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0',port = 5000)