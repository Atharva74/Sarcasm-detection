from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, GRU, LSTM, Bidirectional
from keras.layers import Embedding
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


data1 = pd.read_json('Sarcasm.json',lines=True)
data2 = pd.read_json('Sarcasm_v2.json',lines=True)
# temp = pd.DataFrame(data1,columns={"headline","is_sarcastic"})

data1.rename(columns={'headline': 'tweets','is_sarcastic':'class'}, inplace=True)
data2.rename(columns={'headline': 'tweets','is_sarcastic':'class'}, inplace=True)
data1.drop('article_link', axis=1, inplace=True)
data2.drop('article_link', axis=1, inplace=True)
finalData = pd.concat([data1,data2])
finalData

def clean_text(text):
    text = text.lower()
    
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji.sub(r'', text)
    text = text.lower()

    text = re.sub(r"#\w+", "", text)
    text = re.sub(r":-\w+", "", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text

import nltk

import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def CleanTokenize(df):
    tweets = list()
    lines = df["tweets"].values.tolist()

    for line in lines:
        line = clean_text(line)
        # tokenize the text
        tokens = word_tokenize(line)
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        # remove stop words
        words = [w for w in words if not w in stop_words]
        tweets.append(words)
    return tweets

tweet = CleanTokenize(finalData)

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(tweet)



model = load_model('newModel_v2.h5',compile=False)


def index(request):
   return render(request, 'index.html')

def about(request):
    return HttpResponse("<b>General Kenobi</b> <br> <a href = '/'> back </a>")

def analyze(request):
    data=json.loads(request.body)
    djtext = data.get('my_key')
    print(type(djtext),djtext)
    x_final = pd.DataFrame({"tweets":[djtext]})
    test_lines = CleanTokenize(x_final)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=25, padding='post')
    pred = model.predict(test_review_pad)
    
    pred = float(pred[0][0])*100
    print(test_lines,test_sequences,test_review_pad)
    print(np.round(float(pred),4))
    
    if(pred>=50):                     
        analyzed = "Sarcasm"
    else:
        analyzed ="Not Sarcasm"

    params = {'analyzed_text' : analyzed,'Accuracy':str(round(float(pred),3))}

    return JsonResponse(params)

from django.http import JsonResponse
import json
def my_view(request):
    data = json.loads(request.body)
    my_value = data.get('my_key')
    
    dta = {"data":str(my_value) }
    print(type(my_value),my_value)
    return JsonResponse(dta)





