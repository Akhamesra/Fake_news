import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import json
import pickle
import math
from scrape import c

vectorizer = pickle.load(open("tfidf_vectorizer.pickle",'rb'))
model = pickle.load(open("PA.pickle",'rb'))


import re
import string
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text



link = input("")
test = c(link)
test_x = wordopt(test)
tfidf_x = vectorizer.transform([test_x])
pred = model.predict(tfidf_x)
result = math.ceil(model._predict_proba_lr(tfidf_x)[0][1]*100)
print(result,"% True.")
    
