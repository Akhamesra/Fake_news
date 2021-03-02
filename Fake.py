import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle


frame = pd.read_csv("train.csv")
frame = frame.fillna(' ')
frame['total'] = frame['title']+' '+frame['author']+frame['text']
frame = frame.set_index("id")
frame=frame.drop(["title", 'author'],axis=1)
frame1=frame.drop(['total'],axis=1)

y = frame1.label
frame1.drop("label", axis=1)

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

frame1["text"] = frame["text"].apply(wordopt)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(frame['text'], y, test_size=0.000001, random_state=43)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7) 
tfidf_train = tfidf_vectorizer.fit_transform(X_train)

from sklearn.linear_model import PassiveAggressiveClassifier
PA = PassiveAggressiveClassifier(random_state=0)
PA.fit(tfidf_train, y_train)

pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pickle", "wb"))
pickle.dump(PA, open("PA.pickle", 'wb'))