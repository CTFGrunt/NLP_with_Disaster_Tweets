import joblib 
import numpy as np
import pandas as pd
from pathlib import Path

import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from wordcloud import STOPWORDS

nltk.download('stopwords')
nltk.download('wordnet')


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    
    def predict(self, text):
        new_text_tf= self.nlp_preprocessing(text)
        prediction = self.model.predict(new_text_tf)

        return prediction
    
    def nlp_preprocessing(text):
        # Remove HTTP tags
        text = text.map(lambda x : ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
        #Lower Case
        text = text.map(lambda x: x.lower())
        #Remove punctuations
        text = text.map(lambda x: re.sub(r'[^\w\s]', '', x))
        #Remove unicodes
        text = text.map(lambda x : re.sub(r'[^\x00-\x7F]+',' ', x))
        # Remove stopwords
        stop_words = STOPWORDS.words('english')
        text = text.map(lambda x : ' '.join([w for w in x.split() if w not in stop_words]))
        # Lemmatize the text
        lemmer = WordNetLemmatizer()
        text = text.map(lambda x : ' '.join([lemmer.lemmatize(w) for w in x.split() if w not in stop_words]))
        #Removing Stop words again after Lemmatize
        text = text.map(lambda x : ' '.join([w for w in x.split() if w not in stop_words]))
        # BOW-TF Embedding
        no_features = 800
        tf_vectorizer = CountVectorizer(min_df=.015, max_df=.8, max_features=no_features, ngram_range=(1, 3))

        tpl_tf = tf_vectorizer.fit_transform(text)
        return tpl_tf
