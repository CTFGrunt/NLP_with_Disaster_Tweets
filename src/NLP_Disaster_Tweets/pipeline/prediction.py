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

from NLP_Disaster_Tweets.config.configuration import ConfigurationManager

nltk.download('stopwords')
nltk.download('wordnet')


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    
    def predict(self,text):
        new_text_tf= self.nlp_preprocessing(text)
        prediction = self.model.predict(new_text_tf)

        return prediction

    def nlp_preprocessing(self,text):
        # Remove HTTP tags
        text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        #Lower Case
        text = text.lower()
        #Remove punctuations
        text = re.sub(r'[^\w\s]', '', text)
        #Remove unicodes
        text = re.sub(r'[^\x00-\x7F]+',' ', text)
        # Remove stopwords
        stop_words = stopwords.words('english')
        text = ' '.join([w for w in text.split() if w not in stop_words])
        # Lemmatize the text
        lemmer = WordNetLemmatizer()
        text = ' '.join([lemmer.lemmatize(w) for w in text.split() if w not in stop_words])
        #Removing Stop words again after Lemmatize
        text = ' '.join([w for w in text.split() if w not in stop_words])
        # BOW-TF Embedding
        no_features = 800
        vocab = self.load_vocab_from_training()
        tf_vectorizer = CountVectorizer(vocabulary=vocab)

        tpl_tf = tf_vectorizer.fit_transform([text])
        return tpl_tf

    def load_vocab_from_training(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        train_data = pd.read_csv(model_trainer_config.train_data_path)
        return train_data.columns.drop('target')