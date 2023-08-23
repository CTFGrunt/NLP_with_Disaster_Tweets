import os
from NLP_Disaster_Tweets import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml
from NLP_Disaster_Tweets.constants import *
from NLP_Disaster_Tweets.entity.config_entity import DataTransformationConfig
from NLP_Disaster_Tweets.utils.common import read_yaml


from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')




class DataTransformation:
    def __init__(self, config: DataTransformationConfig,
                params_filepath = PARAMS_FILE_PATH,
                schema_filepath = SCHEMA_FILE_PATH):
        self.config = config
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        self.data=pd.read_csv(self.config.data_path)
        self.tf_data= self.data

    
    ## Note: We are using Bag of Words method for embedding

    def nlp_preprocessing(self):
        # Remove HTTP tags
        self.data['text'] = self.data['text'].map(lambda x : ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))
        #Lower Case
        self.data['text'] = self.data['text'].map(lambda x: x.lower())
        #Remove punctuations
        self.data['text'] = self.data['text'].map(lambda x: re.sub(r'[^\w\s]', '', x))
        #Remove unicodes
        self.data['text'] = self.data['text'].map(lambda x : re.sub(r'[^\x00-\x7F]+',' ', x))
        # Remove stopwords
        stop_words = stopwords.words('english')
        self.data['text'] = self.data['text'].map(lambda x : ' '.join([w for w in x.split() if w not in stop_words]))
        # Lemmatize the text
        lemmer = WordNetLemmatizer()
        self.data['text'] = self.data['text'].map(lambda x : ' '.join([lemmer.lemmatize(w) for w in x.split() if w not in stop_words]))
        #Removing Stop words again after Lemmatize
        self.data['text'] = self.data['text'].map(lambda x : ' '.join([w for w in x.split() if w not in stop_words]))
        # BOW-TF Embedding
        no_features = 800
        tf_vectorizer = CountVectorizer(min_df=.015, max_df=.8, max_features=no_features, ngram_range=(1, 3))

        tpl_tf = tf_vectorizer.fit_transform(self.data['text'])
        self.tf_data = pd.DataFrame(tpl_tf.toarray(), columns=tf_vectorizer.get_feature_names_out())
        self.tf_data = pd.concat([self.data, self.tf_data], axis = 1)
        self.tf_data.drop(columns=['text', 'keyword','location'], inplace = True)

    def train_test_spliting(self):
        self.nlp_preprocessing()
        train, test = train_test_split(self.tf_data,test_size= self.params.TEST_SIZE, train_size= self.params.TRAIN_SIZE)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        