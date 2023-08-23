import os
from NLP_Disaster_Tweets import logger
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

from NLP_Disaster_Tweets.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        with open('../params.yaml', 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        # Access the variable
        self.test_size = yaml_data['TEST_SIZE']
        self.train_size = yaml_data['TRAIN_SIZE']

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

    # I am only adding train_test_spliting cz this data is already cleaned up


    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data,test_size=0.25, train_size=0.75)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

        logger.info("Splited data into training and test sets")
        logger.info(train.shape)
        logger.info(test.shape)

        print(train.shape)
        print(test.shape)
        