import pandas as pd
import yaml
from NLP_Disaster_Tweets.entity.config_entity import DataValidationConfig
from NLP_Disaster_Tweets.utils.common import nullity_filter


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        # Load the YAML file
        with open('../params.yaml', 'r') as yaml_file:
            yaml_data = yaml.safe_load(yaml_file)

        # Access the variable
        self.mcr = yaml_data['MIN_COMPLETION_RATE']


    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            data=nullity_filter(data,'top',self.mcr)
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                        break
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
            
            return validation_status
        except Exception as e:
            raise e