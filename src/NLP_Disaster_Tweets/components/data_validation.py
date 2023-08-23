import pandas as pd
import yaml
from NLP_Disaster_Tweets.entity.config_entity import DataValidationConfig
from NLP_Disaster_Tweets.utils.common import nullity_filter


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        self.mcr = self.config.MIN_COMPLETION_RATE


    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data = pd.read_csv(self.config.unzip_data_dir)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()

            data=nullity_filter(data,'top',self.mcr)
            for col in all_schema:
                if col not in all_cols:
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