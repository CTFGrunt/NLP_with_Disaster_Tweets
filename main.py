from NLP_Disaster_Tweets.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from NLP_Disaster_Tweets import logger

STAGE_NAME = "Data Ingestion stage"


     
try:
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<< \n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e