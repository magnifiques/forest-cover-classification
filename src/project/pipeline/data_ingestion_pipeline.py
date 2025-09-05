from src.project.config.configuration import ConfigurationManager
from src.project.components.data_ingestion import DataIngestion
from src.project import logger


STAGE_NAME = 'Data Ingestion Stage'

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_data_ingestion(self):
        config = ConfigurationManager()
        
        data_ingestion_config=config.get_data_ingestion_config()
        
        data_ingestion=DataIngestion(config=data_ingestion_config)
        
        data_ingestion.extract_zip_file()
        