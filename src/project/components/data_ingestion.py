import os
import urllib.request as request
from src.project import logger
import zipfile
from src.project.entity.config_entity import (DataIngestionConfig)


## component-Data Ingestion

class DataIngestion:
    def __init__(self,config:DataIngestionConfig):
        self.config=config
   
    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)