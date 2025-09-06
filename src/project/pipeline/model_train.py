from src.project.config.configuration import ConfigurationManager
from pathlib import Path
from src.project.components.data_transformation import DataTransformation
from src.project.components.model_train import ModelTrain

STAGE_NAME = 'Model Training Stage'

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_model_training(self):
        
        try:
            config = ConfigurationManager()
            model_train_config = config.get_model_train_config()
            model_train = ModelTrain(config=model_train_config)
            model_train.train()
        except Exception as e:
            raise e