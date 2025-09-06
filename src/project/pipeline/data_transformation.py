from src.project.config.configuration import ConfigurationManager
from pathlib import Path
from src.project.components.data_transformation import DataTransformation


STAGE_NAME = 'Data Transformation Stage'

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_data_transformation(self):
        
        try:
            with open(Path('artifacts/data_validation/status.txt'), 'r') as f:
                status = f.read().split()[-1]
                
            if status == 'True':
                config = ConfigurationManager()
                
                data_transformation_config=config.get_transformation_config()
                
                data_transformation=DataTransformation(config=data_transformation_config)
                
                data_transformation.train_test_splitting()
            else:
                raise Exception("Your dataset hasn't been successfully validated.") 
        except Exception as e:
            raise e