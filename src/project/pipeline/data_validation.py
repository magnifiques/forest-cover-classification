from src.project.config.configuration import ConfigurationManager

from src.project.components.data_validation import DataValidation


STAGE_NAME = 'Data Validation Stage'

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def initiate_data_validation(self):
        config = ConfigurationManager()
        
        data_validation_config=config.get_validation_config()
        
        data_validation=DataValidation(config=data_validation_config)
        
        data_validation.validate_all_columns()