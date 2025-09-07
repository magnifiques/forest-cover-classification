from src.project.constants import *
from src.project.utils.common import read_yaml, create_directories
from src.project import logger
from src.project.entity.config_entity import (DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainConfig)

class ConfigurationManager:
    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH,
                  params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        self.schema=read_yaml(schema_filepath)   

        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self)-> DataIngestionConfig:
        config=self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config=DataIngestionConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir

        )
        return data_ingestion_config

    def get_validation_config(self)-> DataValidationConfig:
        
        config=self.config.data_validation
        schema = self.schema.COLUMNS
        
        create_directories([config.root_dir])

        data_validation_config= DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema)
        
        return data_validation_config
    
    def get_transformation_config(self)-> DataTransformationConfig:
        config=self.config.data_transformation
        
        create_directories([config.root_dir])

        data_transformation_config= DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path)
        
        return data_transformation_config

    def get_model_train_config(self, model_type: str)-> ModelTrainConfig:
        
        config=self.config.model_train
        target = self.schema.TARGET_COLUMN
        
        # pick correct params
        if model_type == "RandomForest":
            params = self.params.RandomForest
        elif model_type == "XGBoost":
            params = self.params.XGBoost
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")
            
        create_directories([config.root_dir])
        
        # Convert params to kwargs, handling missing attributes
        param_kwargs = {}
        param_dict = dict(params) if hasattr(params, '__dict__') else params
        
        # Add parameters that exist
        for key, value in param_dict.items():
            param_kwargs[key] = value

        model_train_config = ModelTrainConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            validation_data_path=config.validation_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            target_column=target.name,
            model_type=model_type,
            **param_kwargs
        )
        return model_train_config
