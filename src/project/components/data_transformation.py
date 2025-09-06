import os
from sklearn.model_selection import train_test_split
import pandas as pd
from src.project.entity.config_entity import DataTransformationConfig
from src.project import logger

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    ## Note: You can add different data transformation techniques such as Scaler, PCA and all
    #You can perform all kinds of EDA in ML cycle here before passing this data to the model

  
    def train_test_splitting(self):
        data = pd.read_csv(self.config.data_path)

        # Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data, test_size=0.2, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "dataset_train.csv"),index = False)
        
        validation, test = train_test_split(test, test_size=0.5, random_state=42)
        
        validation.to_csv(os.path.join(self.config.root_dir, "dataset_validation.csv"), index=False)
        
        test.to_csv(os.path.join(self.config.root_dir, "dataset_test.csv"),index = False)

        logger.info("Splitted data into training, validation, and test sets")
        
        logger.info(train.shape)
        logger.info(validation.shape)
        logger.info(test.shape)

        print(train.shape)
        print(validation.shape)
        print(test.shape)
        