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
            model_types = ["RandomForest", "XGBoost"]  # list of models to train

            results = {}
            
            for model_type in model_types:
                print(f"\n>>> Training {model_type} <<<")
                
                # Update config with current model type
                model_train_config = config.get_model_train_config(model_type)
                model_train_config.model_type = model_type
                
                model_train = ModelTrain(config=model_train_config)
                result = model_train.train()  # train and save model
                
                results[model_type] = result
            # Print summary
            print("\n" + "="*50)
            print("TRAINING SUMMARY")
            print("="*50)
            for model_type, result in results.items():
                print(f"{model_type}:")
                print(f"  Validation Accuracy: {result['validation_accuracy']:.4f}")
                print(f"  Validation F1-Score: {result['validation_f1']:.4f}")
                print(f"  Best Parameters: {result['best_params']}")
                print()
        except Exception as e:
            raise e