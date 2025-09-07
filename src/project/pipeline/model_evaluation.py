from src.project.config.configuration import ConfigurationManager
from src.project.components.model_evaluation import ModelEvaluation

STAGE_NAME = 'Model Evaluation Stage'

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def initiate_model_evaluation(self):
        
        try:
            config = ConfigurationManager()
            model_types = ["RandomForest", "XGBoost"]  # list of models to train

            for model_type in model_types:
                print(f"\n>>> Evaluation {model_type} <<<")
                
                # Update config with current model type
                config = ConfigurationManager()
                model_evaluation_config = config.get_model_evaluation_config()
                model_evaluation = ModelEvaluation(config=model_evaluation_config)
                model_evaluation.log_into_mlflow(model_type)
            
        except Exception as e:
            raise e