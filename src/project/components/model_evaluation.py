import os
from dotenv import load_dotenv
import tempfile
import joblib
import json
from src.project.utils.common import  save_json
import sklearn
from src.project.entity.config_entity import ModelEvaluationConfig
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from urllib.parse import urlparse
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow.xgboost
from pathlib import Path


load_dotenv(override=True)

mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
dagshub_username = os.getenv('DAGSHUB_USERNAME')
dagshub_token = os.getenv('DVC_SECRET_ACCESS_KEY')

if mlflow_tracking_uri is None:
    raise ValueError("mlflow_tracking_uri is not set in your environment!")

if dagshub_token is None:
    raise ValueError("DAGSHUB_TOKEN is not set in your environment!")

if dagshub_username is None:
    raise ValueError("DAGSHUB USERNAME is not set in your environment!")

os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

os.environ['MLFLOW_TRACKING_URI'] = mlflow_tracking_uri

os.environ['MLFLOW_TRACKING_USERNAME'] = dagshub_username


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self, actual, pred):
        
        """Calculate classification metrics for Forest Cover Type dataset"""
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        
        f1 = f1_score(actual, pred)
              
        return accuracy, precision, recall, f1

    def log_into_mlflow(self, model_name):
        """Log all models into MLflow"""
         
        mlflow.set_registry_uri(self.config.mlflow_tracking_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
    
        # Find all model files in the model directory
        model_files = []
        model_dir = Path(self.config.model_path)
    
        # Look for different model files
        all_params_dict = {k: v for d in self.config.all_params for k, v in d.items()}
        model_params = all_params_dict[f'{model_name}Best'] 
        for file_path in model_dir.glob("*.joblib"):
            if model_name in file_path.name:
                model_files.append((model_name, str(file_path)))
                break  # Only need the first match
        print(model_files, model_params)
      
        # print('I am here:', model_dir)
      
        #If no specific model files found, use the default path
        if not model_files:
            model_files = [("DefaultModel", self.config.model_path)]
    
        all_results = {}
    
        # Loop through each model without nesting runs
        for model_name, model_path in model_files:
            print(f"\n>>> Evaluating {model_name} <<<")
        
            # Start a separate run for each model (not nested)
            with mlflow.start_run(run_name=f"{model_name}_evaluation"):
                try:
                    # Load test data and model
                    test_data = pd.read_csv(self.config.test_data_path)
                    model = joblib.load(model_path)

                    test_x = test_data.drop([self.config.target_column], axis=1)
                    test_y = test_data[self.config.target_column]

                    # Make predictions
                    predictions = model.predict(test_x)
                
                    signature = infer_signature(test_x, test_y)

                    # Calculate metrics
                    accuracy, precision, recall, f1 = self.eval_metrics(test_y, predictions)
                
                    # Store results
                    metrics = {
                        "model_name": model_name,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                    }
                
                    all_results[model_name] = metrics
                
                    # Log parameters if available
                    if model_params != '':
                        mlflow.log_params(model_params)
                
                    # Log metrics to MLflow
                    mlflow.log_metric("accuracy", accuracy) # type: ignore
                    mlflow.log_metric("precision", precision) # type: ignore
                    mlflow.log_metric("recall", recall) # type: ignore
                    mlflow.log_metric("f1_score", f1)  #type: ignore
                
                    # Log classification report as artifact
                    class_report = classification_report(test_y, predictions, output_dict=True)
                    class_report_path = f"{model_name}_classification_report.json"
                    with open(class_report_path, 'w') as f:
                        json.dump(class_report, f, indent=2)
                    mlflow.log_artifact(class_report_path)
                    os.remove(class_report_path)  # Clean up
                    
                    # Log confusion Matrix as artifact
                    conf_matrix = confusion_matrix(test_y, predictions)
                    conf_matrix_df = pd.DataFrame(conf_matrix)
                    conf_matrix_path = f"{model_name}_conf_matrix.csv"
                    conf_matrix_df.to_csv(conf_matrix_path, index=True, header=True)
                    mlflow.log_artifact(conf_matrix_path, "metrics")
                    os.remove(conf_matrix_path)
                
                    # Log model based on tracking store type
                    if tracking_url_type_store != 'file':
                        # For remote tracking (DagsHub), use artifact logging          
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Save model as pickle file
                            temp_model_path = os.path.join(temp_dir, f"{model_name}.pkl")
                            joblib.dump(model, temp_model_path)  # Actually save the model
                            mlflow.log_artifact(temp_model_path, "model")
                    
                            # Save and log model signature
                            signature_path = os.path.join(temp_dir, f"{model_name}_signature.txt")
                            with open(signature_path, 'w') as f:
                                f.write(f"Inputs: {signature.inputs.to_dict()}\n")
                                f.write(f"Outputs: {signature.outputs}\n")
                            mlflow.log_artifact(signature_path, "model")
                    
                            # Log model metadata
                            mlflow.log_param("model_type", type(model).__name__)
                            mlflow.log_param("sklearn_version", sklearn.__version__)
                    else:
                        if model_name == 'RandomForest':
                            mlflow.sklearn.log_model(model, "model") # type: ignore
                        else:
                            mlflow.xgboost.log_model(model, 'model') # type: ignore
                          
                except Exception as e:
                    print(f"Error evaluating {model_name}: {str(e)}")
                    continue
    
        #Save all results locally
        if all_results:
            results_path = Path(self.config.metric_file_name)
            save_json(path=results_path, data=all_results)
        
        return all_results