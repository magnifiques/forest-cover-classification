import os
from src.project.config.configuration import ModelTrainConfig
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib
from src.project.utils.common import save_best_params
from sklearn.experimental import enable_halving_search_cv  
from sklearn.model_selection import HalvingRandomSearchCV # type: ignore
from sklearn.preprocessing import LabelEncoder

class ModelTrain:
    def __init__(self, config: ModelTrainConfig):
        self.config = config

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        validation_data = pd.read_csv(self.config.validation_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        encoder = LabelEncoder()
        
        
        
        train_x = train_data.drop([self.config.target_column], axis=1)
        validation_x = validation_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        
        
        train_y = encoder.fit_transform(train_data[self.config.target_column])
        validation_y = encoder.transform(validation_data[self.config.target_column])
        test_y = encoder.transform(test_data[self.config.target_column])

        # Select model & prepare hyperparameter grid
        if self.config.model_type == "RandomForest":
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                "n_estimators": self.config.n_estimators or [100],
                "max_depth": self.config.max_depth or [None],
                "min_samples_split": self.config.min_samples_split or [2],
                "min_samples_leaf": self.config.min_samples_leaf or [1]
            }
        elif self.config.model_type == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            param_grid = {
                "n_estimators": self.config.n_estimators or [100],
                "learning_rate": self.config.learning_rate or [0.1],
                "max_depth": self.config.max_depth or [3],
                "subsample": self.config.subsample or [1.0],
                "colsample_bytree": self.config.colsample_bytree or [1.0]
            }
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # Random search for hyperparameter tuning
        # random_search = RandomizedSearchCV(
        #     estimator=model,
        #     param_distributions=param_grid,
        #     n_iter=10,
        #     cv=2,
        #     scoring="accuracy",
        #     n_jobs=-1,
        #     random_state=42,
        #     verbose=2
        # )
        halving = HalvingRandomSearchCV(
        estimator=model,
        param_distributions=param_grid,
        factor=3,  # reduction factor
        random_state=42,
        n_jobs=-1,
        scoring="accuracy",
        verbose=2)
        
        
        print(f"Training {self.config.model_type} with Halving Random Search CV...")
        print(f"Parameter grid: {param_grid}")
        halving.fit(train_x, train_y)
        best_model = halving.best_estimator_
        

        # Evaluate on validation set
        val_pred = best_model.predict(validation_x) # type: ignore
        val_acc = accuracy_score(validation_y, val_pred)
        val_f1 = f1_score(validation_y, val_pred, average="weighted")
        
        print(f"{self.config.model_type} Best Params: {halving.best_params_}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation F1-Score: {val_f1:.4f}")
        print("Classification Report:")
        print(classification_report(validation_y, val_pred))

        # Save the best model
        model_filename = f"{self.config.model_type}_{self.config.model_name}"
        model_path = os.path.join(self.config.root_dir, model_filename)
        joblib.dump(best_model, model_path)
        print(f"Model saved to: {model_path}")
        
        save_best_params(self.config.model_type, halving.best_params_)
        
        return {
            'model': best_model,
            'validation_accuracy': val_acc,
            'validation_f1': val_f1,
            'best_params': halving.best_params_
        }