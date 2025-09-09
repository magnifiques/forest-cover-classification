# Forest Cover Type Classification

An end-to-end machine learning project for predicting forest cover types using cartographic variables. This project implements a comprehensive MLOps pipeline, encompassing data ingestion, validation, transformation, model training, evaluation, experiment tracking, and deployment.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Components](#pipeline-components)
- [Model Performance](#model-performance)
- [Experiment Tracking](#experiment-tracking)
- [Deployment](#deployment)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🌲 Project Overview

This project predicts forest cover types based on cartographic variables derived from US Forest Service (USFS) Region 2 Resource Information System data. The goal is to classify forest cover into one of seven categories using various environmental and geographic features.

The project follows MLOps best practices with:
- Modular pipeline architecture
- Comprehensive experiment tracking
- Automated hyperparameter tuning
- Model versioning and deployment
- Continuous integration and deployment

## 📊 Dataset

The Forest Cover Type dataset contains cartographic variables for 30x30 meter cells obtained from US Forest Service Region 2 Resource Information System data.

**Features:**
- **Quantitative Variables:** Elevation, Aspect, Slope, distances to water features, roads, and fire points
- **Qualitative Variables:** Wilderness areas (4 binary columns) and soil types (40 binary columns)
- **Target:** 7 forest cover types

**Dataset Statistics:**
- Total samples: 581,012
- Features: 54
- Classes: 7 (Cover_Type 1-7)
- No missing values

## 🏗️ Project Architecture

```
Data Ingestion → Data Validation → Data Transformation → Model Training → Model Evaluation → Deployment
      ↓                ↓                  ↓                    ↓                ↓              ↓
   Raw Data      Schema Check      Feature Engineering    RandomForest      Metrics     GitHub Pages
                Data Quality         Preprocessing         XGBoost        Evaluation      DagsHub
                   Check           Train/Test Split    Hyperparameter   MLflow Logging
                                                         Tuning
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Git

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/magnifiques/forest-cover-classification.git
cd forest-cover-classification
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🔧 Pipeline Components

### 1. Data Ingestion
- Downloads dataset from source
- Performs initial data quality checks
- Splits data into train and test sets
- Saves raw data artifacts

### 2. Data Validation
- Schema validation against predefined structure
- Data drift detection
- Missing value analysis
- Outlier detection and reporting

### 3. Data Transformation
- Feature scaling and normalization
- Categorical encoding
- Feature selection
- Data preprocessing pipeline creation

### 4. Model Training
- **Random Forest Classifier**
  - Ensemble method with bootstrap aggregation
  - Hyperparameter tuning via RandomSearchCV
  
- **XGBoost Classifier**
  - Gradient boosting framework
  - Advanced regularization techniques
  - Hyperparameter optimization

### 5. Model Evaluation
- Cross-validation metrics
- Confusion matrix analysis
- Feature importance ranking
- Model comparison and selection

## 📈 Model Performance

### Best Model Results

| Model | Accuracy |
|-------|----------|
| XGBoost | 0.96 | 
| Random Forest | 0.963 | 

### Hyperparameter Tuning Results

**XGBoost Optimal Parameters:**
- `n_estimators`: 200
- `max_depth`: 6
- `learning_rate`: 0.2
- `subsample`: 0.7
- `colsample_bytree`: 1.0

**Random Forest Optimal Parameters:**
- `n_estimators`: 300
- `max_depth`: 30
- `min_samples_split`: 5
- `min_samples_leaf`: 2

## 🔬 Experiment Tracking

This project uses **MLflow** for comprehensive experiment tracking:

### Tracked Metrics
- Accuracy, Precision, Recall, F1-Score
- Training and validation loss
- Cross-validation scores
- Hyperparameter combinations

### Tracked Artifacts
- Trained models (pickle format)
- Feature importance plots
- Confusion matrices
- Data preprocessing pipelines

### Accessing Experiments
- **Local MLflow UI**: `http://localhost:5000`
- **DagsHub Integration**: 

## 🚀 Deployment

### GitHub Pages
The model is deployed as a web application using GitHub Pages with a simple interface for predictions.

**Live Demo**: [Your GitHub Pages URL]

### DagsHub Integration
- Model versioning and registry
- Experiment comparison and visualization
- Collaborative ML workflows

**DagsHub Repository**: [[Your DagsHub URL]](https://dagshub.com/magnifiques/end-to-end-data-science-project)

## 🛠️ Technologies Used

**Machine Learning:**
- scikit-learn
- XGBoost
- pandas
- numpy

**Experiment Tracking:**
- MLflow
- DagsHub

**Deployment:**
- GitHub Actions
- GitHub Pages
- Docker (optional)

**Development:**
- Python 3.8+
- Jupyter Notebooks
- Git

## 📁 Project Structure

```
forest-cover-classification/
├── data/
│   ├── raw/                            # Raw dataset storage
│   ├── processed/                      # Processed/transformed data
│   └── external/                       # External data sources
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py          # Data loading and splitting
│   │   ├── data_validation.py         # Schema and quality validation  
│   │   ├── data_transformation.py     # Feature preprocessing
│   │   ├── model_training.py          # RandomForest & XGBoost training
│   │   └── model_evaluation.py        # Model evaluation metrics
│   │
│   ├── pipeline/
│   │   ├── training_pipeline.py       # Complete training workflow
│   │   └── prediction_pipeline.py     # Inference pipeline
│   │
│   ├── utils/
│   │   ├── common.py                  # Common utility functions
│   │   └── logger.py                  # Logging configuration
│   │
│   └── config/
│       └── configuration.py           # Configuration management
│
├── artifacts/
│   ├── models/                        # Trained model storage
│   ├── data_validation/               # Validation reports
│   └── model_evaluation/              # Evaluation metrics
│
├── mlruns/                            # MLflow experiment tracking
│
├── notebooks/
│   ├── EDA.ipynb                      # Exploratory Data Analysis
│   ├── model_experiments.ipynb        # Model experimentation
│   └── data_analysis.ipynb            # Data analysis notebooks
│
├── static/                            # Web app static files (CSS, JS)
├── templates/                         # HTML templates for web app
├── tests/                             # Unit and integration tests
│
├── requirements.txt                   # Python dependencies
├── main.py                           # Main pipeline execution
├── app.py                            # Flask web application
├── Dockerfile                        # Container configuration
└── README.md                         # Project documentation
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

⭐ **Star this repository if you found it helpful!**
