Markdown

# Sentiment Analysis Using Machine Learning and Deep Learning

## Project Description

This project implements a comprehensive sentiment analysis system that compares the performance of traditional machine learning algorithms with deep learning approaches. Five distinct models are trained and evaluated: Logistic Regression, Naive Bayes, Support Vector Machine (SVM), Random Forest, and Gated Recurrent Unit (GRU). The project includes complete data preprocessing, feature engineering, model training, evaluation, and model persistence for deployment.

The implementation demonstrates best practices in natural language processing, including text cleaning, TF-IDF vectorization for traditional ML models, sequence tokenization for neural networks, and systematic performance comparison across different algorithmic paradigms.

## Folder Structure
FINAL_ASSIGNMENT/
│
├── data/
│ └── dataset.csv 
│
├── models/
│ ├── gru_model.h5 
│ ├── logistic_regression.pkl 
│ ├── svm.pkl 
│ ├── random_forest.pkl 
│ ├── multinomial_nb.pkl 
│ ├── tfidf_vectorizer.pkl 
│ ├── label_encoder.pkl 
│ └── tokenizer.pkl 
│
├── notebooks/
│ └── Code.ipynb 
│
├── report/
│ ├── Report.pdf 
│ └── AI_Module_II_Assessment.pdf 
│
├── requirements.txt 
└── README.md 

text


## Installation Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Setup Steps

1. Clone or download this repository to your local machine

2. Navigate to the project directory:
   ```bash
   cd FINAL_ASSIGNMENT
Create a virtual environment (recommended):

Bash

python -m venv venv
Activate the virtual environment:

Windows:
Bash

venv\Scripts\activate
macOS/Linux:
Bash

source venv/bin/activate
Install required packages:

Bash

pip install -r requirements.txt
Launch Jupyter Notebook:

Bash

jupyter notebook
How to Run the Notebook
Open notebooks/Code.ipynb in Jupyter Notebook

Update the file paths in the notebook to match your local directory structure:

Dataset path in the data loading cell
Model saving paths in the final cells
Run all cells sequentially from top to bottom using:

Shift + Enter to run individual cells
Kernel > Restart & Run All to execute the entire notebook
The notebook will:

Load and preprocess the dataset
Train all five models
Display accuracy metrics and classification reports
Generate a comparison visualization
Save all trained models to the models/ directory
Expected runtime: 5-15 minutes depending on your hardware (GRU training takes the longest)

Required Python Version
Python 3.8 or higher recommended
Tested on Python 3.8, 3.9, and 3.10
TensorFlow compatibility requires Python <= 3.11
Model Files Description
Traditional ML Models (.pkl files)
logistic_regression.pkl: Serialized Logistic Regression classifier trained on TF-IDF features
svm.pkl: Linear Support Vector Machine classifier
random_forest.pkl: Random Forest ensemble classifier with 200 estimators
multinomial_nb.pkl: Multinomial Naive Bayes probabilistic classifier
Deep Learning Model
gru_model.h5: Keras GRU neural network model with embedding layer
Preprocessing Components
tfidf_vectorizer.pkl: Fitted TF-IDF vectorizer (required for traditional ML model inference)
label_encoder.pkl: Label encoder for sentiment classes (required for all models)
tokenizer.pkl: Keras tokenizer for text sequences (required for GRU model inference)
Loading Saved Models
Python

import joblib
from tensorflow.keras.models import load_model

# Load traditional ML models
lr_model = joblib.load('models/logistic_regression.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

# Load deep learning model
gru_model = load_model('models/gru_model.h5')
tokenizer = joblib.load('models/tokenizer.pkl')
Notes
Ensure the dataset file dataset.csv is present in the data/ directory before running
The dataset should contain at minimum two columns: 'Review' (text) and 'Sentiment' (label)
Model training is deterministic with fixed random seeds for reproducibility
GPU acceleration is optional for TensorFlow but will significantly speed up GRU training
All file paths use absolute paths in the original code - modify to relative paths if needed
The GRU model uses early stopping to prevent overfitting
Troubleshooting
Import errors: Ensure all packages from requirements.txt are installed

Memory errors: Reduce max_features in TF-IDF or max_words in tokenizer

Path errors: Update all file paths to match your directory structure

TensorFlow warnings: These are typically informational and can be ignored

Author
Aashutosh Kuikel
Kuikelaashutosh@gmail.com


For questions or issues, please contact the author or refer to the project report in report/Report.pdf

