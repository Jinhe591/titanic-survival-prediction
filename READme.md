Titanic Survival Prediction

https://img.shields.io/badge/Python-3.8%252B-blue
https://img.shields.io/badge/scikit--learn-1.7.2-orange
https://img.shields.io/badge/Pandas-2.3.3-green
https://img.shields.io/badge/License-MIT-yellow

A comprehensive machine learning project that predicts passenger survival on the RMS Titanic using the classic dataset from Kaggle. The project includes an automated pipeline for model training and an interactive notebook for making predictions on new passenger data.
📊 Project Overview

This project applies modern data mining techniques to analyze survival patterns from the Titanic disaster. It features two main components:

    Automated Pipeline: A complete data preprocessing, feature engineering, and model training workflow

    Interactive Notebook: A small interactive application that predicts survival rates for user-input passenger details

The Random Forest classifier achieves 81.56% accuracy in predicting survival outcomes.
Key Features

    Complete exploratory data analysis with visualizations

    Automated data preprocessing and feature engineering pipeline

    Random Forest classification model with feature importance analysis

    Interactive notebook for real-time survival predictions

    Modular, reusable code structure

🗂️ Project Structure
text

titanic-survival-prediction/
│
├── README.md                       # Project documentation
├── requirements.txt                 # Dependencies
├── run_pipeline.py                  # Main automated pipeline script
│
├── data/
│   └── Titanic-Dataset.csv          # Original dataset from Kaggle
│
├── notebooks/
│   ├── 01_titanic_eda.ipynb         # Exploratory Data Analysis & Interactive Predictions
│   └── 02_titanic_modeling.ipynb    # Model Building & Evaluation
│
├── src/
│   ├── __init__.py                   # Package initializer
│   ├── data_preprocessing.py         # Data cleaning functions
│   ├── feature_engineering.py         # Feature creation
│   └── model.py                       # Model training & prediction
│
└── reports/
    └── Titanic_Survival_Prediction_Report.pdf  # Detailed project report

🚀 Quick Start
Prerequisites

    Python 3.8 or higher

    pip package manager

Installation

    Clone the repository
    bash

    git clone https://github.com/YOUR-USERNAME/titanic-survival-prediction.git
    cd titanic-survival-prediction

    Install dependencies
    bash

    pip install -r requirements.txt

🎯 Two Ways to Use This Project
Option 1: Run the Automated Pipeline

Execute the complete data processing and model training pipeline:
bash

python run_pipeline.py

This will:

    Load and clean the Titanic dataset

    Engineer new features (FamilySize, Alone)

    Train a Random Forest model

    Display model accuracy and feature importance

Option 2: Interactive Predictions in Notebook

Launch Jupyter and open the modeling notebook for interactive predictions:
bash

jupyter notebook notebooks/02_titanic_modeling.ipynb

The notebook includes an interactive section where you can input passenger details and get real-time survival predictions. Example usage from the notebook:
python

# Create sample new passengers
new_passengers = pd.DataFrame({
    'Pclass': [1, 3, 2],
    'Sex': [0, 1, 0],  # 0=female, 1=male
    'Age': [25, 30, 17],
    'SibSp': [0, 1, 2],
    'Parch': [0, 0, 1],
    'Fare': [100.0, 7.5, 25.0],
    'Embarked': [0, 1, 2],
    'FamilySize': [1, 2, 4],
    'Alone': [1, 0, 0]
})

# Get predictions
predictions = model.predict(new_passengers)
probabilities = model.predict_proba(new_passengers)

📈 Results

The Random Forest model achieved:

    Accuracy: 81.56%

    Precision (Survived): 0.79

    Recall (Survived): 0.76

    F1-Score (Survived): 0.77

Feature Importance
Feature	Importance
Sex	28.1%
Fare	25.5%
Age	24.6%
Pclass	8.7%
FamilySize	4.5%
Embarked	3.1%
SibSp	2.7%
Parch	2.0%
Alone	0.8%
🔍 Key Insights from EDA

    Women had significantly higher survival rates (74.2% vs 18.9% for men)

    Children (<18) survived at 54.0% compared to 36.1% for adults

    First-class passengers survived at 63.0% vs 24.2% for third-class

    Medium-sized families (2-4 members) had the best survival odds

    Higher fare correlated with better survival chances

🎮 Interactive Prediction Examples

Test the model with these sample scenarios from the notebook:
Passenger	Class	Sex	Age	Family	Fare	Prediction	Confidence
1	1st	Female	25	Alone	$100	Survive	99%
2	3rd	Male	30	With spouse	$7.5	Not Survive	86%
3	2nd	Female	17	With family	$25	Survive	91%
🛠️ Technologies Used

    Python 3.8+ - Core programming language

    Pandas & NumPy - Data manipulation and analysis

    Matplotlib & Seaborn - Data visualization

    Scikit-learn - Machine learning algorithms (Random Forest, LabelEncoder)

    Jupyter Notebooks - Interactive development and predictions

📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
📬 Author
Hadi Assi