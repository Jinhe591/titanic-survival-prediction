🚢 Titanic Survival Prediction

A comprehensive machine learning project that predicts passenger survival on the Titanic using the classic Kaggle dataset.

The project contains:

A fully automated ML pipeline

Exploratory Data Analysis (EDA)

A Random Forest classification model

Interactive predictions through a Jupyter notebook

The final model achieves 81.56% accuracy in predicting survival outcomes.

📊 Project Overview

This project applies modern data mining and machine learning techniques to analyze survival patterns from the Titanic disaster.

It contains two main components:

Automated Pipeline

A full workflow that performs:

Data cleaning

Feature engineering

Model training

Feature importance analysis

Interactive Notebook

A notebook where users can input passenger details and predict survival probability.

✨ Key Features

Full Exploratory Data Analysis (EDA)

Automated data preprocessing pipeline

Feature engineering (FamilySize, Alone)

Random Forest classifier

Feature importance analysis

Interactive survival prediction notebook

Clean and modular code structure

🗂️ Project Structure
titanic-survival-prediction/
│
├── README.md
├── requirements.txt
├── run_pipeline.py
│
├── data/
│   └── Titanic-Dataset.csv
│
├── notebooks/
│   ├── 01_titanic_eda.ipynb
│   └── 02_titanic_modeling.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── model.py
│
└── reports/
    └── Titanic_Survival_Prediction_Report.pdf
🚀 Quick Start
Prerequisites

Python 3.8+

pip package manager

Installation
1. Clone the repository
git clone https://github.com/YOUR-USERNAME/titanic-survival-prediction.git
cd titanic-survival-prediction
2. Install dependencies
pip install -r requirements.txt
🎯 How to Use the Project
Option 1 — Run the Automated Pipeline

Run the full data pipeline and train the model:

python run_pipeline.py

This will:

Load and clean the Titanic dataset

Engineer new features (FamilySize, Alone)

Train the Random Forest model

Display accuracy and feature importance

Option 2 — Interactive Predictions (Notebook)

Launch Jupyter:

jupyter notebook notebooks/02_titanic_modeling.ipynb

Inside the notebook you can enter passenger details and receive real-time survival predictions.

📈 Results
Model Performance
Metric	Score
Accuracy	81.56%
Precision (Survived)	0.79
Recall (Survived)	0.76
F1 Score (Survived)	0.77
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

Women had significantly higher survival rates
(74.2% vs 18.9% for men)

Children (<18) survived more often
(54.0% vs 36.1% for adults)

First-class passengers had better survival chances
(63.0% vs 24.2% for third-class)

Medium-sized families (2–4 members) had the best survival odds

Higher ticket fares correlated with higher survival probability

🎮 Example Predictions
Passenger	Class	Sex	Age	Family	Fare	Prediction	Confidence
1	1st	Female	25	Alone	$100	Survive	99%
2	3rd	Male	30	With spouse	$7.5	Not Survive	86%
3	2nd	Female	17	With family	$25	Survive	91%
🛠️ Technologies Used

Python 3.8+

Pandas & NumPy — Data manipulation

Matplotlib & Seaborn — Data visualization

Scikit-learn — Machine learning models

Jupyter Notebook — Interactive development

📝 License

This project is licensed under the MIT License.

👤 Author

Hadi Assi