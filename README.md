# 🚢 Titanic Survival Prediction

A comprehensive **machine learning project** that predicts passenger
survival on the Titanic using the classic Kaggle dataset.

The project includes:

-   A fully automated machine learning pipeline
-   Exploratory Data Analysis (EDA)
-   A Random Forest classification model
-   Interactive predictions through a Jupyter notebook

The final model achieves **81.56% accuracy** in predicting survival
outcomes.

------------------------------------------------------------------------

# 📊 Project Overview

This project applies modern **data mining and machine learning
techniques** to analyze survival patterns from the Titanic disaster.

### Main Components

**Automated Pipeline** - Data cleaning - Feature engineering - Model
training - Feature importance analysis

**Interactive Notebook** - Input passenger details - Predict survival
probability in real time

------------------------------------------------------------------------

# ✨ Key Features

-   Complete **Exploratory Data Analysis**
-   Automated preprocessing pipeline
-   Feature engineering (FamilySize, Alone)
-   **Random Forest classifier**
-   Feature importance analysis
-   Interactive prediction notebook
-   Modular code structure

------------------------------------------------------------------------

# 🗂️ Project Structure

    titanic-survival-prediction/

    ├── README.md
    ├── requirements.txt
    ├── run_pipeline.py

    ├── data/
    │   └── Titanic-Dataset.csv

    ├── notebooks/
    │   ├── 01_titanic_eda.ipynb
    │   └── 02_titanic_modeling.ipynb

    ├── src/
    │   ├── __init__.py
    │   ├── data_preprocessing.py
    │   ├── feature_engineering.py
    │   └── model.py

    └── reports/
        └── Titanic_Survival_Prediction_Report.docx

------------------------------------------------------------------------

# 🚀 Quick Start

## Prerequisites

-   Python 3.8+
-   pip

------------------------------------------------------------------------

# Installation

### Clone the repository

``` bash
git clone https://github.com/YOUR-USERNAME/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🎯 How to Use

## Option 1 --- Run the Automated Pipeline

    python run_pipeline.py

This will:

-   Load and clean the dataset
-   Create new features
-   Train the Random Forest model
-   Display accuracy and feature importance

------------------------------------------------------------------------

## Option 2 --- Interactive Notebook

    jupyter notebook notebooks/02_titanic_modeling.ipynb

Use the notebook to input passenger information and receive survival
predictions.

------------------------------------------------------------------------

# 📈 Results

  Metric                 Score
  ---------------------- ------------
  Accuracy               **81.56%**
  Precision (Survived)   0.79
  Recall (Survived)      0.76
  F1 Score (Survived)    0.77

------------------------------------------------------------------------

# Feature Importance

  Feature      Importance
  ------------ ------------
  Sex          28.1%
  Fare         25.5%
  Age          24.6%
  Pclass       8.7%
  FamilySize   4.5%
  Embarked     3.1%
  SibSp        2.7%
  Parch        2.0%
  Alone        0.8%

------------------------------------------------------------------------

# 🔍 Key Insights

-   Women survived far more often than men
-   Children had higher survival rates than adults
-   First class passengers had the best survival chances
-   Medium sized families had the highest survival probability
-   Higher fares correlated with survival

------------------------------------------------------------------------

# 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   Jupyter Notebook

------------------------------------------------------------------------

# 📝 License

MIT License

------------------------------------------------------------------------

# 👤 Author

**Hadi Assi**
