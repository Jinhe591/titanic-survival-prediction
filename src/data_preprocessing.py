"""
Data Preprocessing Module for Titanic Survival Prediction
Based on your working notebook code
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class TitanicDataPreprocessor:
    """Class for preprocessing Titanic dataset"""
    
    def __init__(self, filepath):
        """Initialize with path to Titanic dataset"""
        self.filepath = filepath
        self.data = None
        self.encoder = LabelEncoder()
    
    def load_data(self):
        """Load the Titanic dataset"""
        self.data = pd.read_csv(self.filepath)
        print(f"✅ Data loaded: {self.data.shape[0]} rows, {self.data.shape[1]} columns")
        return self.data
    
    def clean_data(self):
        """Run all cleaning steps at once (matches your notebook)"""
        if self.data is None:
            self.load_data()
        
        # Fill missing Age with median (from your cell 8)
        self.data['Age'].fillna(self.data['Age'].median(), inplace=True)
        
        # Fill missing Embarked with mode (from your cell 9)
        self.data['Embarked'].fillna(self.data['Embarked'].mode()[0], inplace=True)
        
        # Drop unnecessary columns (from your cell 10)
        self.data = self.data.drop(['PassengerId', 'Cabin', 'Ticket', 'Name'], axis=1)
        
        # Encode Sex (from your cell 12)
        self.data['Sex'] = self.encoder.fit_transform(self.data['Sex'])
        
        # Encode Embarked (from your cell 12)
        self.data['Embarked'] = self.encoder.fit_transform(self.data['Embarked'])
        
        print("✅ Data cleaning complete")
        return self.data
    
    def get_data(self):
        """Return the cleaned data"""
        return self.data


# Quick test
if __name__ == "__main__":
    # Test with your data folder structure
    prep = TitanicDataPreprocessor('data/Titanic-Dataset.csv')
    clean_data = prep.clean_data()
    print("\nFirst 5 rows:")
    print(clean_data.head())
    print(f"\nFinal shape: {clean_data.shape}")