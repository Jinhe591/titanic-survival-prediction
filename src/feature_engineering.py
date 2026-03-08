"""
Feature Engineering Module for Titanic Survival Prediction
Creates FamilySize and Alone features (from your cell 11)
"""

import pandas as pd

class TitanicFeatureEngineer:
    """Class for creating engineered features"""
    
    def __init__(self, data):
        """Initialize with dataframe"""
        self.data = data.copy()
    
    def create_family_features(self):
        """Create FamilySize and Alone features (matches your cell 11)"""
        # Create FamilySize
        self.data['FamilySize'] = self.data['SibSp'] + self.data['Parch'] + 1
        
        # Create Alone indicator
        self.data['Alone'] = (self.data['FamilySize'] == 1).astype(int)
        
        print("✅ Family features created")
        return self.data
    
    def get_data(self):
        """Return the data with new features"""
        return self.data


# Quick test
if __name__ == "__main__":
    # This is just for testing - will work after data_preprocessing
    print("This module is meant to be used after data_preprocessing")
    print("Example:")
    print("  from src.data_preprocessing import TitanicDataPreprocessor")
    print("  from src.feature_engineering import TitanicFeatureEngineer")
    print("  prep = TitanicDataPreprocessor('data/Titanic-Dataset.csv')")
    print("  clean_data = prep.clean_data()")
    print("  engineer = TitanicFeatureEngineer(clean_data)")
    print("  final_data = engineer.create_family_features()")