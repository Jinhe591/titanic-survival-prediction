"""
Model Building Module for Titanic Survival Prediction
Based on your Random Forest implementation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class TitanicModel:
    """Class for building and evaluating Titanic survival model"""
    
    def __init__(self, data):
        """Initialize with dataframe (must include 'Survived' column)"""
        self.data = data.copy()
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data into features and target (matches your cells 15-16)"""
        # Separate features and target
        X = self.data.drop('Survived', axis=1)
        y = self.data['Survived']
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"✅ Data prepared: {len(self.X_train)} training, {len(self.X_test)} testing")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, n_estimators=100, random_state=42):
        """Train Random Forest model (matches your cell 17)"""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state
        )
        self.model.fit(self.X_train, self.y_train)
        print("✅ Model trained successfully")
        return self.model
    
    def evaluate(self):
        """Evaluate model performance (matches your cells 19-21)"""
        if self.model is None:
            raise ValueError("Train the model first!")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\n📊 Model Accuracy: {accuracy:.2%}")
        
        # Show sample predictions (like your cell 19)
        print(f"\nSample predictions (first 10):")
        print(f"Predicted: {y_pred[:10]}")
        print(f"Actual:    {self.y_test.values[:10]}")
        
        # Detailed report (like your cell 21)
        print(f"\n📋 Classification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['Did not survive', 'Survived']))
        
        return accuracy, y_pred
    
    def get_feature_importance(self, feature_names):
        """Get feature importance (matches your cell 22)"""
        if self.model is None:
            raise ValueError("Train the model first!")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📈 Feature Importance:")
        print(importance_df)
        return importance_df
    
    def predict_new(self, new_data):
        """Predict for new passengers (matches your cell 23)"""
        if self.model is None:
            raise ValueError("Train the model first!")
        
        predictions = self.model.predict(new_data)
        probabilities = self.model.predict_proba(new_data)
        
        return predictions, probabilities


# Quick test
if __name__ == "__main__":
    print("This module is meant to be used after data_preprocessing and feature_engineering")
    print("\nFull pipeline example:")
    print("  from src.data_preprocessing import TitanicDataPreprocessor")
    print("  from src.feature_engineering import TitanicFeatureEngineer")
    print("  from src.model import TitanicModel")
    print("\n  # 1. Load and clean data")
    print("  prep = TitanicDataPreprocessor('data/Titanic-Dataset.csv')")
    print("  clean_data = prep.clean_data()")
    print("\n  # 2. Add features")
    print("  engineer = TitanicFeatureEngineer(clean_data)")
    print("  final_data = engineer.create_family_features()")
    print("\n  # 3. Train model")
    print("  model = TitanicModel(final_data)")
    print("  model.prepare_data()")
    print("  model.train_model()")
    print("  model.evaluate()")