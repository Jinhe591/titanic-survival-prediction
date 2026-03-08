"""
Complete pipeline script for Titanic Survival Prediction
Run this to see everything working together
"""

from src.data_preprocessing import TitanicDataPreprocessor
from src.feature_engineering import TitanicFeatureEngineer
from src.model import TitanicModel

print("="*60)
print("TITANIC SURVIVAL PREDICTION PIPELINE")
print("="*60)

# Step 1: Load and clean data
print("\n📥 STEP 1: Loading and cleaning data...")
prep = TitanicDataPreprocessor('data/Titanic-Dataset.csv')
clean_data = prep.clean_data()

# Step 2: Engineer features
print("\n🔧 STEP 2: Engineering features...")
engineer = TitanicFeatureEngineer(clean_data)
final_data = engineer.create_family_features()

# Step 3: Train and evaluate model
print("\n🤖 STEP 3: Training model...")
model = TitanicModel(final_data)
model.prepare_data()
model.train_model()
model.evaluate()

# Step 4: Show feature importance
feature_names = final_data.drop('Survived', axis=1).columns.tolist()
model.get_feature_importance(feature_names)

print("\n" + "="*60)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("="*60)