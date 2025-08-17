# train_tabnet.py (Corrected)
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

print("--- Starting TabNet Training and Saving Process ---")

# 1. Load Data
try:
    data = pd.read_csv('creditcard.csv')
    print("✅ Data loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: creditcard.csv not found. Please make sure it's in the same directory.")
    exit()

# 2. Prepare Data
# Separate features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Split data (we only need a small fraction to train and save)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("✅ Data prepared and scaled.")

# 3. Initialize and Train TabNet Model
clf = TabNetClassifier()

# THE FIX IS HERE: We pass y_train.values and y_test.values directly as 1D arrays.
clf.fit(
    X_train, y_train.values,
    eval_set=[(X_test, y_test.values)],
    max_epochs=5,
    patience=3
)
print("✅ Model training complete.")

# 4. Save the Model Correctly
# This will create a perfectly formatted .zip file.
save_path = os.path.join('models', 'tabnet', 'tabnet_model')
clf.save_model(save_path)

print(f"\n🎉 SUCCESS: Model saved as 'tabnet_model.zip' inside the 'models/tabnet/' directory.")
print("--- Script Finished ---")