import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os # Import os for checking file existence

# --- Correction: Load data directly from the specified CSV file ---
FILE_NAME = "vehicle_sensor_data.csv"

if not os.path.exists(FILE_NAME):
    # This check ensures the script doesn't fail if run standalone
    print(f"Error: The file '{FILE_NAME}' was not found. Please ensure the data is saved.")
else:
    # 1. Load Data
    data = pd.read_csv(FILE_NAME)
    print(f"✅ Data loaded successfully from {FILE_NAME}. Total rows: {len(data)}")

    # 2. Define Features (X) and Target (y)
    X = data[['engine_temp', 'oil_pressure', 'vibration', 'battery_voltage', 'mileage', 'fuel_efficiency']]
    y = data['maintenance_required']

    # 3. Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dataset split: Train={len(X_train)} rows, Test={len(X_test)} rows")

    # 4. Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Data scaled.")

    # 5. Train model
    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train_scaled, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")

    # 7. Save model and scaler
    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("✅ Model and Scaler saved successfully!")