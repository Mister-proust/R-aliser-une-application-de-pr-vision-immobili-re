import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import sys

# Chargement des données
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))

df = pd.read_csv(os.path.join(DATA_DIR, "clean_dvf.csv"), sep=';')

# Nettoyage des valeurs extrêmes
df = df[(df["Valeur fonciere"] <= 800000) & (df["Valeur fonciere"] >= 50000)].reset_index(drop=True)

# Use raw categorical columns and build a preprocessing pipeline that encodes categories
categorical_features = ["Type de voie", "Type local"]
numeric_features = [
    "Surface terrain",
    "Surface reelle bati",
    "Nombre pieces principales",
    "densite",
    "population",
    "superficie_km2",
    "latitude_centre",
    "longitude_centre",
]

# Prepare X and y using raw columns
features = categorical_features + numeric_features
X = df[features]
y = df["Valeur fonciere"]

# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=8)

# Build a ColumnTransformer that encodes categorical features and scales numeric ones
preprocessor = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features),
    ("num", StandardScaler(), numeric_features),
])

# Pipeline with preprocessing + XGBoost
pipeline = Pipeline([
    ("pre", preprocessor),
    ("model", xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=8
    ))
])

# Entraînement du modèle
pipeline.fit(X_train, y_train)

# Prédictions
y_pred = pipeline.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {np.sqrt(mse):.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.3f}")

# Sauvegarde du pipeline complet (preprocessor + modèle)
os.makedirs(os.path.join(DATA_DIR, "models"), exist_ok=True)
with open(os.path.join(DATA_DIR, "models", "xgb_pipeline.pkl"), "wb") as f:
    pickle.dump(pipeline, f)

# Chargement du pipeline
with open(os.path.join(DATA_DIR, "models", "xgb_pipeline.pkl"), "rb") as f:
    loaded_pipeline = pickle.load(f)

# Test du pipeline chargé
y_pred_loaded = loaded_pipeline.predict(X_test)
print(f"Test du modèle chargé - R²: {r2_score(y_test, y_pred_loaded):.3f}")

