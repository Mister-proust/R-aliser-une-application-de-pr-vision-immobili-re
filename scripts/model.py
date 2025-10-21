import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# Encoder les colonnes suivantes : "Type de voie", "Voie", "code_insee"


df = pd.read_csv("../data/clean_dvf.csv", sep=';')
df = df[df["Valeur fonciere"] <= 800000].reset_index(drop=True)
df = df[df["Valeur fonciere"] >= 50000].reset_index(drop=True)
le = LabelEncoder()
df["type_voie_encodee"] = le.fit_transform(df["Type de voie"].astype(str))
df["type_local_encodee"] = le.fit_transform(df["Type local"].astype(str))
X = df[["type_voie_encodee", "type_local_encodee", "Surface terrain", "Surface reelle bati", "Nombre pieces principales", "densite", "population", "superficie_km2", "Valeur fonciere", "altitude_moyenne", "latitude_centre", "longitude_centre"]]
y = df["Valeur fonciere"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=8)

model = xgb.XGBRegressor()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

import pickle

# Enregistrer le modèle
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Charger le modèle plus tard
with open("xgb_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Tester le modèle chargé
y_pred_loaded = loaded_model.predict(X_test)


