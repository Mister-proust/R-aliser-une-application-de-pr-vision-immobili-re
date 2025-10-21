from email.policy import HTTP
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, status, Header
from fastapi.responses import JSONResponse
from typing import Annotated, Dict, Any, List, Optional
from datetime import timedelta
import pickle
from fastapi.security import OAuth2PasswordRequestForm
import pandas as pd
import app.config as config

router = APIRouter()


def verify_token(authorization: Optional[str] = Header(None)):
    """
    Vérifie un token d'authentification simple.
    Header demandé: Authorization: Bearer <token>
    """
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Authorization header format")
    token = parts[1]
    if token != config.API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return True


@router.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


@router.get("/meta/options", tags=["Meta"], summary="Renvoie les valeurs uniques pour les selects")
def meta_options():
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..'))
    clean_csv = os.path.join(repo_root, 'data', 'clean_dvf.csv')
    try:
        df_train = pd.read_csv(clean_csv, sep=';')
    except Exception:
        return {"type_voie": [], "type_local": []}

    type_voie_vals = sorted(df_train['Type de voie'].astype(str).dropna().unique().tolist()) if 'Type de voie' in df_train.columns else []
    type_local_vals = sorted(df_train['Type local'].astype(str).dropna().unique().tolist()) if 'Type local' in df_train.columns else []
    return {"type_voie": type_voie_vals, "type_local": type_local_vals}

def prediction_model(payload: Dict[str, Any]):
    """
    charge le fichier pickle, et lance une prédiction sur le bien donné en entrée
    
    Args:
        payload (Dict[str, Any]): payload JSON avec les caractéristiques du bien à estimer
    
    Returns:
        Dict[str, Any]: résultat de la prédiction
    """
    # The payload should contain property-level fields (raw, non-encoded):
    # - 'commune' or 'code_insee' (identifier for commune)
    # - 'type_voie' (string)
    # - 'type_local' (string)
    # - 'surface_terrain' (number)
    # - 'surface_reelle_bati' (number)
    # - 'nombre_pieces_principales' (number)
    # We will:
    # 1) load the saved pipeline
    # 2) recreate LabelEncoders from the training CSV to encode type_voie/type_local
    # 3) lookup commune features (densite, population, superficie_km2, altitude_moyenne, latitude_centre, longitude_centre)
    # 4) build the input dict with exact column names and call pipeline.predict

    model_path = config.MODEL_PATH
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)

    # The pipeline saved by scripts/model.py includes preprocessing (OrdinalEncoder + scaler)
    # so we can pass raw categorical strings and numeric columns directly.
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    communes_csv = os.path.join(repo_root, 'data', 'communes-france-2025.csv')
    try:
        df_communes = pd.read_csv(communes_csv)
    except Exception:
        # If communes file missing, set defaults to zeros
        df_communes = pd.DataFrame()

    # Normalize commune key columns in df_communes for matching
    # Prefer code_insee if provided, otherwise match on uppercased name
    code_insee = payload.get('code_insee')
    commune_name = payload.get('commune')

    # Create a small record with commune features defaulting to 0
    commune_features = {
        'densite': 0.0,
        'population': 0.0,
        'superficie_km2': 0.0,
        'altitude_moyenne': 0.0,
        'latitude_centre': 0.0,
        'longitude_centre': 0.0,
    }

    if not df_communes.empty:
        if code_insee and 'code_insee' in df_communes.columns:
            match = df_communes[df_communes['code_insee'].astype(str) == str(code_insee)]
            if not match.empty:
                row = match.iloc[0]
                for k in commune_features.keys():
                    if k in df_communes.columns:
                        try:
                            commune_features[k] = float(row.get(k) if pd.notna(row.get(k)) else 0.0)
                        except Exception:
                            commune_features[k] = 0.0
        elif commune_name:
            # find a name column to match
            name_col = None
            for c in ['nom_standard_majuscule', 'nom', 'commune']:
                if c in df_communes.columns:
                    name_col = c
                    break
            if name_col:
                match = df_communes[df_communes[name_col].astype(str).str.upper() == str(commune_name).upper()]
                if not match.empty:
                    row = match.iloc[0]
                    for k in commune_features.keys():
                        if k in df_communes.columns:
                            try:
                                commune_features[k] = float(row.get(k) if pd.notna(row.get(k)) else 0.0)
                            except Exception:
                                commune_features[k] = 0.0

    # Build raw input DataFrame with categorical strings and numeric features
    raw_input = {
        'Type de voie': payload.get('type_voie') or '',
        'Type local': payload.get('type_local') or '',
        'Surface terrain': payload.get('surface_terrain') or 0.0,
        'Surface reelle bati': payload.get('surface_reelle_bati') or 0.0,
        'Nombre pieces principales': payload.get('nombre_pieces_principales') or 0.0,
        'densite': commune_features['densite'],
        'population': commune_features['population'],
        'superficie_km2': commune_features['superficie_km2'],
        'altitude_moyenne': commune_features['altitude_moyenne'],
        'latitude_centre': commune_features['latitude_centre'],
        'longitude_centre': commune_features['longitude_centre'],
    }

    # Build DataFrame in the order expected by the pipeline's feature columns
    # Pipeline expects categorical columns first then numeric ones as we defined during training
    df_input = pd.DataFrame([raw_input])
    prediction = pipeline.predict(df_input)
    return {"estimated_price": float(prediction[0]), "currency": "EUR", "input": raw_input}
    

def perform_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Effectue la prédiction via le modèle (pour l'instant formule random)

    Args:
        payload (Dict[str, Any]): payload JSON avec les caractéristiques du bien à estimer

    Returns:
        Dict[str, Any]: résultat de la prédiction
    """
    try:
        surface = float(payload.get("surface") or 0)
    except Exception:
        surface = 0.0
    try:
        rooms = int(payload.get("rooms") or 0)
    except Exception:
        rooms = 0
    prix_m2 = 2500
    bonus_par_piece = 5000
    estimated_price = round(surface * prix_m2 + rooms * bonus_par_piece, 2)
    return {"estimated_price": estimated_price, "currency": "EUR", "input": payload}


@router.post("/predict", tags=["Prediction"], summary="Prévision (requiert token)")
async def predict_protected(payload: Dict[str, Any], auth: bool = Depends(verify_token)):
    try:
        return prediction_model(payload)
    except Exception as e:
        # If the pipeline fails (missing model file, bad payload), return a fallback prediction and error info
        fallback = perform_prediction(payload)
        return {"warning": "pipeline prediction failed, returning fallback", "error": str(e), "result": fallback}