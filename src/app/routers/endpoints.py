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

def prediction_model(payload: Dict[str, Any]):
    """
    charge le fichier pickle, et lance une prédiction sur le bien donné en entrée
    
    Args:
        payload (Dict[str, Any]): payload JSON avec les caractéristiques du bien à estimer
    
    Returns:
        Dict[str, Any]: résultat de la prédiction
    """
    columns_list = ["type_voie_encodee", "type_local_encodee", "Surface terrain", "Surface reelle bati", "Nombre pieces principales", "densite", "population", "superficie_km2", "Valeur fonciere", "altitude_moyenne", "latitude_centre", "longitude_centre"]
    model_path = config.MODEL_PATH
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    input_df = pd.DataFrame([payload], columns=columns_list)
    prediction = model.predict(input_df)
    return {"estimated_price": prediction[0], "currency": "EUR", "input": payload}
    

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
    return perform_prediction(payload)