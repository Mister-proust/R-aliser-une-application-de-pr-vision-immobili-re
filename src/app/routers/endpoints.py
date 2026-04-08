
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, status, Header
from fastapi.responses import JSONResponse
from typing import Annotated, Dict, Any, List, Optional
from datetime import timedelta
import pickle
from fastapi.security import OAuth2PasswordRequestForm
import pandas as pd
import app.config as config
import requests
import shap
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import os
import numpy as np
import logging
import traceback
logger = logging.getLogger(__name__)

router = APIRouter()
_model_cache = None
_explainer_cache = None

def load_model():
    """
    Charge le modèle pickle de manière commune (utilisé par prediction_model et shap_explanation).
    Utilise le cache pour éviter les rechargements.
    """
    global _model_cache
    
    if _model_cache is not None:
        return _model_cache
    
    model_path = getattr(config, "MODEL_PATH", "models/xgb_pipeline.pkl") 
    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
        _model_cache = pipeline
        return pipeline
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Fichier modèle introuvable à: {model_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du modèle: {e}")

def load_explainer():
    """
    Crée l'explainer SHAP à partir du modèle chargé.
    """
    global _explainer_cache
    
    if _explainer_cache is not None:
        return _explainer_cache
    
    try:
        pipeline = load_model()
        xgb_model = pipeline.named_steps.get('model')
        if xgb_model:
            _explainer_cache = shap.TreeExplainer(xgb_model)
        return _explainer_cache
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la création de l'explainer SHAP: {e}")

def safe_float(value, default=0.0):
    """Convertit une valeur en float de manière sécurisée"""
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Convertit une valeur en int de manière sécurisée"""
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return default
TYPE_VOIE_MAP = {
    "Avenue": "AV",
    "Boulevard": "BD",
    "Rue": "RUE",
}
TYPE_LOCAL_MAP = {
    "Appartement": "Appartement",
    "Maison": "Maison",
}

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
    API_TOKEN = getattr(config, "API_TOKEN", "default_token_secret")
    
    if token != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return True


@router.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


@router.get("/meta/options", tags=["Meta"], summary="Renvoie les valeurs uniques pour les selects")
def meta_options():
    type_voie_vals = [
        "Avenue",
        "Boulevard",
        "Chemin",
        "Impasse",
        "Route",
        "Rue",
        "Autres"
    ]

    type_local_vals = [
        "Appartement",
        "Bâtiment industriel",
        "Maison"
    ]
    return {"type_voie": type_voie_vals, "type_local": type_local_vals}

def get_commune_info(commune_or_insee: str) -> Dict[str, Any]:
    """
    Récupère les informations géographiques d'une commune depuis l'API Géo.
    Lève une HTTPException 404 si la commune n'est pas trouvée.
    """
    BASE_URL = "https://geo.api.gouv.fr/communes"
    is_code_insee = (len(commune_or_insee) == 5 and 
                     (commune_or_insee.isdigit() or 
                      commune_or_insee[:2].upper() in ['2A', '2B']))
    
    if is_code_insee:
        params = {
            'code': commune_or_insee,
            'fields': 'nom,code,centre,population,surface'
        }
    else:
        params = {
            'nom': commune_or_insee,
            'fields': 'nom,code,centre,population,surface',
            'boost': 'population',
            'limit': 1
        }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        if not data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Commune non trouvée: '{commune_or_insee}'"
            )
        
        commune_data = data[0]
        population = commune_data.get('population')
        surface_hectares = commune_data.get('surface') 
        coordinates = commune_data.get('centre', {}).get('coordinates')
        pop_safe = safe_float(population, 0.0)
        surf_safe = safe_float(surface_hectares, 0.0)
        surface_km2 = surf_safe / 100.0
        densite = 0.0
        if surface_km2 > 0:
            densite = pop_safe / surface_km2
        
        lat = coordinates[1] if coordinates and len(coordinates) > 1 else 0.0
        lon = coordinates[0] if coordinates and len(coordinates) > 0 else 0.0
        return {
            "densite": round(densite, 2),
            "population": pop_safe,
            "superficie_km2": round(surface_km2, 2),
            "latitude_centre": lat,
            "longitude_centre": lon
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail=f"Erreur de connexion à l'API Géo: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erreur interne lors de la recherche de la commune: {e}"
        )


def prediction_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Charge le modèle et réalise une prédiction de prix immobilier à partir des données utilisateur.
    """
    pipeline = load_model()
    commune_key = payload.get("code_insee") or payload.get("commune")
    
    if commune_key:
        commune_features = get_commune_info(str(commune_key))
    else:
        commune_features = {
            "densite": 0.0,
            "population": 0.0,
            "superficie_km2": 0.0,
            "latitude_centre": 0.0,
            "longitude_centre": 0.0
        }
    type_voie = TYPE_VOIE_MAP.get(payload.get("type_voie", ""), payload.get("type_voie", ""))
    type_local = TYPE_LOCAL_MAP.get(payload.get("type_local", ""), payload.get("type_local", ""))
    surface = safe_float(payload.get("surface_reelle_bati"), 0.0)
    if surface == 0.0:
         surface = safe_float(payload.get("surface"), 0.0)
         
    pieces = safe_int(payload.get("nombre_pieces_principales"), 0)
    if pieces == 0:
        pieces = safe_int(payload.get("rooms"), 0)

    df_input = pd.DataFrame([{
        "Type de voie": type_voie,
        "Type local": type_local,
        "Surface terrain": safe_float(payload.get("surface_terrain"), 0.0),
        "Surface reelle bati": surface,
        "Nombre pieces principales": pieces,
        "densite": commune_features.get("densite", 0.0),
        "population": commune_features.get("population", 0.0),
        "superficie_km2": commune_features.get("superficie_km2", 0.0),
        "latitude_centre": commune_features.get("latitude_centre", 0.0),
        "longitude_centre": commune_features.get("longitude_centre", 0.0)
    }])
    try:
        prediction = pipeline.predict(df_input)
        estimated_price = float(prediction[0])
        if pd.isna(estimated_price):
             raise ValueError("La prédiction a retourné NaN")
            
        return {
            "estimated_price": estimated_price,
            "currency": "EUR",
            "input": df_input.to_dict(orient="records")[0]
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Erreur lors de la prédiction: {e}. Vérifiez les types de données d'entrée."
    )


def perform_prediction(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Effectue la prédiction via le modèle (pour l'instant formule random)
    [NOTE: Cette fonction semble être votre fallback, je la laisse telle quelle]
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


@router.post("/shap-explanation", tags=["SHAP"], summary="Explique une prédiction avec SHAP")
async def shap_explanation(payload: Dict[str, Any], auth: bool = Depends(verify_token)):
    """
    Retourne une explication SHAP pour une prédiction donnée (format base64 d'image).
    """
    try:
        pipeline = load_model()
        explainer = load_explainer()
        
        if not explainer:
            return {
                "error": "Explainer SHAP non disponible pour ce modèle",
                "status": "error"
            }
        commune_key = payload.get("code_insee") or payload.get("commune")
        
        if commune_key:
            commune_features = get_commune_info(str(commune_key))
        else:
            commune_features = {
                "densite": 0.0,
                "population": 0.0,
                "superficie_km2": 0.0,
                "latitude_centre": 0.0,
                "longitude_centre": 0.0
            }
        type_voie = TYPE_VOIE_MAP.get(payload.get("type_voie", ""), payload.get("type_voie", ""))
        type_local = TYPE_LOCAL_MAP.get(payload.get("type_local", ""), payload.get("type_local", ""))
        surface = safe_float(payload.get("surface_reelle_bati"), 0.0)
        if surface == 0.0:
             surface = safe_float(payload.get("surface"), 0.0)
             
        pieces = safe_int(payload.get("nombre_pieces_principales"), 0)
        if pieces == 0:
            pieces = safe_int(payload.get("rooms"), 0)
        
        df_input = pd.DataFrame([{
            "Type de voie": type_voie,
            "Type local": type_local,
            "Surface terrain": safe_float(payload.get("surface_terrain"), 0.0),
            "Surface reelle bati": surface,
            "Nombre pieces principales": pieces,
            "densite": commune_features.get("densite", 0.0),
            "population": commune_features.get("population", 0.0),
            "superficie_km2": commune_features.get("superficie_km2", 0.0),
            "latitude_centre": commune_features.get("latitude_centre", 0.0),
            "longitude_centre": commune_features.get("longitude_centre", 0.0)
        }])
        preprocessor = pipeline.named_steps.get('scaler') or pipeline.named_steps.get('preprocessor') or pipeline.named_steps.get('pre')
        if preprocessor:
            X_processed = preprocessor.transform(df_input)
        else:
            X_processed = df_input.values
        explanation = explainer(X_processed)
        shap_values_val = explanation.values
        if len(shap_values_val.shape) > 1:
             shap_values_val = shap_values_val[0]
        explanation.feature_names = df_input.columns.tolist()
        prediction = pipeline.predict(df_input)[0]
        plt.figure(figsize=(10, 4))
        try:
            shap.plots.waterfall(explanation[0], show=False)
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            return {
                "status": "success",
                "prediction": safe_float(prediction),
                "input": df_input.to_dict(orient="records")[0],
                "shap_plot": f"data:image/png;base64,{image_base64}",
                "shap_values": shap_values_val.tolist() if hasattr(shap_values_val, 'tolist') else shap_values_val,
                "feature_names": list(df_input.columns)
            }
        except Exception as plot_error:
            logger.error(f"SHAP Plot error: {plot_error}")
            traceback.print_exc()
            return {
                "status": "partial_success",
                "prediction": safe_float(prediction),
                "input": df_input.to_dict(orient="records")[0],
                "shap_values": shap_values_val.tolist() if hasattr(shap_values_val, 'tolist') else shap_values_val,
                "feature_names": list(df_input.columns),
                "plot_error": str(plot_error)
            }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in shap_explanation: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "status": "error"
        }


@router.post("/predict", tags=["Prediction"], summary="Prévision (requiert token)")
async def predict_protected(payload: Dict[str, Any], auth: bool = Depends(verify_token)):
    try:
        return prediction_model(payload)
    except HTTPException as e:
        raise e
    except Exception as e:
        fallback = perform_prediction(payload)
        return {"warning": "pipeline prediction failed, returning fallback", "error": str(e), "result": fallback}


@router.get("/map-data", tags=["Map"], summary="Récupère les données de prix pour la carte interactive")
async def get_map_data(
    south: float = Query(..., description="Latitude sud de la bounding box"),
    west: float = Query(..., description="Longitude ouest de la bounding box"),
    north: float = Query(..., description="Latitude nord de la bounding box"),
    east: float = Query(..., description="Longitude est de la bounding box"),
    zoom: int = Query(..., description="Niveau de zoom (6-18)")
):
    """
    Récupère les données de transactions immobilières agrégées selon le niveau de zoom.
    - zoom 6-8: Agrégation par département (code_commune[:2])
    - zoom 9-11: Agrégation par commune (code_commune)
    - zoom 12+: Points individuels avec clustering
    """
    import sqlite3
    import os
    from pathlib import Path
    base_dir = Path(__file__).parent.parent.parent.parent
    db_path = base_dir / "src" / "agentia" / "bdd" / "donnees_immo.db"
    
    if not db_path.exists():
        raise HTTPException(status_code=500, detail=f"Base de données introuvable: {db_path}")
    
    try:
        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        if zoom <= 7:
            query = """
                SELECT 
                    SUBSTR(CAST(code_commune AS TEXT), 1, 2) as zone_code,
                    AVG(valeur_fonciere / NULLIF(surface_reelle_bati, 0)) as prix_m2,
                    COUNT(*) as nb_transactions,
                    AVG(valeur_fonciere) as prix_moyen
                FROM Transactions
                WHERE latitude BETWEEN ? AND ?
                    AND longitude BETWEEN ? AND ?
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                    AND valeur_fonciere > 0
                    AND surface_reelle_bati > 0
                GROUP BY SUBSTR(CAST(code_commune AS TEXT), 1, 2)
                HAVING nb_transactions >= 5
            """
            level = "departement"
        elif zoom <= 10:
            query = """
                SELECT 
                    SUBSTR(CAST(code_commune AS TEXT), 1, 2) as zone_code,
                    AVG(valeur_fonciere / NULLIF(surface_reelle_bati, 0)) as prix_m2,
                    COUNT(*) as nb_transactions,
                    AVG(valeur_fonciere) as prix_moyen
                FROM Transactions
                WHERE latitude BETWEEN ? AND ?
                    AND longitude BETWEEN ? AND ?
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                    AND valeur_fonciere > 0
                    AND surface_reelle_bati > 0
                GROUP BY SUBSTR(CAST(code_commune AS TEXT), 1, 2)
                HAVING nb_transactions >= 5
            """
            level = "departement"
        else:
            query = """
                SELECT 
                    CAST(code_commune AS TEXT) as zone_code,
                    AVG(valeur_fonciere / NULLIF(surface_reelle_bati, 0)) as prix_m2,
                    COUNT(*) as nb_transactions,
                    AVG(valeur_fonciere) as prix_moyen
                FROM Transactions
                WHERE latitude BETWEEN ? AND ?
                    AND longitude BETWEEN ? AND ?
                    AND latitude IS NOT NULL
                    AND longitude IS NOT NULL
                    AND valeur_fonciere > 0
                    AND surface_reelle_bati > 0
                GROUP BY CAST(code_commune AS TEXT)
                HAVING nb_transactions >= 3
            """
            level = "commune"
        cursor.execute(query, (south, north, west, east))
        results = cursor.fetchall()
        data_points = []
        for row in results:
            zone_code, prix_m2, nb_trans, prix_moyen = row
            if prix_m2 is not None:
                data_points.append({
                    "zone_code": zone_code,
                    "prix_m2": round(prix_m2, 2),
                    "nb_transactions": nb_trans,
                    "prix_moyen": round(prix_moyen, 2)
                })
        
        connection.close()
        if data_points:
            prix_list = [p["prix_m2"] for p in data_points]
            stats = {
                "min": round(min(prix_list), 2),
                "max": round(max(prix_list), 2),
                "mean": round(sum(prix_list) / len(prix_list), 2),
                "count": len(data_points)
            }
        else:
            stats = {"min": 0, "max": 0, "mean": 0, "count": 0}
        
        return {
            "data": data_points,
            "stats": stats,
            "zoom_level": zoom,
            "level": level
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données de carte: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur base de données: {str(e)}")
