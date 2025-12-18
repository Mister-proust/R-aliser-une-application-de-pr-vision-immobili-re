
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, status, Header
from fastapi.responses import JSONResponse
from typing import Annotated, Dict, Any, List, Optional
from datetime import timedelta
import pickle
from fastapi.security import OAuth2PasswordRequestForm
import pandas as pd
import app.config as config  # Assurez-vous que ce fichier existe
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

# Setup logger
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Cache global du modèle ---
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

# --- Constantes (manquantes dans votre script original) ---
# J'ajoute des placeholders pour que le script soit fonctionnel.
# Remplissez-les avec vos vraies valeurs de mapping.
TYPE_VOIE_MAP = {
    "Avenue": "AV",
    "Boulevard": "BD",
    "Rue": "RUE",
    # etc.
}
TYPE_LOCAL_MAP = {
    "Appartement": "Appartement",
    "Maison": "Maison",
    # etc.
}

# --- Fonctions de l'API ---

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
    
    # Simule l'existence de config.API_TOKEN si config n'est pas entièrement setup
    # Dans un vrai scénario, config.API_TOKEN doit être défini.
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
    
    # 1. Déterminer si l'entrée est un code INSEE ou un nom
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
            'boost': 'population', # Priorise les grandes villes
            'limit': 1
        }

    try:
        # 2. Exécuter l'appel API
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status() # Lève une erreur si le statut HTTP est 4xx ou 5xx
        
        data = response.json()
        
        # 3. GESTION DES ERREURS (votre contrainte)
        if not data:
            # Si data est une liste vide, l'API n'a rien trouvé
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail=f"Commune non trouvée: '{commune_or_insee}'"
            )
        
        commune_data = data[0]
        
        # 4. Extraire les données
        population = commune_data.get('population')
        surface_hectares = commune_data.get('surface') 
        coordinates = commune_data.get('centre', {}).get('coordinates')
        
        # Validation plus souple avec conversion sécurisée
        pop_safe = safe_float(population, 0.0)
        surf_safe = safe_float(surface_hectares, 0.0)
        
        # 5. Calculer la densité
        surface_km2 = surf_safe / 100.0
        densite = 0.0
        if surface_km2 > 0:
            densite = pop_safe / surface_km2
        
        lat = coordinates[1] if coordinates and len(coordinates) > 1 else 0.0
        lon = coordinates[0] if coordinates and len(coordinates) > 0 else 0.0
        
        # 6. Retourner le dictionnaire EXACT que 'prediction_model' attend
        return {
            "densite": round(densite, 2),
            "population": pop_safe,
            "superficie_km2": round(surface_km2, 2),
            "latitude_centre": lat,
            "longitude_centre": lon
        }

    except requests.exceptions.RequestException as e:
        # L'API Géo est peut-être inaccessible
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail=f"Erreur de connexion à l'API Géo: {e}"
        )
    except Exception as e:
        # Attrape d'autres erreurs inattendues (ex: parsing JSON)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Erreur interne lors de la recherche de la commune: {e}"
        )


def prediction_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Charge le modèle et réalise une prédiction de prix immobilier à partir des données utilisateur.
    """
    # --- 1️⃣ Chargement du modèle (utilise la fonction commune) ---
    pipeline = load_model()

    # --- 2️⃣ Récupération des infos communes ---
    commune_key = payload.get("code_insee") or payload.get("commune")
    
    if commune_key:
        # Si la clé existe, get_commune_info est appelée.
        # Si la commune n'est pas trouvée, cette fonction lèvera une HTTPException 404
        # et l'exécution de 'prediction_model' s'arrêtera ici.
        commune_features = get_commune_info(str(commune_key))
    else:
        # Si l'utilisateur n'a fourni ni code_insee ni commune
        commune_features = {
            "densite": 0.0,
            "population": 0.0,
            "superficie_km2": 0.0,
            "latitude_centre": 0.0,
            "longitude_centre": 0.0
        }

    # --- 3️⃣ Mapping des valeurs utilisateur vers celles du modèle ---
    type_voie = TYPE_VOIE_MAP.get(payload.get("type_voie", ""), payload.get("type_voie", ""))
    type_local = TYPE_LOCAL_MAP.get(payload.get("type_local", ""), payload.get("type_local", ""))

    # --- 4️⃣ Construction de l'entrée modèle ---
    # S'assure que les clés correspondent exactement à celles attendues par votre pipeline
    # Support des clés alternatives venant du frontend simplifié
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

    # --- 5️⃣ Prédiction ---
    try:
        prediction = pipeline.predict(df_input)
        
        # S'assure que la prédiction est un nombre simple
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
        # 1. Charger le modèle et l'explainer
        pipeline = load_model()
        explainer = load_explainer()
        
        if not explainer:
            return {
                "error": "Explainer SHAP non disponible pour ce modèle",
                "status": "error"
            }
        
        # 2. Récupérer les infos communes
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
        
        # 3. Construire l'entrée modèle (identique à prediction_model)
        type_voie = TYPE_VOIE_MAP.get(payload.get("type_voie", ""), payload.get("type_voie", ""))
        type_local = TYPE_LOCAL_MAP.get(payload.get("type_local", ""), payload.get("type_local", ""))
        
        # Support des clés alternatives venant du frontend simplifié
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
        
        # 4. Préparer les données (passer par le preprocessor de la pipeline)
        preprocessor = pipeline.named_steps.get('scaler') or pipeline.named_steps.get('preprocessor') or pipeline.named_steps.get('pre')
        if preprocessor:
            X_processed = preprocessor.transform(df_input)
        else:
            X_processed = df_input.values
        
        # 5. Générer l'explication SHAP
        # Pour waterfall, on a besoin d'un objet Explanation, pas juste des valeurs numpy
        explanation = explainer(X_processed)
        
        # L'objet explanation peut contenir plusieurs lignes, on prend la première (et unique)
        # shap_values_val est le tableau numpy des valeurs SHAP (pour la rétrocompatibilité du JSON de réponse)
        shap_values_val = explanation.values
        if len(shap_values_val.shape) > 1: # Si (1, features)
             shap_values_val = shap_values_val[0]
        
        # Assigner les noms de features à l'explication pour que le graphique soit lisible
        explanation.feature_names = df_input.columns.tolist()
        
        # 6. Prédiction
        prediction = pipeline.predict(df_input)[0]
        
        # 7. Générer un graphique SHAP (Summary plot encodé en base64)
        plt.figure(figsize=(10, 4))
        try:
            # Summary plot
            # shap.summary_plot(shap_values, X_processed, feature_names=df_input.columns, plot_type="bar", show=False)
            
            # Waterfall plot (requiert un objet Explanation [index])
            # On prend explanation[0] car on a une seule prédiction
            shap.plots.waterfall(explanation[0], show=False)
            #shap.plots.beeswarm(explanation, show=False)
            
            # Convertir en base64
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
            # Si le graphique échoue, retourner quand même les valeurs SHAP
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
        # Tente d'utiliser le pipeline de ML complet
        return prediction_model(payload)
    except HTTPException as e:
        # Si une HTTPException (ex: 404 commune non trouvée) est levée, 
        # la re-lève pour que FastAPI la gère
        raise e
    except Exception as e:
        # Si le pipeline échoue pour une autre raison (ex: chargement pickle, erreur pandas)
        # on utilise le fallback
        fallback = perform_prediction(payload)
        return {"warning": "pipeline prediction failed, returning fallback", "error": str(e), "result": fallback}
