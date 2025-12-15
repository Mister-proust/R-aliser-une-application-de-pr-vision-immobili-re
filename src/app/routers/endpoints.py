from email.policy import HTTP
from fastapi import APIRouter, HTTPException, Depends, Query, UploadFile, File, status, Header
from fastapi.responses import JSONResponse
from typing import Annotated, Dict, Any, List, Optional
from datetime import timedelta
import pickle
from fastapi.security import OAuth2PasswordRequestForm
import pandas as pd
import app.config as config  # Assurez-vous que ce fichier existe
import requests

router = APIRouter()

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
        
        if not all([population, surface_hectares, coordinates]):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                detail=f"Données incomplètes reçues de l'API Géo pour '{commune_or_insee}'"
            )

        # 5. Calculer la densité
        surface_km2 = surface_hectares / 100.0
        densite = 0
        if surface_km2 > 0:
            densite = population / surface_km2
        
        # 6. Retourner le dictionnaire EXACT que 'prediction_model' attend
        return {
            "densite": round(densite, 2),
            "population": float(population),
            "superficie_km2": round(surface_km2, 2),
            "latitude_centre": coordinates[1],
            "longitude_centre": coordinates[0]
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
    # --- 1️⃣ Chargement du modèle ---
    # S'assure que config.MODEL_PATH existe
    model_path = getattr(config, "MODEL_PATH", "model.pkl") 
    try:
        with open(model_path, "rb") as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Fichier modèle introuvable à: {model_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement du modèle: {e}")


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

    # --- 4️⃣ Construction de l’entrée modèle ---
    # S'assure que les clés correspondent exactement à celles attendues par votre pipeline
    df_input = pd.DataFrame([{
        "latitude_centre": commune_features.get("latitude_centre", 0.0),
        "longitude_centre": commune_features.get("longitude_centre", 0.0),
        "Surface reelle bati": float(payload.get("surface_reelle_bati", 0.0)),
        "Nombre pieces principales": int(payload.get("nombre_pieces_principales", 0.0)),
        "Surface terrain": float(payload.get("surface_terrain", 0.0)),        
        "densite": commune_features.get("densite", 0.0),
        "population": commune_features.get("population", 0.0),
        "superficie_km2": commune_features.get("superficie_km2", 0.0),
        "altitude_moyenne": 0.0,
        "niveau_equipements_services": 2.0,
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