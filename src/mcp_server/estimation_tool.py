import os
import pickle
import pandas as pd
import requests
import logging
from typing import Dict, Any, Optional
import app.config as config
from .instance import mcp

logger = logging.getLogger(__name__)

_model_cache = None

def load_model():
    """
    Load the pickle model.
    Use cache to avoid reloads.
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
        alt_path = os.path.join(os.path.dirname(__file__), "..", "data", "models", "xgb_pipeline.pkl")
        try:
            with open(alt_path, "rb") as f:
                pipeline = pickle.load(f)
            _model_cache = pipeline
            return pipeline
        except FileNotFoundError:
            raise Exception(f"Fichier modèle introuvable à: {model_path} ou {alt_path}")
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du modèle: {e}")

def safe_float(value, default=0.0):
    """Converts a value to float securely"""
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Converts a value to int securely"""
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

def get_commune_info(commune_or_insee: str) -> Dict[str, Any]:
    """
    Retrieves the geographic information of a municipality from the Geo API.
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
            return None
        
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
    except Exception as e:
        logger.error(f"Erreur lors de la recherche de la commune: {e}")
        return None

@mcp.tool()
def estimate_property(
    commune: str,
    type_bien: str,
    surface: int,
    rooms: int = 1,
    surface_terrain: int = 0,
    type_voie: str = "Rue"
) -> str:
    """
    Estimate the price of real estate in France.
    :param commune: Name of the commune or INSEE code.
    :param property_type: Type of property ('House' or 'Apartment').
    :param surface: Living space in m2.
    :param rooms: Number of main rooms.
    :param surface_land: Surface area of ​​the land in m2 (0 if not applicable).
    :param type_voie: Type of road (eg: 'Street', 'Avenue', 'Boulevard').
    :return: A string containing the estimated price.
    """
    try:
        pipeline = load_model()

        commune_features = get_commune_info(commune)
        if not commune_features:
            return f"Désolé, je n'ai pas pu trouver d'informations pour la commune : {commune}."

        type_voie_mapped = TYPE_VOIE_MAP.get(type_voie, type_voie)
        type_local_mapped = TYPE_LOCAL_MAP.get(type_bien, type_bien)

        df_input = pd.DataFrame([{
            "Type de voie": type_voie_mapped,
            "Type local": type_local_mapped,
            "Surface terrain": safe_float(surface_terrain, 0.0),
            "Surface reelle bati": safe_float(surface, 0.0),
            "Nombre pieces principales": safe_int(rooms, 1),
            "densite": commune_features.get("densite", 0.0),
            "population": commune_features.get("population", 0.0),
            "superficie_km2": commune_features.get("superficie_km2", 0.0),
            "latitude_centre": commune_features.get("latitude_centre", 0.0),
            "longitude_centre": commune_features.get("longitude_centre", 0.0)
        }])

        prediction = pipeline.predict(df_input)
        estimated_price = float(prediction[0])

        return f"L'estimation pour votre {type_bien} de {surface}m² ({rooms} pièces) à {commune} est de {estimated_price:,.0f} €."

    except Exception as e:
        logger.error(f"Erreur lors de l'estimation : {e}")
        return f"Une erreur est survenue lors de l'estimation : {str(e)}"
