import requests
import logging
from typing import Optional, List, Dict, Any
from .instance import mcp

logger = logging.getLogger(__name__)

BASE_URL = "https://data.geopf.fr/geocodage"

@mcp.tool()
def geocoding_search(
    q: str,
    index: str = "address",
    limit: int = 5,
    postcode: Optional[str] = None,
    citycode: Optional[str] = None,
    city: Optional[str] = None,
    type: Optional[str] = None
) -> str:
    """
    Search for geographic coordinates or addresses in France.
    :param q: The character string to search for (ex: '73 Avenue de Paris Saint-Mandé').
    :param index: Search index: 'address' (default), 'poi' (places), 'parcel' (parcels).
    :param limit: Maximum number of results (max 50).
    :param postcode: Filter by postal code.
    :param citycode: Filter by INSEE code.
    :param city: Filter by town name.
    :param type: Address data type (eg: 'housenumber', 'street', 'municipality').
    :return: A formatted string containing the search results.
    """
    endpoint = f"{BASE_URL}/search"
    params = {
        "q": q,
        "index": index,
        "limit": limit,
        "autocomplete": 0
    }
    
    if postcode: params["postcode"] = postcode
    if citycode: params["citycode"] = citycode
    if city: params["city"] = city
    if type: params["type"] = type

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        
        features = data.get("features", [])
        if not features:
            return f"Aucun résultat trouvé pour la recherche : '{q}'"

        results = []
        for feat in features:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            coords = geom.get("coordinates", [0, 0])
            label = props.get("label", "N/A")
            score = props.get("score", 0)
            res = f"- {label} (Lon: {coords[0]}, Lat: {coords[1]}) [Score: {score:.2f}]"
            results.append(res)

        return "\n".join(results)

    except Exception as e:
        logger.error(f"Erreur lors du géocodage : {e}")
        return f"Une erreur est survenue lors de la recherche : {str(e)}"

@mcp.tool()
def reverse_geocoding(
    lon: float,
    lat: float,
    index: str = "address",
    limit: int = 1
) -> str:
    """
    Finds the address or place corresponding to geographic coordinates (longitude, latitude).
    :param lon: Longitude of the point.
    :param lat: Latitude of the point.
    :param index: Search index: 'address', 'poi', 'parcel'.
    :param limit: Number of results.
    :return: The found address or place.
    """
    endpoint = f"{BASE_URL}/reverse"
    params = {
        "lon": lon,
        "lat": lat,
        "index": index,
        "limit": limit
    }

    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        
        features = data.get("features", [])
        if not features:
            return f"Aucune adresse trouvée pour les coordonnées {lat}, {lon}."

        results = []
        for feat in features:
            props = feat.get("properties", {})
            label = props.get("label", "N/A")
            distance = props.get("distance", 0)
            results.append(f"- {label} (à {distance:.1f}m)")

        return "\n".join(results)

    except Exception as e:
        logger.error(f"Erreur lors du géocodage inverse : {e}")
        return f"Une erreur est survenue lors du géocodage inverse : {str(e)}"
