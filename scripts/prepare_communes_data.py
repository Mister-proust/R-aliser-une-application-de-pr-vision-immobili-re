import sys
import os
import pandas as pd
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Affichage dans le terminal
        logging.FileHandler('all_logs.log', mode='a')  # fichier de log en mode append
    ],
    force=True  # Force la configuration du logging pour écraser les précédentes configurations
)

logger = logging.getLogger(__name__)

import requests

def get_commune_info(name_or_code):
    """
    Récupère la latitude, longitude et densité (hab/km²) d'une commune
    à partir de son nom (str) ou de son code INSEE (int).
    Utilise l'API Découpage administratif (geo.api.gouv.fr) pour les données.
    """
    try:
        # Préparer la requête selon le type d'entrée
        if isinstance(name_or_code, str):
            # Recherche par nom de commune (première correspondance)
            params = {
                'nom': name_or_code,
                'fields': 'code,centre,surface,population',
                'limit': 1
            }
            res = requests.get("https://geo.api.gouv.fr/communes", params=params, timeout=10)
            res.raise_for_status()
            communes = res.json()
            if not communes:
                print(f"Commune '{name_or_code}' non trouvée.")
                return {}
            data = communes[0]
            nom = name_or_code
            code = data.get("code")
        else:
            # Recherche par code INSEE (format 5 chiffres)
            code = f"{name_or_code:05d}"
            res = requests.get(f"https://geo.api.gouv.fr/communes/{code}",
                               params={'fields': 'nom,centre,surface,population'}, timeout=10)
            res.raise_for_status()
            data = res.json()
            nom = data.get("nom")
        # Extraire les informations utiles
        
        centre = data.get("centre")
        if not centre or "coordinates" not in centre:
            print("Coordonnées non disponibles pour la commune.")
            return {}
        
        lon, lat = centre["coordinates"]
        surface = data.get("surface")
        population = data.get("population")
        # Calcul de la densité (habitants par km²)
        densite = None
        if surface and population is not None:
            densite = population / surface
        
        return {"nom": nom, "code INSEE": code, "latitude": lat, "longitude": lon, "surface": surface, "population": population, "densite": densite}

    except requests.RequestException as e:
        print(f"Erreur réseau ou API : {e}")
        return {}

def load_communes_file(file_path: str) -> pd.DataFrame:
    """
    Charge le fichier des communes dans un DataFrame pandas.
    
    Parameters:
        file_path (str): Chemin vers le fichier DVF
    
    Returns:
        pd.DataFrame: DataFrame contenant les données du fichier DVF
    """

    df = pd.DataFrame()
    # Vérification de l'existence du fichier
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas.")
    # Chargement du fichier DVF

    col_to_keep = [
        'code_insee',
        'nom_standard',
        'population',
        'superficie_km2',
        'densite',
        'niveau_equipements_services',
        'latitude_centre',
        'longitude_centre',
        'altitude_moyenne'
    ]
    
    col_type = {
        'code_insee' : 'object',
        'nom_standard' : 'object',
        'population' : 'Int32',
        'superficie_km2' : 'Int32',
        'densite' : 'float32',
        'niveau_equipements_services' : 'float32',
        'latitude_centre' : 'float32',
        'longitude_centre' : 'float32',
        'altitude_moyenne' : 'Int32'
    }
    
    df = pd.read_csv(
        file_path,
        sep=',',  # Le séparateur est une barre verticale
        decimal='.',  # Le séparateur décimal est une virgule
        usecols=col_to_keep,  # On ne garde que les colonnes nécessaires
        dtype=col_type,  # On spécifie les types de colonnes
        na_values=['NULL', '', '-']  # Valeurs à considérer comme manquantes
    )
    return df

def clean_communes_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame DVF.
    
    Parameters:
        df (pd.DataFrame): DataFrame contenant les données DVF
    
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """

    # On ne garde que les lignes où population != 0
    df = df[df['population'] != 0]

    # 132 communes ont 'niveau_equipements_services' à NaN, on les remplace par la moyenne 2.0
    df.loc[df['niveau_equipements_services'].isna(), 'niveau_equipements_services'] = 2.0
    
    # On supprime les doublons s'il y en a. Ici il n'y en a pas mais si jamais un nouveau fichhier en contient...
    df = df.drop_duplicates()

    # Pas tres propre mais on remplit les coordonnées manquantes à la main pour quelques villes
    df.loc[df['nom_standard'] == 'Marseille', ['latitude_centre', 'longitude_centre']]  = [43.2965, 5.3698]
    df.loc[df['nom_standard'] == 'Culey', ['latitude_centre', 'longitude_centre']]  = [48.75, 5.2669]
    df.loc[df['nom_standard'] == 'Lyon', ['latitude_centre', 'longitude_centre']]  = [45.76, 4.8320]
    df.loc[df['nom_standard'] == 'Paris', ['latitude_centre', 'longitude_centre']]  = [48.85, 2.3484]
    df.loc[df['nom_standard'] == 'Bihorel', ['latitude_centre', 'longitude_centre']]  = [49.45, 1.1160]
    df.loc[df['nom_standard'] == 'Saint-Lucien', ['latitude_centre', 'longitude_centre']]  = [48.65, 1.6231]

    # LEs code insee du fichier commune regroupe les arrondissements de Paris, Lyon et Marseille
    # Par simplification, on duplique les lignes correspondantes pour chaque arrondissement mais il faudrait completer 
    # avec les bonnes valeurs de population, superficie, densité, etc. issue d'une api insee.
    
    # Pour Paris
   
    for code in range(75101, 75121):
        data = get_commune_info(code)
        if data:
            row = pd.DataFrame([{
            "code_insee": data.get("code INSEE"),
            "nom_standard": data.get("nom"),
            "population": data.get("population"),
            "superficie_km2": data.get("surface"),
            "densite":data.get("densite"),
            "latitude_centre": data.get("latitude"),
            "longitude_centre": data.get("longitude"),
            "altitude_moyenne": df.loc[df['code_insee'] == '75056', 'altitude_moyenne'].iloc[0],
            "niveau_equipements_services": df.loc[df['code_insee'] == '75056', 'niveau_equipements_services'].iloc[0]
            }])
            df = pd.concat([df, row], ignore_index=True)
        time.sleep(0.2)  # petite pause pour éviter de spammer l'API

    # Pour Lyon
    for code in range(69381, 69390):
        data = get_commune_info(code)
        if data:
            row = pd.DataFrame([{
            "code_insee": data.get("code INSEE"),
            "nom_standard": data.get("nom"),
            "population": data.get("population"),
            "superficie_km2": data.get("surface"),
            "densite":data.get("densite"),
            "latitude_centre": data.get("latitude"),
            "longitude_centre": data.get("longitude"),
            "altitude_moyenne": df.loc[df['code_insee'] == '69123', 'altitude_moyenne'].iloc[0],
            "niveau_equipements_services": df.loc[df['code_insee'] == '69123', 'niveau_equipements_services'].iloc[0]
            }])
            df = pd.concat([df, row], ignore_index=True)
        time.sleep(0.2)  # petite pause pour éviter de spammer l'API

    
    # Pour Marseille
    for code in range(13201, 13217):
        data = get_commune_info(code)
        if data:
            row = pd.DataFrame([{
            "code_insee": data.get("code INSEE"),
            "nom_standard": data.get("nom"),
            "population": data.get("population"),
            "superficie_km2": data.get("surface"),
            "densite":data.get("densite"),
            "latitude_centre": data.get("latitude"),
            "longitude_centre": data.get("longitude"),
            "altitude_moyenne": df.loc[df['code_insee'] == '13055', 'altitude_moyenne'].iloc[0],
            "niveau_equipements_services": df.loc[df['code_insee'] == '13055', 'niveau_equipements_services'].iloc[0]
            }])
            df = pd.concat([df, row], ignore_index=True)
        time.sleep(0.2)  # petite pause pour éviter de spammer l'API

    return df

def save_df_to_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Sauvegarde le DataFrame DVF dans un fichier CSV.
    
    Parameters:
        df (pd.DataFrame): DataFrame contenant les données DVF
        output_path (str): Path et Nom du fichier de sortie
    """

    # On sauvegarde le DataFrame dans un fichier CSV
    try:

        df.to_csv(output_path, index=False, sep=';')
        logger.info(f"Le fichier a été sauvegardé sous {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier CSV : {e}")

if __name__ == "__main__":
    filename = sys.argv[1]
    df = load_communes_file(filename)
    df = clean_communes_data(df)
    # Retirer l'extension existante (si présente) avant d'ajouter .csv
    file = os.path.basename(filename)
    base, _ = os.path.splitext(file)
    output_filename = os.path.join('data', 'prepared', base + ".csv")
    save_df_to_csv(df, output_filename)
  
