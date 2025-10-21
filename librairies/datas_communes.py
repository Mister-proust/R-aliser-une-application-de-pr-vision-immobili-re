import os
import pandas as pd


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
        'niveau_equipements_services'
    ]
    
    col_type = {
        'code_insee' : 'object',
        'nom_standard' : 'object',
        'population' : 'Int32',
        'superficie_km2' : 'Int32',
        'densite' : 'float32',
        'niveau_equipements_services' : 'float32'
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

    # On remplace les densites naN par 0
    df.loc[df['densite'].isna(), 'densite'] = 0
    # 132 communes ont 'niveau_equipements_services' à NaN, on les remplace par la moyenne 2.0
    df.loc[df['niveau_equipements_services'].isna(), 'niveau_equipements_services'] = 2.0
    # On supprime les doublons s'il y en a. Ici il n'y en a pas mais si jamais un nouveau fichhier en contient...
    df = df.drop_duplicates()

    return df

