import sys
import os
import pandas as pd
import logging

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
       "code_insee", "nom_standard_majuscule", "population", "superficie_km2", "densite", "altitude_moyenne", "latitude_centre", "longitude_centre","niveau_equipements_services"
    ]
    
    col_type = {
        'code_insee' : 'object',
        'nom_standard' : 'object',
        'population' : 'Int32',
        'superficie_km2' : 'Int32',
        'densite' : 'float32',
        'altitude_moyenne' : 'float32',
        'latitude_centre' : 'float32',
        'longitude_centre' : 'float32',
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
    # idem pour altitude_moyenne, latitude_centre, longitude_centre
    df.loc[df['altitude_moyenne'].isna(), 'altitude_moyenne'] = 0
    df.loc[df['latitude_centre'].isna(), 'latitude_centre'] = 0
    df.loc[df['longitude_centre'].isna(), 'longitude_centre'] = 0
    # 132 communes ont 'niveau_equipements_services' à NaN, on les remplace par la moyenne 2.0
    df.loc[df['niveau_equipements_services'].isna(), 'niveau_equipements_services'] = 2.0
    # On supprime les doublons s'il y en a. Ici il n'y en a pas mais si jamais un nouveau fichhier en contient...
    df = df.drop_duplicates()

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
        # S'assurer que le répertoire de sortie existe
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

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
  
