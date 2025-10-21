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

def load_dvf_file(file_path: str) -> pd.DataFrame:
    """
    Charge le fichier DVF dans un DataFrame pandas.
    
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
        'Date mutation',
        'Nature mutation',
        'Valeur fonciere',
        'No voie',
        'B/T/Q',
        'Type de voie',
        'Code voie',
        'Voie',
        'Code postal',
        'Commune',
        'Code departement',
        'Code commune',
        'Prefixe de section',
        'Section',
        'No plan',
        'Code type local',
        'Type local',
        'Surface reelle bati',
        'Nombre pieces principales',
        'Surface terrain'
    ]
    
    col_type = {
        'Nature mutation' : 'object',
        'Valeur fonciere' : 'float32',
        'No voie' : 'object',
        'B/T/Q' : 'object',
        'Type de voie' : 'object',
        'Code voie' : 'object',
        'Voie' : 'object',
        'Code postal' : 'object',
        'Commune' : 'object',
        'Code departement' : 'object',
        'Code commune' : 'object',
        'Prefixe de section' : 'object',
        'Section' : 'object',
        'No plan' : 'Int16',
        'Code type local' : 'Int8',
        'Type local' : 'object',
        'Surface reelle bati' : 'Int32',
        'Nombre pieces principales' : 'Int8',
        'Surface terrain' : 'Int32'
    }
    
    df = pd.read_csv(
        file_path,
        sep='|',  # Le séparateur est une barre verticale
        decimal=',',  # Le séparateur décimal est une virgule
        usecols=col_to_keep,  # On ne garde que les colonnes nécessaires
        dtype=col_type,  # On spécifie les types de colonnes
        parse_dates=['Date mutation'],  # On parse la colonne des dates au  format datetime
        na_values=['NULL', '', '-']  # Valeurs à considérer comme
    )
    return df

def clean_dvf_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame DVF.
    
    Parameters:
        df (pd.DataFrame): DataFrame contenant les données DVF
    
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """

    # On supprime les lignes dont la nature de mutation est 'Echange'
    df = df[df['Nature mutation'] != 'Echange']
    
    # On élimine les lignes dont les valeurs foncières sont manquantes
    df = df[df['Valeur fonciere'].notna()]
    
    # Les valeurs foncières en virgule flottante ne nous intéressent pas vraiment, on va les convertir en entiers
    df['Valeur fonciere'] = df['Valeur fonciere'].astype('int32')
    
    # Supprimer les doublons (conserver la première occurrence)
    df = df.drop_duplicates()
    # On va compléter les NaN des Surfaces et nombre de pièces par 0
    df.loc[:, 'Surface reelle bati'] = df['Surface reelle bati'].fillna(0)
    df.loc[:, 'Surface terrain'] = df['Surface terrain'].fillna(0)
    df.loc[:, 'Nombre pieces principales'] = df['Nombre pieces principales'].fillna(0)
    
    # De même sur la colonne B/T/Q par un champ vide
    df.loc[:, 'B/T/Q'] = df['B/T/Q'].fillna('')
    df.loc[:, 'No voie'] = df['No voie'].fillna('')
    df.loc[:, 'Type de voie'] = df['Type de voie'].fillna('')
    df.loc[:, 'Voie'] = df['Voie'].fillna('')
    df.loc[:, 'Code postal'] = df['Code postal'].fillna('')
    df.loc[:, 'Prefixe de section'] = df['Prefixe de section'].fillna('')
    df.loc[:, 'Section'] = df['Section'].fillna('')
    
    # Le prix d'un terrain étant insignifiant dans le prix d'un bien immobilier,
    # nous allons supprimer les lignes : 
    #     dont le type de local est Nan car ils n'ont pas de surface bati 
    #     dont le type de local est Dépendances car ils n'ont que des que des surfaces de terrain.
    # Ce choix est propre au projet qui ne concernera que les investissmeent en surface bati soit :  maisons, appartements et locaux industriels 
    filtre = (df['Type local'].isna()) | (df['Type local'] == 'Dépendance') 
    df = df[~filtre]


    # On regroupe les transactions par date et valeurs et on crée un id_transaction pour chacun de ces couples
    df.loc[:, 'Date mutation'] = pd.to_datetime(df['Date mutation'], format='%d/%m/%Y', errors='coerce') # on s'assure que toutes les dates soient au même format pour le tri, c'est mieux !
    df = df.sort_values(by=['Date mutation', 'Valeur fonciere', 'Code departement', 'Code commune']).reset_index(drop=True) 
    col_id_transaction = df.groupby(['Date mutation', 'Valeur fonciere', 'Code departement', 'Code commune', 'Code voie']).ngroup()
    df.insert(loc=0, column='id_transaction', value=col_id_transaction)

    # On ne garde que les transactions avec une ligne unique car il y a souvent les memes biens qui apparaissent 2 fois sur la même transaction
    # Sans doute dû à des corrections dans les champs.
    id_transaction_count = df['id_transaction'].value_counts()
    unique_id_transactions_values = id_transaction_count[id_transaction_count == 1].index
    unique_id_transactions_list = unique_id_transactions_values.tolist()
    df =  df[df['id_transaction'].isin(unique_id_transactions_list)]

    # On efface les lignes dont la valeur fonciere est inférieure à 1€ car ce n'est pas représentatif.
    # On pourra sans doute augmenter cette valeur autour de 1000€ ou plus
    filtre = df['Valeur fonciere'] <= 1
    df = df[~filtre]
    # Et on retire les lignes dont la surface reelle bati est nulle car seul le bati nous interesse
    df = df[df['Surface reelle bati'] != 0]

    # On va recréer l'index du dataframe pour qu'il soit propre
    df.reset_index(drop=True, inplace=True)
    
    return df


def save_dvf_df_to_csv(df: pd.DataFrame, output_path: str) -> None:
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
    df = load_dvf_file(filename)
    df = clean_dvf_data(df)
    # Retirer l'extension existante (si présente) avant d'ajouter .csv et le path
    file = os.path.basename(filename)
    base, _ = os.path.splitext(file)
    output_filename = os.path.join('data', 'prepared', base + ".csv")
    save_dvf_df_to_csv(df, output_filename)
