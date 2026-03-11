# transformer un csv en une base de donnée sql lite
import pandas as pd
import sqlite3

# todo : Filtrer les colonnes que l'on souhaite conserver !

connection = sqlite3.connect("donnees_immo.db")

# Charger le fichier CSV dans un DataFrame pandas
df = pd.read_csv("../data/clean_dvf.csv", sep=";")

print(df.columns.tolist())
# Garder les colonnes suivantes : "id_transaction, Date mutation, Nature mutation, Valeur fonciere, Voie, Surface reelle bati, Nombre pieces principales, Surface terrain, longitude_centre et latitude_centre avec clé primaire id_transaction
df = df[
    [
        "id_transaction",
        "Date mutation",
        "Nature mutation",
        "Valeur fonciere",
        "Voie",
        "code_insee",
        "Surface reelle bati",
        "Nombre pieces principales",
        "Surface terrain",
        "longitude_centre",
        "latitude_centre",
    ]
]
print(df.head())
print(df.columns.tolist())

# Renommer les colonnes pour qu'elles soient compatibles avec les noms de colonnes SQL. id_transaction devient id_mutation,
# Date mutation devient date_mutation, Nature mutation devient nature_mutation, Valeur fonciere devient valeur_fonciere, 
# Voie devient adresse_nom_voie, Surface reelle bati devient surface_reelle_bati, Nombre pieces principales devient 
# nombre_pieces_principales, Surface terrain devient surface_terrain, longitude_centre devient longitude et 
# latitude_centre devient latitude, Code commune devient code_commune
df = df.rename(
    columns={
        "id_transaction": "id_mutation",
        "Date mutation": "date_mutation",
        "Nature mutation": "nature_mutation",
        "Valeur fonciere": "valeur_fonciere",
        "Voie": "adresse_nom_voie",
        "Surface reelle bati": "surface_reelle_bati",
        "Nombre pieces principales": "nombre_pieces_principales",
        "Surface terrain": "surface_terrain",
        "longitude_centre": "longitude",
        "latitude_centre": "latitude",
        "code_insee": "code_commune",
    }
)

# Enregistrer le DataFrame dans une base de données SQLite
df.to_sql("Transactions", "sqlite:///donnees_immo.db", if_exists="replace", index=False)
print("Base de données créée avec succès !")

# Charger le fichier CSV dans un DataFrame pandas
df_communes = pd.read_csv("../data/communes-france-2025.csv", sep=",")
print(df_communes.head())
print(df_communes.columns.tolist())

# Garder les colonnes suivantes : "code_insee, nom_standard, code_postal, dep_code, niveau_equipements_services, densite, superficie_km2, altitude_moyenne, type_commune_unite_urbaine"
df_communes = df_communes[
    [
        "code_insee",
        "nom_standard",
        "code_postal",
        "dep_code",
        "niveau_equipements_services",
        "densite",
        "superficie_km2",
        "altitude_moyenne",
        "type_commune_unite_urbaine",
    ]
]

# Renommer les colonnes : code_insee devient code_commune, nom_standard devient nom_commune, code_postal devient code_postal, dep_code devient code_departement, niveau_equipements_services devient niveau_equipements_services, densite devient densite, superficie_km2 devient superficie_km2, altitude_moyenne devient altitude_moyenne et type_commune_unite_urbaine devient type_commune_unite_urbaine
df_communes = df_communes.rename(
    columns={
        "code_insee": "code_commune",
        "nom_standard": "nom_commune",
        "code_postal": "code_postal",
        "dep_code": "code_departement",
        "niveau_equipements_services": "niveau_equipements_services",
        "densite": "densite",
        "superficie_km2": "superficie_km2",
        "altitude_moyenne": "altitude_moyenne",
        "type_commune_unite_urbaine": "type_commune_unite_urbaine",
    }
)

# Enregistrer le DataFrame dans une base de données SQLite dans une table "Communes"
df_communes.to_sql("Communes", "sqlite:///donnees_immo.db", if_exists="replace", index=False)
print("Base de données créée avec succès !")