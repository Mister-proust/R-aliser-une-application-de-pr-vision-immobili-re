# transformer un csv en une base de donnée sql lite
import pandas as pd
import sqlite3

# todo : Filtrer les colonnes que l'on souhaite conserver !

# Charger le fichier CSV
csv_file = "data/clean_dvf.csv"
df = pd.read_csv(csv_file)

# Créer une connexion à la base de données SQLite
db_file = "data/clean_dvf.db"
conn = sqlite3.connect(db_file)

# Exporter le DataFrame vers la base de données SQLite
df.to_sql("clean_dvf", conn, if_exists="replace", index=False)

# Fermer la connexion
conn.close()
