# tester si la base de données est correctement configurée et accessible
import sqlite3
# Se connecter à la base de données
conn = sqlite3.connect("donnees_immo.db")
cursor = conn.cursor()
# Exécuter une requête pour vérifier que la table "Transactions" existe et contient des données
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Transactions';")
table_exists = cursor.fetchone()
if table_exists:
    print("La table 'Transactions' existe dans la base de données.")
else:
    print("La table 'Transactions' n'existe pas dans la base de données.")

# Afficher les 5 premières lignes de la table "Transactions"
cursor.execute("SELECT * FROM Transactions LIMIT 5;")
rows = cursor.fetchall()
print("Voici les 5 premières lignes de la table 'Transactions':")
for row in rows:
    print(row)

# Exécuter une requête pour vérifier que la table "Communes" existe et contient des données
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Communes';")
table_exists = cursor.fetchone()
if table_exists:
    print("La table 'Communes' existe dans la base de données.")
else:
    print("La table 'Communes' n'existe pas dans la base de données.")

# Afficher les 5 premières lignes de la table "Communes"
cursor.execute("SELECT * FROM Communes LIMIT 5;")
rows = cursor.fetchall()
print("Voici les 5 premières lignes de la table 'Communes':")
for row in rows:
    print(row)


# Fermer la connexion à la base de données
conn.close()
