import sqlite3
from langchain_core.tools import tool
import logging

@tool
def get_database_schema():
    """
    Récupère la structure de la base de données SQLite.
    Retourne un dictionnaire avec les noms des tables et leurs colonnes.
    :return: dict
    """
    connection = sqlite3.connect("bdd/donnees_immo.db")
    cursor = connection.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    schema = {}

    for table in tables:
        table_name = table[0]

        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()

        schema[table_name] = [column[1] for column in columns]

    connection.close()

    return schema

@tool
def execute_sql(query: str):
    """
    Exécute une requête SQL SELECT sur la base de données SQLite.
    Seules les requêtes SELECT sont autorisées pour des raisons de sécurité.
    :param query: str - La requête SQL à exécuter.
    :return: list - Les résultats de la requête ou un message d'erreur.
    """
    connection = sqlite3.connect("bdd/donnees_immo.db")
    cursor = connection.cursor()
    if not query.lower().startswith("select"):
        return "Only SELECT queries are allowed."
    try:
        cursor.execute(query)
        result = cursor.fetchall()
    except Exception as e:
        result = str(e)

    connection.close()
    return result