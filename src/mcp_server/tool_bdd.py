import os
import sqlite3
import logging
from .instance import mcp

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "agentia", "bdd", "donnees_immo.db")

@mcp.tool()
def get_database_schema():
    """
    Retrieves the SQLite database structure.
    Returns a dictionary with the table names and their columns.
    :return: dict
    """
    if not os.path.exists(DB_PATH):
        return {"error": f"Base de données introuvable à : {DB_PATH}"}
        
    connection = sqlite3.connect(DB_PATH)
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

@mcp.tool()
def execute_sql(query: str):
    """
    Executes a SQL SELECT query against the SQLite database.
    Only SELECT queries are allowed for security reasons.
    :param query: str - The SQL query to execute.
    :return: list - Query results or an error message.
    """
    if not os.path.exists(DB_PATH):
        return [f"Error: Base de données introuvable à : {DB_PATH}"]

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    if not query.lower().startswith("select"):
        return "Only SELECT queries are allowed."
    try:
        cursor.execute(query)
        result = cursor.fetchall()
    except Exception as e:
        result = [str(e)]

    connection.close()
    return result
