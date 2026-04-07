"""
Tests unitaires pour src/agentia/tool_bdd.py
Couvre : get_database_schema, execute_sql (avec base SQLite en mémoire)
"""
import os
import sys
import sqlite3
import tempfile
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp_server.tool_bdd import get_database_schema, execute_sql


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_temp_db(tmp_path):
    """Crée une base SQLite temporaire avec des données pour les tests."""
    db_path = str(tmp_path / "test_immo.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE Transactions (
            id_mutation INTEGER PRIMARY KEY,
            date_mutation TEXT,
            nature_mutation TEXT,
            valeur_fonciere REAL,
            adresse_nom_voie TEXT,
            surface_reelle_bati REAL,
            nombre_pieces_principales INTEGER,
            surface_terrain REAL,
            longitude REAL,
            latitude REAL,
            code_commune TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE Communes (
            code_commune TEXT PRIMARY KEY,
            nom_commune TEXT,
            code_postal TEXT,
            code_departement TEXT,
            densite REAL,
            superficie_km2 REAL,
            altitude_moyenne REAL
        )
    """)
    cursor.executemany(
        "INSERT INTO Transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "2024-01-15", "Vente", 250000.0, "RUE DE LA PAIX", 65.0, 3, 0.0, 2.3522, 48.8566, "75056"),
            (2, "2024-02-20", "Vente", 180000.0, "AVENUE DE PARIS", 45.0, 2, 0.0, 2.3200, 48.8700, "75056"),
        ]
    )
    cursor.executemany(
        "INSERT INTO Communes VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("75056", "Paris", "75001", "75", 20494.3, 105.4, 35.0),
        ]
    )
    conn.commit()
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# get_database_schema
# ---------------------------------------------------------------------------

class TestGetDatabaseSchema:
    def test_returns_dict(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = get_database_schema.invoke({})
        assert isinstance(result, dict)

    def test_contains_transactions_table(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = get_database_schema.invoke({})
        assert "Transactions" in result

    def test_contains_communes_table(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = get_database_schema.invoke({})
        assert "Communes" in result

    def test_transactions_has_expected_columns(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = get_database_schema.invoke({})
        cols = result.get("Transactions", [])
        assert "id_mutation" in cols
        assert "valeur_fonciere" in cols

    def test_communes_has_expected_columns(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = get_database_schema.invoke({})
        cols = result.get("Communes", [])
        assert "code_commune" in cols
        assert "nom_commune" in cols

    def test_empty_database_returns_empty_dict(self):
        empty_conn = sqlite3.connect(":memory:")
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=empty_conn):
            result = get_database_schema.invoke({})
        assert result == {}


# ---------------------------------------------------------------------------
# execute_sql
# ---------------------------------------------------------------------------

class TestExecuteSql:
    def test_blocks_drop_table(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "DROP TABLE Transactions"})
        assert result == "Only SELECT queries are allowed."

    def test_blocks_insert(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "INSERT INTO Transactions VALUES (99, '2024-01-01', 'Vente', 100000, 'test', 50, 2, 0, 0, 0, '75056')"})
        assert result == "Only SELECT queries are allowed."

    def test_blocks_update(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "UPDATE Transactions SET valeur_fonciere = 1"})
        assert result == "Only SELECT queries are allowed."

    def test_blocks_delete(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "DELETE FROM Transactions"})
        assert result == "Only SELECT queries are allowed."

    def test_select_returns_list(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "SELECT * FROM Transactions"})
        assert isinstance(result, list)

    def test_select_returns_correct_row_count(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "SELECT * FROM Transactions"})
        assert len(result) == 2

    def test_select_with_where_clause(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "SELECT * FROM Transactions WHERE valeur_fonciere > 200000"})
        assert len(result) == 1

    def test_select_count(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "SELECT COUNT(*) FROM Transactions"})
        assert result[0][0] == 2

    def test_invalid_sql_returns_error_string(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "SELECT * FROM table_inexistante"})
        assert isinstance(result, str)  # Message d'erreur

    def test_select_is_case_insensitive_check(self, tmp_path):
        """Vérifie que 'select' en minuscules est aussi autorisé."""
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "select * from Transactions"})
        assert isinstance(result, list)

    def test_select_communes(self, tmp_path):
        db_path = _create_temp_db(tmp_path)
        with patch("src.agentia.tool_bdd.sqlite3.connect", return_value=sqlite3.connect(db_path)):
            result = execute_sql.invoke({"query": "SELECT nom_commune FROM Communes"})
        assert len(result) == 1
        assert result[0][0] == "Paris"
