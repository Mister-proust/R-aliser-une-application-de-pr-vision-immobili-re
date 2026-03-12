"""
Tests unitaires pour scripts/clean_dvf.py
Couvre : load_dvf_file, load_communes_file, clean_dvf_data, save_dvf__df_to_csv
"""
import os
import sys
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from scripts.clean_dvf import (
    load_dvf_file,
    load_communes_file,
    clean_dvf_data,
    save_dvf__df_to_csv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dvf_df(**overrides):
    """Construit un DataFrame DVF minimal valide et nettoyable."""
    base = {
        "Date mutation": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10"]),
        "Nature mutation": ["Vente", "Vente", "Vente"],
        "Valeur fonciere": pd.array([250000.0, 180000.0, 320000.0], dtype="float32"),
        "No voie": ["12", "5", "8"],
        "B/T/Q": [None, "B", None],
        "Type de voie": ["RUE", "AV", "IMP"],
        "Code voie": ["0001", "0002", "0003"],
        "Voie": ["RUE DE LA PAIX", "AVENUE DE PARIS", "IMPASSE DES FLEURS"],
        "Code postal": ["75001", "75002", "75003"],
        "Commune": ["PARIS", "PARIS", "PARIS"],
        "Code departement": ["75", "75", "75"],
        "Code commune": ["056", "056", "057"],
        "Prefixe de section": [None, None, None],
        "Section": ["AB", "CD", "EF"],
        "No plan": pd.array([1, 2, 3], dtype="Int16"),
        "Code type local": pd.array([2, 1, 2], dtype="Int8"),
        "Type local": ["Appartement", "Maison", "Appartement"],
        "Surface reelle bati": pd.array([65, 120, 45], dtype="Int32"),
        "Nombre pieces principales": pd.array([3, 5, 2], dtype="Int8"),
        "Surface terrain": pd.array([0, 500, 0], dtype="Int32"),
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _make_communes_df():
    """Construit un DataFrame communes minimal."""
    return pd.DataFrame({
        "code_insee": ["75056", "75057"],
        "nom_standard_majuscule": ["PARIS", "PARIS 17E"],
        "population": [2161000, 170000],
        "superficie_km2": [105.4, 5.6],
        "densite": [20494.3, 30357.1],
        "latitude_centre": [48.8566, 48.8850],
        "longitude_centre": [2.3522, 2.3086],
    })


# ---------------------------------------------------------------------------
# load_dvf_file
# ---------------------------------------------------------------------------

class TestLoadDvfFile:
    def test_raises_when_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            load_dvf_file("/chemin/qui/nexiste/pas.txt")

    def test_raises_with_meaningful_message(self):
        bad_path = "/tmp/fichier_absent.txt"
        with pytest.raises(FileNotFoundError, match=bad_path):
            load_dvf_file(bad_path)

    def test_returns_dataframe_when_file_exists(self, tmp_path):
        # Crée un fichier DVF minimal au bon format
        content = (
            "Date mutation|Nature mutation|Valeur fonciere|No voie|B/T/Q|Type de voie|"
            "Code voie|Voie|Code postal|Commune|Code departement|Code commune|"
            "Prefixe de section|Section|No plan|Code type local|Type local|"
            "Surface reelle bati|Nombre pieces principales|Surface terrain\n"
            "15/01/2024|Vente|250000|12||RUE|0001|RUE DE LA PAIX|75001|PARIS|75|056||AB|1|2|Appartement|65|3|0\n"
        )
        dvf_file = tmp_path / "dvf_test.txt"
        dvf_file.write_text(content, encoding="utf-8")
        df = load_dvf_file(str(dvf_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "Valeur fonciere" in df.columns

    def test_dataframe_has_expected_columns(self, tmp_path):
        content = (
            "Date mutation|Nature mutation|Valeur fonciere|No voie|B/T/Q|Type de voie|"
            "Code voie|Voie|Code postal|Commune|Code departement|Code commune|"
            "Prefixe de section|Section|No plan|Code type local|Type local|"
            "Surface reelle bati|Nombre pieces principales|Surface terrain\n"
            "15/01/2024|Vente|250000|12||RUE|0001|RUE DE LA PAIX|75001|PARIS|75|056||AB|1|2|Appartement|65|3|0\n"
        )
        dvf_file = tmp_path / "dvf_test.txt"
        dvf_file.write_text(content, encoding="utf-8")
        df = load_dvf_file(str(dvf_file))
        expected_cols = ["Date mutation", "Nature mutation", "Valeur fonciere", "Type local"]
        for col in expected_cols:
            assert col in df.columns, f"Colonne manquante: {col}"


# ---------------------------------------------------------------------------
# load_communes_file
# ---------------------------------------------------------------------------

class TestLoadCommunesFile:
    def test_raises_when_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError):
            load_communes_file("/chemin/inexistant/communes.csv")

    def test_returns_dataframe_when_file_exists(self, tmp_path):
        content = "code_insee,nom_standard,population\n75056,Paris,2161000\n"
        communes_file = tmp_path / "communes_test.csv"
        communes_file.write_text(content, encoding="utf-8")
        df = load_communes_file(str(communes_file))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "code_insee" in df.columns

    def test_dataframe_has_correct_data(self, tmp_path):
        content = "code_insee,nom_standard,population\n75056,Paris,2161000\n"
        communes_file = tmp_path / "communes_test.csv"
        communes_file.write_text(content, encoding="utf-8")
        df = load_communes_file(str(communes_file))
        # pandas peut lire code_insee comme int64 si la colonne ne contient que des chiffres
        assert str(df["code_insee"].iloc[0]) == "75056"
        assert df["nom_standard"].iloc[0] == "Paris"


# ---------------------------------------------------------------------------
# clean_dvf_data
# ---------------------------------------------------------------------------

class TestCleanDvfData:
    def test_removes_echange_mutations(self):
        df = _make_dvf_df(
            **{
                "Nature mutation": ["Vente", "Echange", "Vente"],
                "Date mutation": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-10"]),
                "Code commune": ["056", "056", "057"],
                "Code voie": ["0001", "0002", "0003"],
            }
        )
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "Echange" not in result["Nature mutation"].values

    def test_removes_rows_with_null_valeur_fonciere(self):
        df = _make_dvf_df()
        df.loc[0, "Valeur fonciere"] = None
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert result["Valeur fonciere"].notna().all()

    def test_valeur_fonciere_converted_to_int(self):
        df = _make_dvf_df()
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert result["Valeur fonciere"].dtype in [np.int32, "int32", "Int32", int]

    def test_fills_surface_reelle_bati_nan_with_zero(self):
        df = _make_dvf_df()
        df.loc[0, "Surface reelle bati"] = pd.NA
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert result["Surface reelle bati"].isna().sum() == 0

    def test_fills_surface_terrain_nan_with_zero(self):
        df = _make_dvf_df()
        df.loc[0, "Surface terrain"] = pd.NA
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert result["Surface terrain"].isna().sum() == 0

    def test_fills_nombre_pieces_nan_with_zero(self):
        df = _make_dvf_df()
        df.loc[0, "Nombre pieces principales"] = pd.NA
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert result["Nombre pieces principales"].isna().sum() == 0

    def test_fills_btq_nan_with_empty_string(self):
        df = _make_dvf_df()
        df.loc[0, "B/T/Q"] = None  # Forcer une NaN
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert result["B/T/Q"].isna().sum() == 0

    def test_drops_duplicates(self):
        df = _make_dvf_df()
        df_with_dup = pd.concat([df, df], ignore_index=True)
        communes = _make_communes_df()
        result_single = clean_dvf_data(df, communes)
        result_dup = clean_dvf_data(df_with_dup, communes)
        assert len(result_single) == len(result_dup)

    def test_maps_type_voie_rue(self):
        df = _make_dvf_df(**{"Type de voie": ["RUE", "RUE", "RUE"]})
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "Rue" in result["Type de voie"].values

    def test_maps_type_voie_avenue(self):
        df = _make_dvf_df(**{"Type de voie": ["AV", "AV", "AV"]})
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "Avenue" in result["Type de voie"].values

    def test_maps_type_voie_impasse(self):
        df = _make_dvf_df(**{"Type de voie": ["IMP", "IMP", "IMP"]})
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "Impasse" in result["Type de voie"].values

    def test_unknown_type_voie_stays_as_is(self):
        """Les valeurs non-null qui ne sont pas dans le mapping restent telles quelles.
        Seules les valeurs NaN sont remplacées par 'Autre'."""
        df = _make_dvf_df(**{"Type de voie": ["ZZUNK", "ZZUNK", "ZZUNK"]})
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        # Les valeurs inconnues ne sont PAS remplacées par 'Autre' — seulement les NaN le sont
        assert "ZZUNK" in result["Type de voie"].values

    def test_maps_type_local_appartement(self):
        df = _make_dvf_df(**{"Type local": ["Appartement", "Appartement", "Appartement"]})
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "Appartement" in result["Type local"].values

    def test_maps_type_local_maison(self):
        df = _make_dvf_df(**{"Type local": ["Maison", "Maison", "Maison"]})
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "Maison" in result["Type local"].values

    def test_removes_rows_with_dependance_type_local(self):
        df = _make_dvf_df(**{"Type local": ["Appartemenet", "Dépendance", "Maison"]})
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "Dépendance" not in result["Type local"].values

    def test_removes_rows_with_zero_surface_bati(self):
        df = _make_dvf_df()
        df.loc[0, "Surface reelle bati"] = 0
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert (result["Surface reelle bati"] == 0).sum() == 0

    def test_removes_valeur_fonciere_below_or_equal_1(self):
        df = _make_dvf_df(**{"Valeur fonciere": pd.array([1.0, 180000.0, 320000.0], dtype="float32")})
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert (result["Valeur fonciere"] <= 1).sum() == 0

    def test_adds_id_transaction_column(self):
        df = _make_dvf_df()
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "id_transaction" in result.columns

    def test_adds_code_insee_column(self):
        df = _make_dvf_df()
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "code_insee" in result.columns

    def test_merges_with_communes_adds_densite(self):
        df = _make_dvf_df()
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "densite" in result.columns

    def test_merges_with_communes_adds_coordinates(self):
        df = _make_dvf_df()
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert "latitude_centre" in result.columns
        assert "longitude_centre" in result.columns

    def test_drops_rows_without_commune_match(self):
        df = _make_dvf_df(**{"Code departement": ["99", "99", "99"], "Code commune": ["999", "999", "999"]})
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        # Aucune commune ne correspond → toutes les lignes sont éliminées (densite NaN → drop)
        assert len(result) == 0

    def test_returns_dataframe(self):
        df = _make_dvf_df()
        communes = _make_communes_df()
        result = clean_dvf_data(df, communes)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# save_dvf__df_to_csv
# ---------------------------------------------------------------------------

class TestSaveDvfDfToCsv:
    def test_creates_file(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        output = str(tmp_path / "output.csv")
        save_dvf__df_to_csv(df, output)
        assert os.path.exists(output)

    def test_saved_file_is_semicolon_separated(self, tmp_path):
        df = pd.DataFrame({"col1": [10], "col2": [20]})
        output = str(tmp_path / "output.csv")
        save_dvf__df_to_csv(df, output)
        with open(output, "r") as f:
            first_line = f.readline()
        assert ";" in first_line

    def test_saved_file_contains_data(self, tmp_path):
        df = pd.DataFrame({"valeur": [123456]})
        output = str(tmp_path / "output.csv")
        save_dvf__df_to_csv(df, output)
        df_loaded = pd.read_csv(output, sep=";")
        assert df_loaded["valeur"].iloc[0] == 123456

    def test_roundtrip_save_and_load(self, tmp_path):
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        output = str(tmp_path / "roundtrip.csv")
        save_dvf__df_to_csv(df, output)
        df_loaded = pd.read_csv(output, sep=";")
        assert list(df.columns) == list(df_loaded.columns)
        assert len(df) == len(df_loaded)
