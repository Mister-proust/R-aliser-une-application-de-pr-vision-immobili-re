import os
import sqlite3
import logging
import re
from typing import Optional, Dict, Any
from .instance import mcp

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "agentia", "bdd", "donnees_immo.db")

# ==================== Validation Functions ====================

def validate_postal_code(postal_code: str) -> bool:
    """Validate French postal code format (5 digits)."""
    return bool(re.match(r"^\d{5}$", str(postal_code).strip()))

def validate_commune_name(commune_name: str) -> bool:
    """Validate commune name (alphanumeric, spaces, hyphens, accents)."""
    return bool(re.match(r"^[a-zA-Zàâäçèéêëîïôùûüœæ\s\-']{1,100}$", str(commune_name).strip()))

def calculate_median(values: list) -> Optional[float]:
    """Calculate median from a list of numeric values."""
    if not values:
        return None
    sorted_vals = sorted([v for v in values if v is not None])
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    return sorted_vals[n // 2]

# ==================== Quartier Diagnostic Tool ====================

@mcp.tool()
def diagnostic_quartier(
    code_insee: Optional[str] = None,
    code_postal: Optional[str] = None,
    commune_name: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
) -> str:
    """
    Get local neighborhood statistics for a French area based on DVF transaction data.
    
    Provides: transaction count, average price, median price, average rooms, 
    property type distribution, and price trend (1st vs 2nd half 2025).
    
    Parameters:
    - code_insee: INSEE code (5 digits), e.g., '75056' for Paris
    - code_postal: French postal code (5 digits) - will be converted to INSEE code, e.g., '75001'
    - commune_name: Municipality name, e.g., 'Paris'
    - latitude, longitude: Geographic coordinates (returns closest municipality stats)
    
    Returns: Formatted neighborhood diagnostics or error message.
    """
    
    # === Validation ===
    if not any([code_insee, code_postal, commune_name, (latitude is not None and longitude is not None)]):
        return "Erreur: Veuillez fournir soit un code INSEE, un code postal, un nom de commune, soit des coordonnées (latitude, longitude)."
    
    # Validate formats
    if code_insee and not validate_postal_code(code_insee):
        return f"Erreur: Code INSEE invalide '{code_insee}'. Format attendu: 5 chiffres (ex: 75056)."
    
    if code_postal and not validate_postal_code(code_postal):
        return f"Erreur: Code postal invalide '{code_postal}'. Format attendu: 5 chiffres (ex: 75001)."
    
    if commune_name and not validate_commune_name(commune_name):
        return f"Erreur: Nom de commune invalide '{commune_name}'. Caractères autorisés: lettres, espaces, tirets, accents."
    
    if not os.path.exists(DB_PATH):
        return f"Erreur: Base de données introuvable à {DB_PATH}"
    
    try:
        connection = sqlite3.connect(DB_PATH)
        connection.row_factory = sqlite3.Row
        cursor = connection.cursor()
        
        # === Build Query ===
        # Get code_commune from INSEE code, postal code, or commune name
        code_commune = None
        
        if code_insee:
            # Code INSEE is directly the code_commune
            code_commune = code_insee
            # Verify the code exists in database
            cursor.execute(
                """
                SELECT code_commune FROM Communes 
                WHERE code_commune = ?
                LIMIT 1
                """,
                (code_insee,)
            )
            result = cursor.fetchone()
            if not result:
                connection.close()
                return f"Aucune commune trouvée avec le code INSEE {code_insee}. Vérifiez que le code est correct et que la région est couverte par la base de données."
        
        elif code_postal:
            # Convert postal code to INSEE code
            cursor.execute(
                """
                SELECT code_commune FROM Communes 
                WHERE code_postal = ?
                LIMIT 1
                """,
                (code_postal,)
            )
            result = cursor.fetchone()
            if result:
                code_commune = result[0]
            else:
                connection.close()
                return f"Aucune commune trouvée avec le code postal {code_postal}. Vérifiez que le code est correct et que la région est couverte par la base de données."
        
        elif commune_name:
            cursor.execute(
                """
                SELECT code_commune FROM Communes 
                WHERE LOWER(nom_commune) LIKE LOWER(?)
                LIMIT 1
                """,
                (f"%{commune_name.strip()}%",)
            )
            result = cursor.fetchone()
            if result:
                code_commune = result[0]
            else:
                connection.close()
                return f"Aucune commune trouvée portant le nom '{commune_name}'. Vérifiez l'orthographe."
        
        elif latitude is not None and longitude is not None:
            # Find closest commune by coordinates (simplified: find by closest transaction)
            cursor.execute(
                """
                SELECT code_commune, 
                       (latitude - ?) * (latitude - ?) + (longitude - ?) * (longitude - ?) as distance
                FROM Transactions
                ORDER BY distance ASC
                LIMIT 1
                """,
                (latitude, latitude, longitude, longitude)
            )
            result = cursor.fetchone()
            if result:
                code_commune = result[0]
            else:
                connection.close()
                return f"Aucune transaction trouvée près des coordonnées ({latitude}, {longitude})."
        
        if not code_commune:
            connection.close()
            return "Impossible de déterminer la commune. Vérifiez vos paramètres."
        
        # === Get Transactions Data ===
        cursor.execute(
            """
            SELECT 
                valeur_fonciere,
                nombre_pieces_principales,
                nature_mutation,
                date_mutation
            FROM Transactions
            WHERE code_commune = ?
            AND valeur_fonciere > 0
            AND nombre_pieces_principales > 0
            """,
            (code_commune,)
        )
        transactions = cursor.fetchall()
        
        if not transactions:
            connection.close()
            return f"Aucune transaction immobilière trouvée pour cette commune dans la base de données (période: janvier-juin 2025)."
        
        # === Calculate Statistics ===
        prices = []
        rooms = []
        property_types = {}
        early_period_prices = []  # Jan-Mar 2025
        late_period_prices = []   # Apr-Jun 2025
        
        for tx in transactions:
            price = tx[0]
            room_count = tx[1]
            nature = tx[2]
            date_str = tx[3]
            
            prices.append(price)
            rooms.append(room_count)
            
            # Count property types
            if nature not in property_types:
                property_types[nature] = 0
            property_types[nature] += 1
            
            # Split by period (rough split: before April 15)
            try:
                date_parts = date_str.split("-") if date_str else ["2025", "01", "01"]
                month = int(date_parts[1]) if len(date_parts) > 1 else 1
                if month <= 3:
                    early_period_prices.append(price)
                else:
                    late_period_prices.append(price)
            except:
                early_period_prices.append(price)
        
        # Calculate metrics
        nb_transactions = len(prices)
        avg_price = sum(prices) / len(prices) if prices else 0
        median_price = calculate_median(prices)
        avg_rooms = sum(rooms) / len(rooms) if rooms else 0
        
        # Determine property type distribution
        main_type = max(property_types, key=property_types.get) if property_types else "Inconnu"
        type_dist = ", ".join([f"{k}: {v}" for k, v in sorted(property_types.items(), key=lambda x: x[1], reverse=True)])
        
        # Calculate trend
        early_avg = sum(early_period_prices) / len(early_period_prices) if early_period_prices else 0
        late_avg = sum(late_period_prices) / len(late_period_prices) if late_period_prices else 0
        
        if early_avg > 0 and late_avg > 0:
            trend_percent = ((late_avg - early_avg) / early_avg) * 100
            if trend_percent > 2:
                trend = f"📈 Hausse (+{trend_percent:.1f}%)"
            elif trend_percent < -2:
                trend = f"📉 Baisse ({trend_percent:.1f}%)"
            else:
                trend = f"➡️ Stable ({trend_percent:+.1f}%)"
        else:
            trend = "➡️ Données insuffisantes pour tendance"
        
        # === Get Commune Info ===
        cursor.execute(
            """
            SELECT nom_commune, code_postal, densite, superficie_km2
            FROM Communes
            WHERE code_commune = ?
            """,
            (code_commune,)
        )
        commune_info = cursor.fetchone()
        commune_name_result = commune_info[0] if commune_info else "Inconnue"
        postal_code_result = commune_info[1] if commune_info else "N/A"
        densite = commune_info[2] if commune_info else 0
        
        connection.close()
        
        # === Format Output ===
        result = f"""
🏘️ **DIAGNOSTIC DE QUARTIER: {commune_name_result}** ({postal_code_result})

📊 **STATISTIQUES TRANSACTIONS** (Période: Jan-Juin 2025)
├─ Nombre de transactions: {nb_transactions}
├─ Prix moyen: {avg_price:,.0f} €
├─ Prix médian: {median_price:,.0f} € (50% des transactions)
├─ Nombre de pièces moyen: {avg_rooms:.1f}
└─ Types de bien: {type_dist}

💰 **TENDANCE MARCHÉ**
└─ {trend}

🌍 **CARACTÉRISTIQUES GÉOGRAPHIQUES**
├─ Densité: {densite:,.0f} hab/km²
└─ Code commune INSEE: {code_commune}

ℹ️ Données basées sur les demandes de valeurs foncières (DVF) officielles.
        """
        
        return result.strip()
    
    except Exception as e:
        logger.error(f"Erreur diagnostic_quartier: {e}")
        return f"Erreur lors du diagnostic: {str(e)}"