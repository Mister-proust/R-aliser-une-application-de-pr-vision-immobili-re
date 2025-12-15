import uvicorn
from fastapi import FastAPI
from src.api.routers import auth, users
from src.core.config import settings

# Crée l'instance principale de l'application FastAPI
app = FastAPI(
    title="FastAPI Clean Architecture Template",
    description="Un template pour démarrer des projets FastAPI avec la Clean Architecture.",
    version="0.1.0"
)

# --- Inclusion des Routeurs ---
# Les routeurs définis dans le module `src.api.routers` sont inclus ici.
# Chaque routeur gère une partie spécifique de l'API.

# Routeur pour l'authentification (login)
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])

# Routeur pour les opérations sur les utilisateurs (enregistrement, infos 'me')
app.include_router(users.router, prefix="/users", tags=["Users"])

# --- Route de Santé (Health Check) ---
# Une route simple pour vérifier que l'API est en ligne.
@app.get("/", tags=["Health Check"])
def read_root():
    """
    Route racine pour une vérification de santé de base.
    """
    return {"status": "ok", "message": f"Welcome to {settings.PROJECT_NAME}!"}


# Point d'entrée pour l'exécution directe du serveur (pour le débogage local)
if __name__ == "__main__":
    # Note : Normalement, vous lancez avec `uvicorn src.main:app --reload`
    # Ceci est juste pour un débogage simple.
    uvicorn.run(app, host="127.0.0.1", port=8000)