import json
import os
import time
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import sys
from contextlib import asynccontextmanager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.routers.endpoints import router as api_router
import app.config as config
from prometheus_fastapi_instrumentator import Instrumentator
from app.state import init_state

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🔹 Startup
    init_state()
    print("Modèle ML + SHAP chargés en mémoire")
    yield
    # 🔹 Shutdown
    print("Arrêt de l'application")

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[],
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)  # Include the router

Instrumentator().instrument(app).expose(app, endpoint="/metrics")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app", 
        host=config.HOST, 
        port=config.PORT, 
        reload=config.RELOAD
    )