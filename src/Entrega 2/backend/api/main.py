import os
import sys

# Add backend directory to sys.path so bd modules can be imported
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from api.core.config import settings

from bd.models.usuario import Usuario
from bd.models.grupo import Grupo
from bd.models.aluno import Aluno
from bd.models.grupo_alimento import GrupoAlimento
from bd.models.alimento import Alimento
from bd.models.sessao import Sessao
from bd.models.item_declarado import ItemDeclarado
from bd.models.deteccao import Deteccao

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.routers import (
    auth,
    deteccoes,
    sessoes,
    relatorios,
    grupos,
    alimentos,
    itens_declarados,
    ws_auditoria,
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS: cookie exige origins explícitas (evitar "*" com credenciais).
_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(grupos.router, prefix=f"{settings.API_V1_STR}/grupos", tags=["grupos"])
app.include_router(alimentos.router, prefix=f"{settings.API_V1_STR}/alimentos", tags=["alimentos"])
app.include_router(itens_declarados.router, prefix=f"{settings.API_V1_STR}/itens-declarados", tags=["itens-declarados"])
app.include_router(sessoes.router, prefix=f"{settings.API_V1_STR}/sessoes", tags=["sessoes"])
app.include_router(deteccoes.router, prefix=f"{settings.API_V1_STR}/deteccoes", tags=["deteccoes"])
app.include_router(relatorios.router, prefix=f"{settings.API_V1_STR}/relatorios", tags=["relatorios"])
# WebSocket sob /ws (sem prefixo /api/v1, padrão do plano)
app.include_router(ws_auditoria.router, prefix="/ws", tags=["ws-auditoria"])

# Servir evidências (JPEGs gravados pelo evidencia_service) sob /evidencias/...
os.makedirs(settings.EVIDENCIA_DIR, exist_ok=True)
app.mount(
    "/evidencias",
    StaticFiles(directory=settings.EVIDENCIA_DIR),
    name="evidencias",
)

@app.get("/")
def root():
    return {"message": "Bem-vindo à API AbraceAI"}
