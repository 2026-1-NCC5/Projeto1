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
from api.routers import auth, deteccoes, sessoes, relatorios

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(sessoes.router, prefix=f"{settings.API_V1_STR}/sessoes", tags=["sessoes"])
app.include_router(deteccoes.router, prefix=f"{settings.API_V1_STR}/deteccoes", tags=["deteccoes"])
app.include_router(relatorios.router, prefix=f"{settings.API_V1_STR}/relatorios", tags=["relatorios"])

@app.get("/")
def root():
    return {"message": "Bem-vindo à API AbraceAI"}
