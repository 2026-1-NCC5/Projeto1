from fastapi import APIRouter 

auth_router = APIRouter(prefix="/auth", tags=["auth"])
#cria um roteador para as rotas de autenticação, com o prefixo "/auth" e a tag "auth" para organização.

@auth_router.get("/")
async def login():
    """
    Rota de login padrão
    """
    return {"message": "Rota de login", "autenticado": False}
#cria uma rota GET para a raiz do roteador de autenticação, que retorna uma mensagem indicando que é a rota de login