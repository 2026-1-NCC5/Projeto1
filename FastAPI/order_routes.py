from fastapi import APIRouter 

order_router = APIRouter(prefix="/orders", tags=["orders"])
#cria um roteador para as rotas de pedidos, com o prefixo "/orders" e a tag "orders" para organização.

@order_router.get("/")
async def contagem_alimentos():
    return {"message": "Contagem de alimentos"}
#cria uma rota GET para a raiz do roteador de pedidos, que retorna uma mensagem indicando que é a rota de contagem de alimentos.