from fastapi import FastAPI

app = FastAPI()
from auth_routes import auth_router
from order_routes import order_router
#importa os roteadores de autenticação e pedidos para serem incluídos na aplicação FastAPI.

app.include_router(auth_router)
app.include_router(order_router)
#coloca os roteadores na aplicação FastAPI para que as rotas definidas neles sejam acessíveis.

#.venv\Scripts\activate ativa o ambiente virtual
#uvicorn main:app --reload inicia o servidor
#pip install fastapi uvicorn sqlalchemy passlib[bcrypt] python-jose[cryptography] python-dotenv python-multipart bibliotecas
#alembic revision --autogenerate -m "migracao inicial" para toda alteração no BD 
#alembic upgrade head para subir as alterações