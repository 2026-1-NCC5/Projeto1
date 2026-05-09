from pydantic import BaseModel, Field


class AdminCadastro(BaseModel):
    nome: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=255)


class AdminLoginSemSenha(BaseModel):
    nome: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=3, max_length=255)


class AdminMe(BaseModel):
    id: int
    nome: str
    email: str
    perfil: str
    ativo: bool

    class Config:
        from_attributes = True
