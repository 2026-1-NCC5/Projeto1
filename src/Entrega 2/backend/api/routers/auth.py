from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.core.config import settings
from api.dependencies import get_db
from api.schemas.auth import AdminCadastro, AdminLoginSemSenha, AdminMe
from api.services import auth_service
from bd.models.usuario import Usuario

router = APIRouter()


def _set_session_cookie(response: Response, raw_token: str) -> None:
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=raw_token,
        httponly=True,
        samesite="lax",
        secure=settings.AUTH_COOKIE_SECURE,
        max_age=settings.AUTH_SESSION_IDLE_MINUTES * 60 * 2,
        path="/",
    )


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(settings.AUTH_COOKIE_NAME, path="/")


@router.post("/cadastro-admin", response_model=AdminMe)
def cadastro_admin(
    body: AdminCadastro,
    response: Response,
    db: Session = Depends(get_db),
):
    email_norm = auth_service.normalizar_email(body.email)
    if not auth_service.email_institucional_valido(email_norm):
        raise HTTPException(
            status_code=400,
            detail="Use um e-mail institucional permitido.",
        )
    existe = (
        db.query(Usuario)
        .filter(func.lower(Usuario.email) == email_norm)
        .first()
    )
    if existe:
        raise HTTPException(status_code=409, detail="E-mail já cadastrado")

    nome_limpo = body.nome.strip()
    if not nome_limpo:
        raise HTTPException(status_code=400, detail="Nome inválido")

    user = Usuario(
        nome=nome_limpo[:100],
        email=email_norm,
        perfil="admin",
        ativo=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    raw, _ = auth_service.criar_sessao_auth(db, user.id)
    _set_session_cookie(response, raw)
    return user


@router.post("/login", response_model=AdminMe)
def login_admin(
    body: AdminLoginSemSenha,
    response: Response,
    db: Session = Depends(get_db),
):
    email_norm = auth_service.normalizar_email(body.email)
    if not auth_service.email_institucional_valido(email_norm):
        raise HTTPException(
            status_code=400,
            detail="Use um e-mail institucional permitido.",
        )
    user = auth_service.buscar_usuario_login(db, body.nome, email_norm)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Nome ou e-mail não conferem com um cadastro de administrador.",
        )

    raw, _ = auth_service.criar_sessao_auth(db, user.id)
    _set_session_cookie(response, raw)
    return user


@router.post("/logout")
def logout_admin(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    token = request.cookies.get(settings.AUTH_COOKIE_NAME)
    auth_service.revogar_sessao_por_token(db, token)
    _clear_session_cookie(response)
    return {"ok": True}


@router.get("/me", response_model=AdminMe)
def me_admin(
    request: Request,
    db: Session = Depends(get_db),
):
    token = request.cookies.get(settings.AUTH_COOKIE_NAME)
    user = auth_service.obter_usuario_por_token(db, token, renovar_acesso=True)
    if user is None:
        raise HTTPException(status_code=401, detail="Sessão inválida ou expirada")
    return user
