import os
from pydantic_settings import BaseSettings

# BASE_DIR is backend/api/core -> backend/api -> backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "bd", "abraceai.db")

# Caminho default para o modelo YOLO (relativo ao backend/)
YOLO_DEFAULT_MODEL = os.path.join(
    BASE_DIR, "modelo-visao-computacional", "v3_final.pt"
)
EVIDENCIA_DEFAULT_DIR = os.path.join(BASE_DIR, "evidencias")


class Settings(BaseSettings):
    PROJECT_NAME: str = "AbraceAI API"
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///bd/abraceai.db")

    # ===== Visão computacional =====
    YOLO_MODEL_PATH: str = YOLO_DEFAULT_MODEL
    YOLO_CONF_THRESHOLD: float = 0.35

    # ===== Gemini =====
    # Chave deve vir do .env (não comitar). Fica vazia em dev sem créditos.
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3.1-flash-lite-preview"

    # ===== Gatilho por estabilidade / lock anti-duplicidade =====
    STABILITY_SECONDS: float = 1.5
    STABILITY_IOU_MIN: float = 0.85
    LOCK_SECONDS: float = 3.0

    # ===== Evidências =====
    EVIDENCIA_DIR: str = EVIDENCIA_DEFAULT_DIR

    # ===== Autenticação admin (PoC, sem senha) =====
    AUTH_SESSION_IDLE_MINUTES: int = 240
    AUTH_COOKIE_NAME: str = "abraceai_session"
    AUTH_COOKIE_SECURE: bool = False
    # Origens permitidas com credenciais (Vite dev + produção). Separadas por vírgula.
    CORS_ORIGINS: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Domínios após @ permitidos em cadastro/login (ex.: fecap.edu.br,abraceai.com.br)
    AUTH_EMAIL_DOMAINS: str = "fecap.edu.br,abraceai.com.br"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Force the database URL so bd.models.database uses it
os.environ["DATABASE_URL"] = settings.DATABASE_URL
