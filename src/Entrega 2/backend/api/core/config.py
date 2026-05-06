import os
from pydantic_settings import BaseSettings

# BASE_DIR is backend/api/core -> backend/api -> backend
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(BASE_DIR, "bd", "abraceai.db")

class Settings(BaseSettings):
    PROJECT_NAME: str = "AbraceAI API"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "uma-chave-secreta-muito-segura-para-dev"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///bd/abraceai.db")

    class Config:
        env_file = ".env"

settings = Settings()

# Force the database URL so bd.models.database uses it
os.environ["DATABASE_URL"] = settings.DATABASE_URL
