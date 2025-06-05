from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # API Config
    api_v1_str: str = "/api/v1"
    project_name: str = "Modelado Predictivo de Condiciones Ambientales en Agricultura Urbana Vertical Mediante Fotoluminiscencia y Aprendizaje Automático"
    version: str = "1.0.0"

    # Security - Agregar valor por defecto
    secret_key: str = "dev-secret-key-change-in-production-123456789"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # ML Models
    models_path: str = "../models"

    # Redis Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    cache_ttl: int = 300  # 5 minutes

    # Database
    database_url: str = "postgresql+asyncpg://user:pass@localhost/vertical_farming"

    # Monitoring
    enable_metrics: bool = True

    # Limits
    max_batch_size: int = 100
    rate_limit_per_minute: int = 60

    # Debug mode
    debug: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = False


@lru_cache()
def get_settings():
    return Settings()