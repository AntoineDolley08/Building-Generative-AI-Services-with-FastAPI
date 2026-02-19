from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+psycopg://fastapi:mysecretpassword@localhost:5432/backend_db"
    jwt_secret_key: str = "your_secret_key"
    jwt_algorithm: str = "HS256"
    jwt_expires_in_minutes: int = 60
