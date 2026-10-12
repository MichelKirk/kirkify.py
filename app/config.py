import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

class Settings:
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    KIRKS_DIR: Path = BASE_DIR / "kirks"

    # Admin Authentication
    ADMIN_EMAIL: str = os.getenv("ADMIN_EMAIL", "admin@kirkify.local")
    ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "admin123")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "kirkify-super-secret-jwt-key-2026")
    COOKIE_NAME: str = "kirkify_session"

    # Server Settings
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Google Drive
    GDRIVE_FOLDER_ID: str = os.getenv("GDRIVE_FOLDER_ID", "")
    GDRIVE_CREDENTIALS_FILE: str = os.getenv("GDRIVE_CREDENTIALS_FILE", str(DATA_DIR / "service_account.json"))

    # Instagram
    INSTAGRAM_USERNAME: str = os.getenv("INSTAGRAM_USERNAME", "")
    INSTAGRAM_PASSWORD: str = os.getenv("INSTAGRAM_PASSWORD", "")
    INSTAGRAM_SESSION_FILE: str = os.getenv("INSTAGRAM_SESSION_FILE", str(DATA_DIR / "ig_session.json"))

    # Scheduler & Queue
    DAILY_PUBLISH_TIME: str = os.getenv("DAILY_PUBLISH_TIME", "18:00")
    AUTO_PUBLISH_ENABLED: bool = os.getenv("AUTO_PUBLISH_ENABLED", "true").lower() in ("true", "1", "yes")
    DEFAULT_CAPTION: str = os.getenv("DEFAULT_CAPTION", "Kirkified with love 🚀\n\n#kirkify #faceswap #reels #ai")

    # Processing Flags
    USE_GPU: bool = os.getenv("USE_GPU", "false").lower() in ("true", "1", "yes")
    SCALE: float = float(os.getenv("SCALE", "1.0"))
    MIN_FPS: float = float(os.getenv("MIN_FPS", "60.0"))
    ENHANCE_FACE: bool = os.getenv("ENHANCE_FACE", "true").lower() in ("true", "1", "yes")
    CRF: int = int(os.getenv("CRF", "15"))
    WORKERS: int = int(os.getenv("WORKERS", "4"))

    # SQLite
    DB_PATH: Path = DATA_DIR / "kirkify.db"
    DATABASE_URL: str = f"sqlite:///{DB_PATH}"

settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.KIRKS_DIR.mkdir(parents=True, exist_ok=True)

