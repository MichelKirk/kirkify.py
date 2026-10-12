import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Request, Depends, HTTPException, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.auth import get_current_admin, get_current_admin_api
from app.config import settings
from app.database import get_db
from app.models import SystemSetting
from app.services.gdrive_uploader import is_gdrive_configured
from app.services.instagram_service import is_instagram_configured

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

@router.get("/settings", response_class=HTMLResponse)
async def settings_page(
    request: Request,
    admin_email: str = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    # Get current reference face files
    face_files = []
    if settings.KIRKS_DIR.exists():
        face_files = [f.name for f in settings.KIRKS_DIR.iterdir() if f.is_file() and f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]

    context = {
        "request": request,
        "admin_email": admin_email,
        "default_caption": settings.DEFAULT_CAPTION,
        "daily_publish_time": settings.DAILY_PUBLISH_TIME,
        "auto_publish_enabled": settings.AUTO_PUBLISH_ENABLED,
        "gdrive_configured": is_gdrive_configured(),
        "gdrive_folder_id": settings.GDRIVE_FOLDER_ID,
        "instagram_configured": is_instagram_configured(),
        "instagram_user": settings.INSTAGRAM_USERNAME,
        "face_files": face_files,
        "use_gpu": settings.USE_GPU,
        "scale": settings.SCALE,
        "min_fps": settings.MIN_FPS
    }
    return templates.TemplateResponse("settings.html", context)


@router.post("/api/settings")
async def update_settings(
    default_caption: str = Form(...),
    daily_publish_time: str = Form(...),
    auto_publish_enabled: bool = Form(False),
    admin_email: str = Depends(get_current_admin_api),
    db: Session = Depends(get_db)
):
    settings.DEFAULT_CAPTION = default_caption
    settings.DAILY_PUBLISH_TIME = daily_publish_time
    settings.AUTO_PUBLISH_ENABLED = auto_publish_enabled

    return {"status": "success", "message": "Impostazioni salvate con successo!"}


@router.post("/api/faces/upload")
async def upload_face_image(
    file: UploadFile = File(...),
    admin_email: str = Depends(get_current_admin_api)
):
    ext = Path(file.filename).suffix.lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.webp'):
        raise HTTPException(status_code=400, detail="Formato file non supportato. Usa JPG, PNG o WebP.")

    settings.KIRKS_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = settings.KIRKS_DIR / file.filename
    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"status": "success", "message": f"Volto '{file.filename}' caricato con successo in kirks/!"}


@router.delete("/api/faces/{filename}")
async def delete_face_image(
    filename: str,
    admin_email: str = Depends(get_current_admin_api)
):
    file_path = settings.KIRKS_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"status": "success", "message": f"Volto '{filename}' eliminato."}
    raise HTTPException(status_code=404, detail="File non trovato")

