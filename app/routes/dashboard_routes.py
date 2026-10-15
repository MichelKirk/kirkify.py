from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from pathlib import Path
from app.auth import get_current_admin
from app.config import settings
from app.database import get_db
from app.models import Job, QueuePost, ActivityLog
from app.services.gdrive_uploader import is_gdrive_configured
from app.services.instagram_service import is_instagram_configured

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

@router.get("/", response_class=HTMLResponse)
async def dashboard_page(
    request: Request,
    admin_email: str = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    recent_jobs = db.query(Job).order_by(Job.created_at.desc()).limit(10).all()
    queue_posts = db.query(QueuePost).filter(QueuePost.status == "SCHEDULED").order_by(QueuePost.scheduled_for.asc()).all()
    recent_logs = db.query(ActivityLog).order_by(ActivityLog.created_at.desc()).limit(8).all()
    
    total_processed = db.query(Job).filter(Job.status == "COMPLETED").count()
    total_queued = len(queue_posts)
    total_posted = db.query(QueuePost).filter(QueuePost.status == "POSTED").count()

    context = {
        "request": request,
        "admin_email": admin_email,
        "recent_jobs": recent_jobs,
        "queue_posts": queue_posts,
        "recent_logs": recent_logs,
        "stats": {
            "processed": total_processed,
            "queued": total_queued,
            "posted": total_posted
        },
        "status": {
            "gdrive": is_gdrive_configured(),
            "instagram": is_instagram_configured(),
            "auto_publish": settings.AUTO_PUBLISH_ENABLED,
            "publish_time": settings.DAILY_PUBLISH_TIME,
        }
    }
    return templates.TemplateResponse("dashboard.html", context)


@router.get("/media/{filename}")
async def serve_media(
    filename: str,
    admin_email: str = Depends(get_current_admin)
):
    # Check in processed directory
    file_path = settings.PROCESSED_DIR / filename
    if not file_path.exists():
        file_path = settings.UPLOADS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File non trovato")

    ext = filename.lower()
    if ext.endswith((".mp4", ".mov", ".mkv", ".webm")):
        mime = "video/mp4"
    elif ext.endswith(".pdf"):
        mime = "application/pdf"
    else:
        mime = "image/jpeg"
    return FileResponse(file_path, media_type=mime)


