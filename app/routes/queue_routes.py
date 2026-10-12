from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, Request, Depends, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from app.auth import get_current_admin, get_current_admin_api
from app.config import settings
from app.database import get_db
from app.models import QueuePost
from app.services.scheduler_service import publish_queue_post

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

@router.get("/queue", response_class=HTMLResponse)
async def queue_page(
    request: Request,
    admin_email: str = Depends(get_current_admin),
    db: Session = Depends(get_db)
):
    scheduled_posts = db.query(QueuePost).filter(
        QueuePost.status == "SCHEDULED"
    ).order_by(QueuePost.scheduled_for.asc()).all()

    published_posts = db.query(QueuePost).filter(
        QueuePost.status.in_(["POSTED", "FAILED"])
    ).order_by(QueuePost.published_at.desc()).limit(15).all()

    context = {
        "request": request,
        "admin_email": admin_email,
        "scheduled_posts": scheduled_posts,
        "published_posts": published_posts,
        "default_caption": settings.DEFAULT_CAPTION,
        "daily_publish_time": settings.DAILY_PUBLISH_TIME
    }
    return templates.TemplateResponse("queue.html", context)


@router.post("/api/queue/{post_id}/publish")
async def trigger_publish_now(
    post_id: int,
    admin_email: str = Depends(get_current_admin_api),
    db: Session = Depends(get_db)
):
    success = publish_queue_post(post_id, db)
    if not success:
        post = db.query(QueuePost).filter(QueuePost.id == post_id).first()
        err = post.publish_error if post else "Errore sconosciuto"
        raise HTTPException(status_code=500, detail=f"Pubblicazione fallita: {err}")

    return {"status": "success", "message": "Post pubblicato con successo su Instagram!"}


@router.post("/api/queue/{post_id}/caption")
async def update_caption(
    post_id: int,
    caption: str = Form(...),
    admin_email: str = Depends(get_current_admin_api),
    db: Session = Depends(get_db)
):
    post = db.query(QueuePost).filter(QueuePost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post non trovato")

    post.caption = caption
    db.commit()
    return {"status": "success", "message": "Descrizione aggiornata con successo!"}


@router.post("/api/queue/{post_id}/reschedule")
async def reschedule_post(
    post_id: int,
    new_date: str = Form(...),  # Format: YYYY-MM-DDTHH:MM
    admin_email: str = Depends(get_current_admin_api),
    db: Session = Depends(get_db)
):
    post = db.query(QueuePost).filter(QueuePost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post non trovato")

    try:
        dt = datetime.fromisoformat(new_date)
        post.scheduled_for = dt
        db.commit()
        return {"status": "success", "message": "Orario riprogrammato con successo!"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Formato data non valido: {e}")


@router.delete("/api/queue/{post_id}")
async def delete_queue_post(
    post_id: int,
    admin_email: str = Depends(get_current_admin_api),
    db: Session = Depends(get_db)
):
    post = db.query(QueuePost).filter(QueuePost.id == post_id).first()
    if not post:
        raise HTTPException(status_code=404, detail="Post non trovato")

    db.delete(post)
    db.commit()
    return {"status": "success", "message": "Post rimosso dalla coda."}

