import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, Form, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from app.auth import get_current_admin_api
from app.config import settings
from app.database import get_db
from app.models import Job
from app.services.pipeline_worker import dispatch_job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

@router.post("/create")
async def create_job(
    instagram_url: Optional[str] = Form(None),
    caption: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    admin_email: str = Depends(get_current_admin_api),
    db: Session = Depends(get_db)
):
    if not instagram_url and not file:
        raise HTTPException(status_code=400, detail="Fornisci un link Instagram Reel oppure carica un file.")

    input_path = None
    if file and file.filename:
        settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        dest_path = settings.UPLOADS_DIR / file.filename
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        input_path = str(dest_path)

    new_job = Job(
        source_url=instagram_url.strip() if instagram_url else None,
        input_path=input_path,
        status="PENDING",
        progress=0.0,
        current_step="Job accodato..."
    )
    db.add(new_job)
    db.commit()
    db.refresh(new_job)

    # Launch background processing pipeline
    dispatch_job(new_job.id, custom_caption=caption)

    return {
        "status": "success",
        "job_id": new_job.id,
        "message": "Job avviato con successo!"
    }


@router.get("")
async def list_jobs(
    admin_email: str = Depends(get_current_admin_api),
    db: Session = Depends(get_db)
):
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(20).all()
    return [{
        "id": j.id,
        "source_url": j.source_url,
        "status": j.status,
        "progress": j.progress,
        "current_step": j.current_step,
        "created_at": j.created_at.strftime("%Y-%m-%d %H:%M:%S") if j.created_at else None,
        "error_message": j.error_message,
        "output_filename": Path(j.output_path).name if j.output_path else None
    } for j in jobs]


@router.get("/{job_id}")
async def get_job_status(
    job_id: str,
    admin_email: str = Depends(get_current_admin_api),
    db: Session = Depends(get_db)
):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job non trovato")

    return {
        "id": job.id,
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "error_message": job.error_message,
        "completed_at": job.completed_at.strftime("%Y-%m-%d %H:%M:%S") if job.completed_at else None,
        "output_filename": Path(job.output_path).name if job.output_path else None
    }

