import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from app.config import settings
from app.database import SessionLocal
from app.models import Job, QueuePost, ActivityLog
from app.services.downloader import download_instagram_media
from app.services.kirkifier_service import process_media
from app.services.gdrive_uploader import upload_to_google_drive
from app.services.scheduler_service import get_next_available_slot

def run_job_pipeline(job_id: str, custom_caption: Optional[str] = None):
    """
    Executes the full pipeline in a separate background thread:
    1. Download Reel with yt-dlp
    2. Kirkify face-swap & 60fps restoration
    3. Upload to Google Drive
    4. Enqueue into daily Instagram schedule
    """
    db = SessionLocal()
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        db.close()
        return

    try:
        # Step 1: Download Reel
        if job.source_url:
            job.status = "DOWNLOADING"
            job.current_step = "Download del Reel Instagram con yt-dlp..."
            job.progress = 5.0
            db.commit()

            downloaded_path, meta = download_instagram_media(job.source_url)
            job.input_path = downloaded_path
            job.progress = 20.0
            db.commit()
        elif not job.input_path or not os.path.exists(job.input_path):
            raise FileNotFoundError("File multimediale di origine non trovato.")

        # Determine media type and output path
        ext = Path(job.input_path).suffix.lower()
        is_video = ext in ('.mp4', '.mov', '.avi', '.mkv', '.webm')
        job.media_type = "video" if is_video else "image"

        output_filename = f"kirkified_{Path(job.input_path).stem}.mp4" if is_video else f"kirkified_{Path(job.input_path).stem}.jpg"
        output_path = str(settings.PROCESSED_DIR / output_filename)
        job.output_path = output_path

        # Step 2: Kirkify Processing
        job.status = "PROCESSING"
        job.current_step = "Elaborazione Face-Swap, GFPGAN & 60 FPS in corso..."
        job.progress = 30.0
        db.commit()

        def update_progress(pct: float, step_name: str):
            job.progress = min(85.0, 30.0 + (pct * 0.55))
            job.current_step = step_name
            db.commit()

        process_media(
            input_path=job.input_path,
            output_path=output_path,
            progress_callback=update_progress
        )

        # Step 3: Google Drive Upload
        job.status = "UPLOADING_GDRIVE"
        job.current_step = "Caricamento su Google Drive..."
        job.progress = 88.0
        db.commit()

        gdrive_id, gdrive_link = upload_to_google_drive(output_path)

        # Step 4: Schedule Post (1 post per day)
        job.status = "ENQUEUING"
        job.current_step = "Pianificazione nella coda Instagram..."
        job.progress = 95.0
        db.commit()

        slot_time = get_next_available_slot(db)
        caption_text = custom_caption if custom_caption else settings.DEFAULT_CAPTION

        queue_post = QueuePost(
            job_id=job.id,
            media_path=output_path,
            media_type=job.media_type,
            caption=caption_text,
            scheduled_for=slot_time,
            status="SCHEDULED",
            gdrive_file_id=gdrive_id,
            gdrive_link=gdrive_link
        )
        db.add(queue_post)

        job.status = "COMPLETED"
        job.current_step = f"Completato! In coda per il {slot_time.strftime('%d/%m/%Y alle %H:%M')}"
        job.progress = 100.0
        job.completed_at = datetime.utcnow()

        log = ActivityLog(
            level="SUCCESS",
            message=f"Job {job.id} completato con successo. Media salvato, sincronizzato su Google Drive e programmato per il {slot_time.strftime('%d/%m/%Y %H:%M')}."
        )
        db.add(log)
        db.commit()

    except Exception as e:
        job.status = "FAILED"
        job.current_step = "Errore durante l'elaborazione"
        job.error_message = str(e)
        log = ActivityLog(
            level="ERROR",
            message=f"Job {job.id} fallito: {e}"
        )
        db.add(log)
        db.commit()

    finally:
        db.close()


def dispatch_job(job_id: str, custom_caption: Optional[str] = None):
    """Launches the pipeline asynchronously in background thread"""
    t = threading.Thread(target=run_job_pipeline, args=(job_id, custom_caption), daemon=True)
    t.start()

