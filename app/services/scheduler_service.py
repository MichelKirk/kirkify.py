from datetime import datetime, timedelta, time
from typing import Optional
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy.orm import Session
from app.config import settings
from app.database import SessionLocal
from app.models import QueuePost, ActivityLog
from app.services.instagram_service import publish_to_instagram

scheduler = BackgroundScheduler()

def get_next_available_slot(db: Session) -> datetime:
    """
    Computes the next available date/time for a new post in the 1-post-per-day schedule.
    """
    try:
        hour, minute = map(int, settings.DAILY_PUBLISH_TIME.split(":"))
    except Exception:
        hour, minute = 18, 0

    now = datetime.now()
    today_slot = datetime.combine(now.date(), time(hour, minute))

    # Find the latest scheduled post in the queue
    latest_post = db.query(QueuePost).filter(
        QueuePost.status == "SCHEDULED"
    ).order_by(QueuePost.scheduled_for.desc()).first()

    if latest_post and latest_post.scheduled_for:
        next_date = latest_post.scheduled_for.date() + timedelta(days=1)
        return datetime.combine(next_date, time(hour, minute))
    else:
        # If today's slot hasn't passed yet, use today; otherwise tomorrow
        if now < today_slot:
            return today_slot
        else:
            return today_slot + timedelta(days=1)


def check_and_publish_due_posts():
    """
    Scheduler worker called periodically: checks for any scheduled post due for publication.
    """
    db = SessionLocal()
    try:
        if not settings.AUTO_PUBLISH_ENABLED:
            return

        now = datetime.now()
        # Find next due post
        due_post = db.query(QueuePost).filter(
            QueuePost.status == "SCHEDULED",
            QueuePost.scheduled_for <= now
        ).order_by(QueuePost.queue_order.asc(), QueuePost.scheduled_for.asc()).first()

        if due_post:
            print(f"[SCHEDULER] Trovato post in scadenza per la pubblicazione: ID {due_post.id}")
            publish_queue_post(due_post.id, db)

    except Exception as e:
        print(f"[SCHEDULER ERROR] {e}")
    finally:
        db.close()


def publish_queue_post(post_id: int, db: Session) -> bool:
    """
    Publishes a specific queue post to Instagram immediately.
    """
    post = db.query(QueuePost).filter(QueuePost.id == post_id).first()
    if not post:
        return False

    try:
        is_vid = (post.media_type == "video")
        ig_id, permalink = publish_to_instagram(
            media_path=post.media_path,
            caption=post.caption or settings.DEFAULT_CAPTION,
            is_video=is_vid
        )

        post.status = "POSTED"
        post.ig_post_id = ig_id
        post.ig_permalink = permalink
        post.published_at = datetime.utcnow()
        post.publish_error = None

        log = ActivityLog(
            level="SUCCESS",
            message=f"Post ID {post.id} pubblicato con successo su Instagram! Link: {permalink or 'N/A'}"
        )
        db.add(log)
        db.commit()
        return True

    except Exception as e:
        post.status = "FAILED"
        post.publish_error = str(e)
        log = ActivityLog(
            level="ERROR",
            message=f"Errore pubblicazione post ID {post.id}: {e}"
        )
        db.add(log)
        db.commit()
        return False


def start_scheduler():
    """Starts the background scheduler loop"""
    if not scheduler.running:
        scheduler.add_job(
            check_and_publish_due_posts,
            trigger=IntervalTrigger(seconds=60),
            id="daily_instagram_publisher",
            replace_existing=True
        )
        scheduler.start()
        print("[SCHEDULER] Background scheduler avviato (controllo ogni 60 secondi).")

