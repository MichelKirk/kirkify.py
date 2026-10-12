import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from app.database import Base

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_url = Column(String(500), nullable=True)
    input_path = Column(String(500), nullable=True)
    output_path = Column(String(500), nullable=True)
    media_type = Column(String(20), default="video")  # 'video' or 'image'
    
    # Status: PENDING, DOWNLOADING, PROCESSING, COMPLETED, FAILED
    status = Column(String(30), default="PENDING")
    progress = Column(Float, default=0.0)
    current_step = Column(String(100), default="In attesa...")
    error_message = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationship to scheduled post
    queue_post = relationship("QueuePost", back_populates="job", uselist=False, cascade="all, delete-orphan")


class QueuePost(Base):
    __tablename__ = "queue_posts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(36), ForeignKey("jobs.id"), nullable=True)
    
    media_path = Column(String(500), nullable=False)
    media_type = Column(String(20), default="video")  # 'video' or 'image'
    thumbnail_path = Column(String(500), nullable=True)
    
    caption = Column(Text, nullable=True)
    scheduled_for = Column(DateTime, nullable=True)
    queue_order = Column(Integer, default=0)
    
    # Status: SCHEDULED, POSTED, FAILED, PAUSED
    status = Column(String(30), default="SCHEDULED")
    
    # Google Drive metadata
    gdrive_file_id = Column(String(100), nullable=True)
    gdrive_link = Column(String(500), nullable=True)
    
    # Instagram publishing metadata
    ig_post_id = Column(String(100), nullable=True)
    ig_permalink = Column(String(500), nullable=True)
    published_at = Column(DateTime, nullable=True)
    publish_error = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    job = relationship("Job", back_populates="queue_post")


class SystemSetting(Base):
    __tablename__ = "system_settings"

    key = Column(String(50), primary_key=True)
    value = Column(Text, nullable=True)


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    level = Column(String(20), default="INFO")  # INFO, WARNING, ERROR, SUCCESS
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

