import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.database import init_db
from app.services.scheduler_service import start_scheduler
from app.routes.auth_routes import router as auth_router, templates as auth_templates
from app.routes.dashboard_routes import router as dashboard_router, templates as dash_templates
from app.routes.jobs_routes import router as jobs_router
from app.routes.queue_routes import router as queue_router, templates as queue_templates
from app.routes.settings_routes import router as settings_router, templates as settings_templates

# Jinja custom filter: basename
def basename_filter(path_str):
    if not path_str:
        return ""
    return Path(path_str).name

for t in (auth_templates, dash_templates, queue_templates, settings_templates):
    t.env.filters["basename"] = basename_filter

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[APP] Inizializzazione database SQLite...")
    init_db()
    print("[APP] Avvio scheduler pubblicazione Instagram (1 al giorno)...")
    start_scheduler()
    yield
    # Shutdown
    print("[APP] Arresto completato.")

app = FastAPI(
    title="Kirkify Studio Automation",
    description="Web platform for Instagram Reel downloading, Face-Swapping, Google Drive uploading, and automated daily scheduling.",
    version="1.0.0",
    lifespan=lifespan
)

# Include Routers
app.include_router(auth_router)
app.include_router(dashboard_router)
app.include_router(jobs_router)
app.include_router(queue_router)
app.include_router(settings_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)

