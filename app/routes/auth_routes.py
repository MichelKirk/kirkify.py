from fastapi import APIRouter, Request, Form, Response, Depends, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from app.auth import verify_credentials, create_session_token
from app.config import settings

router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@router.post("/login", response_class=HTMLResponse)
async def login_action(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    if verify_credentials(email, password):
        token = create_session_token(email)
        response = RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(
            key=settings.COOKIE_NAME,
            value=token,
            httponly=True,
            max_age=60 * 60 * 24 * 30,
            samesite="lax"
        )
        return response
    else:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Credenziali non valide. Verifica email e password configurate in .env."}
        )

@router.get("/logout")
async def logout_action():
    response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(settings.COOKIE_NAME)
    return response

