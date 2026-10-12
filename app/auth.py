import hmac
import hashlib
from typing import Optional
from fastapi import Request, HTTPException, status, Depends
from fastapi.responses import RedirectResponse
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from app.config import settings

serializer = URLSafeTimedSerializer(settings.SECRET_KEY)
SESSION_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

def create_session_token(email: str) -> str:
    return serializer.dumps({"email": email})

def verify_session_token(token: str) -> Optional[str]:
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        return data.get("email")
    except (BadSignature, SignatureExpired):
        return None

def verify_credentials(email: str, password: str) -> bool:
    # Constant-time comparison to prevent timing attacks
    email_ok = hmac.compare_digest(email.strip().lower(), settings.ADMIN_EMAIL.strip().lower())
    password_ok = hmac.compare_digest(password, settings.ADMIN_PASSWORD)
    return email_ok and password_ok

def get_current_admin(request: Request) -> str:
    token = request.cookies.get(settings.COOKIE_NAME)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"}
        )
    email = verify_session_token(token)
    if not email or email.strip().lower() != settings.ADMIN_EMAIL.strip().lower():
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"}
        )
    return email

def get_current_admin_api(request: Request) -> str:
    token = request.cookies.get(settings.COOKIE_NAME)
    # Also support Authorization: Bearer <token>
    if not token and "Authorization" in request.headers:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sessione non valida o scaduta. Effettua il login."
        )
    email = verify_session_token(token)
    if not email or email.strip().lower() != settings.ADMIN_EMAIL.strip().lower():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenziali non autorizzate."
        )
    return email

