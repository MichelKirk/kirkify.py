import os
import json
from pathlib import Path
from typing import Tuple, Optional
from app.config import settings

_cl = None

def get_instagram_client():
    global _cl
    if not settings.INSTAGRAM_USERNAME or not settings.INSTAGRAM_PASSWORD:
        return None

    try:
        from instagrapi import Client
        cl = Client()
        cl.delay_range = [2, 5]

        session_path = Path(settings.INSTAGRAM_SESSION_FILE)
        if session_path.exists():
            try:
                cl.load_settings(str(session_path))
                cl.login(settings.INSTAGRAM_USERNAME, settings.INSTAGRAM_PASSWORD)
                _cl = cl
                return _cl
            except Exception as e:
                print(f"[INSTAGRAM] Sessione non valida ({e}), nuovo login in corso...")

        cl.login(settings.INSTAGRAM_USERNAME, settings.INSTAGRAM_PASSWORD)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        cl.dump_settings(str(session_path))
        _cl = cl
        return _cl
    except Exception as e:
        print(f"[INSTAGRAM ERROR] Login fallito: {e}")
        return None


def is_instagram_configured() -> bool:
    return bool(settings.INSTAGRAM_USERNAME and settings.INSTAGRAM_PASSWORD)


def publish_to_instagram(media_path: str, caption: str, is_video: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Publishes a video (Reel) or photo to Instagram.
    Returns: (media_id, permalink)
    """
    if not is_instagram_configured():
        print("[INSTAGRAM] Credenziali non impostate in .env. Salto pubblicazione reale.")
        return None, None

    cl = get_instagram_client()
    if not cl:
        raise RuntimeError("Impossibile connettersi ad Instagram. Verifica credenziali e 2FA.")

    media_p = Path(media_path)
    if not media_p.exists():
        raise FileNotFoundError(f"File multimediale non trovato: {media_path}")

    try:
        if is_video:
            # Upload as Reel
            print(f"[INSTAGRAM] Caricamento Reel: {media_path}")
            media = cl.clip_upload(str(media_p), caption=caption)
        else:
            print(f"[INSTAGRAM] Caricamento Foto: {media_path}")
            media = cl.photo_upload(str(media_p), caption=caption)

        media_id = getattr(media, 'id', None) or getattr(media, 'pk', None)
        code = getattr(media, 'code', None)
        permalink = f"https://www.instagram.com/reel/{code}/" if (is_video and code) else (f"https://www.instagram.com/p/{code}/" if code else None)

        print(f"[INSTAGRAM] Pubblicato con successo! ID: {media_id}, Link: {permalink}")
        return str(media_id), permalink

    except Exception as e:
        print(f"[INSTAGRAM ERROR] Errore durante la pubblicazione: {e}")
        raise

