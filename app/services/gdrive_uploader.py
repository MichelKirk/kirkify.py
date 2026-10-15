import os
from pathlib import Path
from typing import Tuple, Optional
from app.config import settings

def is_gdrive_configured() -> bool:
    cred_file = Path(settings.GDRIVE_CREDENTIALS_FILE)
    return bool(settings.GDRIVE_FOLDER_ID and cred_file.exists())

def upload_to_google_drive(file_path: str, custom_name: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Uploads a file to the configured Google Drive folder.
    Returns: (file_id, web_view_link)
    """
    if not is_gdrive_configured():
        print(f"[GDRIVE] Google Drive non configurato o credenziali assenti ({settings.GDRIVE_CREDENTIALS_FILE}). Salto upload cloud.")
        return None, None

    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        SCOPES = ['https://www.googleapis.com/auth/drive.file']
        creds = service_account.Credentials.from_service_account_file(
            settings.GDRIVE_CREDENTIALS_FILE, scopes=SCOPES
        )
        service = build('drive', 'v3', credentials=creds)

        file_name = custom_name or Path(file_path).name
        if file_path.lower().endswith(('.mp4', '.mov', '.mkv', '.webm')):
            mime_type = 'video/mp4'
        elif file_path.lower().endswith('.pdf'):
            mime_type = 'application/pdf'
        else:
            mime_type = 'image/jpeg'

        file_metadata = {
            'name': file_name,
            'parents': [settings.GDRIVE_FOLDER_ID]
        }

        media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
        file_obj = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()

        file_id = file_obj.get('id')
        link = file_obj.get('webViewLink')
        print(f"[GDRIVE] Upload completato! ID: {file_id}, Link: {link}")
        return file_id, link

    except Exception as e:
        print(f"[GDRIVE ERROR] Errore durante l'upload su Google Drive: {e}")
        return None, None

