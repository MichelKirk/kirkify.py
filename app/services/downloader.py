import os
import uuid
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import yt_dlp
from app.config import settings

def download_instagram_media(url: str, output_dir: Optional[Path] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Downloads an Instagram Reel or Post video using yt-dlp.
    Returns: (downloaded_file_path, metadata_dict)
    """
    if output_dir is None:
        output_dir = settings.UPLOADS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    file_id = str(uuid.uuid4())
    out_tmpl = str(output_dir / f"{file_id}.%(ext)s")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': out_tmpl,
        'noplaylist': True,
        'quiet': False,
        'no_warnings': False,
        # Common headers to avoid bot blocks
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        },
    }

    # If an Instagram cookies file exists in data/, use it
    cookies_path = settings.DATA_DIR / "instagram_cookies.txt"
    if cookies_path.exists():
        ydl_opts['cookiefile'] = str(cookies_path)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)

    # Some extractors might download .webm or mkv, ensure we return the actual written file
    if not os.path.exists(filename):
        # Scan for matching file_id
        for p in output_dir.glob(f"{file_id}.*"):
            filename = str(p)
            break

    metadata = {
        'title': info.get('title') or '',
        'description': info.get('description') or '',
        'thumbnail': info.get('thumbnail') or '',
        'duration': info.get('duration') or 0,
        'uploader': info.get('uploader') or '',
        'original_url': url,
    }

    return filename, metadata

