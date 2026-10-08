import sys
import os
import shutil

# Ensure FFmpeg is accessible even if the terminal session hasn't been reloaded after install
if os.name == 'nt' and not shutil.which('ffmpeg'):
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            user_path, _ = winreg.QueryValueEx(key, "Path")
            for p in user_path.split(';'):
                if p and os.path.isdir(p) and p not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")
    except Exception:
        pass

from subprocess import run
from concurrent.futures import ThreadPoolExecutor, as_completed
from shutil import rmtree
from pathlib import Path
from random import randint
from contextlib import contextmanager
from tqdm import tqdm

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from cv2 import imread, imwrite, VideoCapture, CAP_PROP_FPS

# Constant directories
UNPROCESSED_DIR = Path('unprocessed_frames')
PROCESSED_DIR = Path('processed_frames')
KIRKS_DIR = Path('kirks')

# Utilities for ONNX Runtime/GPU verification
def ort_available_and_providers() -> tuple[bool, list[str]]:
    try:
        import onnxruntime as ort  # type: ignore
        try:
            # If directory is not provided, it will auto-search in site-packages/nvidia/...
            ort.preload_dlls()
            print("[INFO] CUDA DLLs preloaded from nvidia-* wheels")
        except Exception as e:
            print("[WARN] Could not preload CUDA DLLs:", e)
        providers = []
        try:
            providers = list(ort.get_available_providers())  # e.g., ['CUDAExecutionProvider','CPUExecutionProvider']
        except Exception:
            providers = []
        return True, providers
    except Exception:
        return False, []

def get_session_providers(session) -> list[str]:
    try:
        return list(session.get_providers())
    except Exception:
        return []

@contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

def initialize_faceanalysis_and_swapper(det_size=(640, 640), ctx_id: int = 0, swapper_providers: list[str] | None = None):
    faceanalysis = FaceAnalysis(name="buffalo_l")
    faceanalysis.prepare(ctx_id=ctx_id, det_size=det_size)
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=False, download_zip=False)
    if swapper_providers:
        try:
            import onnxruntime as ort  # noqa: F401
            sess = getattr(swapper, 'session', None) or getattr(swapper, 'model', None)
            if sess and hasattr(sess, 'set_providers'):
                sess.set_providers(swapper_providers)
        except Exception:
            pass
    return faceanalysis, swapper

def get_video_fps(video_path: str) -> float:
    cap = VideoCapture(video_path)
    fps = cap.get(CAP_PROP_FPS)
    cap.release()
    return fps

def extract_frames(video_path: str):
    """Extract all frames to unprocessed_frames/"""
    UNPROCESSED_DIR.mkdir(exist_ok=True)
    run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', video_path,
        '-q:v', '2',
        str(UNPROCESSED_DIR / 'frame_%04d.png')
    ], check=True)

def extract_audio(video_path: str) -> str | None:
    """Extract audio to audio.aac if present, otherwise return None"""
    audio_path = 'audio.aac'
    if os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            pass
    res = run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', video_path,
        '-map', '0:a:0', '-acodec', 'copy',
        audio_path
    ], capture_output=True)
    if res.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
        return audio_path
    return None

def reconstruct_video(fps: float, audio_path: str | None, output_path: str, frame_step: int):
    """Combine processed frames with audio (if available)"""
    PROCESSED_DIR.mkdir(exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(fps / frame_step),
        '-i', str(PROCESSED_DIR / 'frame_%04d.png'),
    ]
    if audio_path and os.path.exists(audio_path):
        cmd.extend([
            '-i', audio_path,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-c:a', 'copy', '-shortest',
            output_path
        ])
    else:
        cmd.extend([
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            output_path
        ])
    run(cmd, check=True)

def cleanup(audio_path: str | None = None):
    """Remove intermediate files"""
    rmtree(UNPROCESSED_DIR, ignore_errors=True)
    rmtree(PROCESSED_DIR, ignore_errors=True)
    if audio_path and os.path.exists(audio_path):
        try:
            os.remove(audio_path)
        except OSError:
            pass

def get_canonical_kirk_face(faceanalysis: FaceAnalysis):
    """
    Computes an averaged, normalized identity embedding from all available Kirk reference images.
    This guarantees that the target face identity embedding remains constant and jitter-free.
    """
    embeddings = []
    best_face = None
    best_score = -1.0

    for i in range(3):
        kirk_path = KIRKS_DIR / f'kirk_{i}.jpg'
        if not kirk_path.exists():
            continue
        kirk_img = imread(str(kirk_path))
        if kirk_img is None:
            continue
        faces = faceanalysis.get(kirk_img)
        if faces:
            face = faces[0]
            embeddings.append(face.normed_embedding)
            score = getattr(face, 'det_score', 0.0)
            if score > best_score:
                best_score = score
                best_face = face

    if best_face is None or not embeddings:
        raise RuntimeError("Could not detect a face in any Kirk reference image")

    avg_emb = np.mean(embeddings, axis=0)
    avg_emb /= np.linalg.norm(avg_emb)

    canonical_face = Face(best_face)
    canonical_face.embedding = avg_emb
    return canonical_face

class FaceTracker:
    """
    Temporal Face Tracker and Stabilizer for video processing.
    Eliminates flickering through:
    1. Adaptive Exponential Smoothing on facial landmarks (kps) and bounding box (bbox)
       to remove high-frequency neural net detection jitter.
    2. Missed-detection recovery (grace period) preventing single-frame dropped face flicker.
    3. Spatial IoU + centroid tracking to ensure consistent face-to-identity assignment across frames.
    """
    def __init__(self, alpha: float = 0.60, max_missed: int = 2, iou_thresh: float = 0.25):
        self.alpha = alpha
        self.max_missed = max_missed
        self.iou_thresh = iou_thresh
        self.tracks: list[dict] = []
        self.next_id = 0

    @staticmethod
    def compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
        boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
        boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))

        unionArea = boxAArea + boxBArea - interArea
        return interArea / unionArea if unionArea > 0 else 0.0

    def update(self, detected_faces: list) -> list:
        matched_track_indices = set()
        matched_det_indices = set()

        if self.tracks and detected_faces:
            pairs = []
            for t_idx, track in enumerate(self.tracks):
                track_center = np.mean(track['kps'], axis=0)
                for d_idx, det in enumerate(detected_faces):
                    iou = self.compute_iou(track['bbox'], det.bbox)
                    det_center = np.mean(det.kps, axis=0)
                    dist = np.linalg.norm(track_center - det_center)
                    if iou >= self.iou_thresh or dist < 60:
                        score = iou + max(0.0, 1.0 - (dist / 120.0))
                        pairs.append((score, t_idx, d_idx))

            pairs.sort(key=lambda x: x[0], reverse=True)
            for _, t_idx, d_idx in pairs:
                if t_idx not in matched_track_indices and d_idx not in matched_det_indices:
                    matched_track_indices.add(t_idx)
                    matched_det_indices.add(d_idx)
                    track = self.tracks[t_idx]
                    det = detected_faces[d_idx]

                    # Adaptive smoothing: snap if large sudden movement, smooth if small jitter
                    diff = np.linalg.norm(det.kps - track['kps'])
                    if diff > 80:
                        a = 1.0
                    elif diff > 30:
                        a = 0.85
                    else:
                        a = self.alpha

                    track['kps'] = a * det.kps + (1.0 - a) * track['kps']
                    track['bbox'] = a * det.bbox + (1.0 - a) * track['bbox']
                    track['missed'] = 0
                    track['last_face'] = det

        # Register new detected faces that didn't match existing tracks
        for d_idx, det in enumerate(detected_faces):
            if d_idx not in matched_det_indices:
                self.tracks.append({
                    'id': self.next_id,
                    'bbox': det.bbox.copy(),
                    'kps': det.kps.copy(),
                    'missed': 0,
                    'last_face': det
                })
                self.next_id += 1

        # Handle unmatched tracks (grace period for temporarily dropped detections)
        active_tracks = []
        for t_idx, track in enumerate(self.tracks):
            if t_idx not in matched_track_indices:
                track['missed'] += 1
            if track['missed'] <= self.max_missed:
                active_tracks.append(track)
        self.tracks = active_tracks

        # Return stabilized face objects for the current frame
        stabilized = []
        for track in self.tracks:
            f = Face(track['last_face'])
            f.kps = track['kps'].copy()
            f.bbox = track['bbox'].copy()
            stabilized.append(f)
        return stabilized

def kirkify_frame(frame_path: str, output_path: str, faceanalysis: FaceAnalysis, swapper, kirk_face):
    """Process a single frame"""
    img = imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {frame_path}")
    faces = faceanalysis.get(img)

    if faces:
        res = img.copy()
        for face in faces:
            res = swapper.get(res, face, kirk_face, paste_back=True)
        
        if not isinstance(res, np.ndarray):
            res = np.array(res)
        imwrite(output_path, res)
        return True
    else:
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        imwrite(output_path, img)
        return False

def process_all_frames(faceanalysis: FaceAnalysis, swapper, kirk_face, frame_step: int = 1, workers: int | None = None, smooth: bool = True) -> dict:
    """Process each frame in unprocessed_frames/ with tqdm progress bar.
    - smooth: enables temporal face tracking and landmark smoothing to eliminate flickering.
    - frame_step: process 1 of every N frames (skip to speed up).
    - workers: number of threads for parallel processing (used when smooth=False).
    """
    PROCESSED_DIR.mkdir(exist_ok=True)
    all_files = sorted([f for f in os.listdir(UNPROCESSED_DIR) if f.endswith('.png')])
    frame_files = all_files[::max(1, frame_step)]
    total = len(frame_files)
    with_faces = 0

    if smooth:
        tracker = FaceTracker(alpha=0.60, max_missed=2)
        desc = "Processing frames (stabilized)"
        for idx, filename in enumerate(tqdm(frame_files, total=total, desc=desc, unit="frame")):
            input_path = str(UNPROCESSED_DIR / filename)
            output_path = str(PROCESSED_DIR / f"frame_{idx:04d}.png")
            
            img = imread(input_path)
            if img is None:
                continue

            detected_faces = faceanalysis.get(img)
            stabilized_faces = tracker.update(detected_faces)

            if stabilized_faces:
                with_faces += 1
                res = img.copy()
                for face in stabilized_faces:
                    res = swapper.get(res, face, kirk_face, paste_back=True)
                imwrite(output_path, res)
            else:
                imwrite(output_path, img)

    else:
        def _process(idx_and_name: tuple[int, str]) -> bool:
            idx, filename = idx_and_name
            input_path = str(UNPROCESSED_DIR / filename)
            output_path = str(PROCESSED_DIR / f"frame_{idx:04d}.png")
            return kirkify_frame(input_path, output_path, faceanalysis, swapper, kirk_face)

        workers = workers or max(1, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_process, (i, fn)) for i, fn in enumerate(frame_files)]
            for fut in tqdm(as_completed(futures), total=total, desc="Processing frames", unit="frame"):
                try:
                    if fut.result():
                        with_faces += 1
                except Exception:
                    pass

    return {'total': len(all_files), 'processed': len(frame_files), 'with_faces': with_faces}

SUPPORTED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
SUPPORTED_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
SUPPORTED_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_VIDEO_EXTS

def kirkify_video(TARGET_PATH, OUTPUT_PATH, FACE_ANALYSIS, FACE_SWAPPER, frame_step: int = 1, workers: int | None = None, smooth: bool = True):
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    audio_path = None
    kirk_face = get_canonical_kirk_face(FACE_ANALYSIS)
    try:
        print("Extracting frames...")
        extract_frames(TARGET_PATH)
        
        print("Extracting audio...")
        audio_path = extract_audio(TARGET_PATH)
        
        print("Processing frames..." + (" (temporal stabilization ON)" if smooth else " (stabilization OFF)"))
        stats = process_all_frames(FACE_ANALYSIS, FACE_SWAPPER, kirk_face, frame_step=frame_step, workers=workers, smooth=smooth)
        print(f"  Frames: total {stats['total']}, processed {stats['processed']}, with faces {stats['with_faces']}")
        
        print("Reconstructing video...")
        fps = get_video_fps(TARGET_PATH)
        reconstruct_video(fps, audio_path, OUTPUT_PATH, frame_step)
    finally:
        print("Cleaning up...")
        cleanup(audio_path)
    
    print(f"Done! Output saved to {OUTPUT_PATH}")

def kirkify_image(TARGET_PATH, OUTPUT_PATH, FACE_ANALYSIS, FACE_SWAPPER):
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    kirk_face = get_canonical_kirk_face(FACE_ANALYSIS)

    print("Kirkifying image...")
    face_detected = kirkify_frame(TARGET_PATH, OUTPUT_PATH, FACE_ANALYSIS, FACE_SWAPPER, kirk_face)

    if not face_detected:
        print("  No faces detected. Image copied unchanged.")

    print(f"Done! Output saved to {OUTPUT_PATH}")

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "init":
        initialize_faceanalysis_and_swapper()
        print("Initialized!")
        return

    # Parse arguments
    args = sys.argv[1:]
    fast = '--fast' in args
    use_gpu = '--gpu' in args
    use_cpu = '--cpu' in args
    smooth = '--no-smooth' not in args
    step = 1
    workers = None

    positional = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ('--fast', '--gpu', '--cpu'):
            i += 1
        elif arg == '--frame-step':
            if i + 1 < len(args):
                try:
                    step = max(1, int(args[i+1]))
                except ValueError:
                    pass
                i += 2
            else:
                i += 1
        elif arg == '--workers':
            if i + 1 < len(args):
                try:
                    workers = max(1, int(args[i+1]))
                except ValueError:
                    pass
                i += 2
            else:
                i += 1
        elif arg.startswith('-'):
            i += 1
        else:
            positional.append(arg)
            i += 1

    input_dir_default = Path("input")
    output_dir_default = Path("output")
    input_dir_default.mkdir(exist_ok=True)
    output_dir_default.mkdir(exist_ok=True)

    items_to_process: list[tuple[Path, Path, bool]] = []  # (src_path, dst_path, is_image)

    if len(positional) == 0:
        # Batch mode: process all files in input/ -> output/
        valid_files = sorted([
            f for f in input_dir_default.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
        ])
        if not valid_files:
            print("Nessun file trovato da elaborare.")
            print(f"Inserisci video o immagini nella cartella: {input_dir_default.resolve()}")
            print("Formati supportati:")
            print(f"  Immagini: {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}")
            print(f"  Video:    {', '.join(sorted(SUPPORTED_VIDEO_EXTS))}")
            return
        
        for f in valid_files:
            is_img = f.suffix.lower() in SUPPORTED_IMAGE_EXTS
            items_to_process.append((f, output_dir_default / f.name, is_img))

    else:
        target = Path(positional[0])
        if not target.exists():
            print(f"ERRORE: Il percorso specificato '{target}' non esiste.")
            return

        if target.is_dir():
            out_dir = Path(positional[1]) if len(positional) > 1 else output_dir_default
            out_dir.mkdir(parents=True, exist_ok=True)
            valid_files = sorted([
                f for f in target.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
            ])
            if not valid_files:
                print(f"Nessun file supportato trovato nella cartella '{target}'.")
                return
            for f in valid_files:
                is_img = f.suffix.lower() in SUPPORTED_IMAGE_EXTS
                items_to_process.append((f, out_dir / f.name, is_img))
        else:
            ext = target.suffix.lower()
            if ext not in SUPPORTED_EXTS:
                raise ValueError(f"Formato non supportato: {ext}. Formati supportati: {SUPPORTED_EXTS}")
            is_img = ext in SUPPORTED_IMAGE_EXTS

            if len(positional) > 1:
                dest = Path(positional[1])
                if dest.is_dir() or positional[1].endswith(('/', '\\')):
                    dest.mkdir(parents=True, exist_ok=True)
                    out_path = dest / target.name
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    out_path = dest
            else:
                out_path = output_dir_default / target.name

            items_to_process.append((target, out_path, is_img))

    det = (320, 320) if fast else (640, 640)
    ctx_id = 0 if use_gpu and not use_cpu else (-1 if use_cpu else 0)
    providers = None
    if use_gpu and not use_cpu:
        providers = ['CUDAExecutionProvider']
    elif use_cpu:
        providers = ['CPUExecutionProvider']

    # Initialize models once
    ok, avail = ort_available_and_providers()
    if use_gpu and (not ok or 'CUDAExecutionProvider' not in avail):
        print("[WARN] --gpu requested, but ONNX Runtime GPU is not available; falling back to CPU.")
        providers = ['CPUExecutionProvider']
        ctx_id = -1

    print("Initializing models..." + (" (fast mode)" if fast else "") + (" [GPU]" if ctx_id == 0 else " [CPU]"))
    with suppress_output():
        face_analysis, face_swapper = initialize_faceanalysis_and_swapper(
            det_size=det, ctx_id=ctx_id, swapper_providers=providers
        )

    sess = getattr(face_swapper, 'session', None) or getattr(face_swapper, 'model', None)
    active = get_session_providers(sess) if sess is not None else []
    if active:
        print(f"INSwapper active providers: {active}")
    else:
        if providers:
            print(f"INSwapper active providers unavailable; attempted: {providers}")

    total_items = len(items_to_process)
    if total_items > 1:
        print(f"\n--- Inizio elaborazione di {total_items} file ---")

    try:
        for idx, (src_file, dst_file, is_img) in enumerate(items_to_process, 1):
            if total_items > 1:
                print(f"\n[{idx}/{total_items}] Elaborazione: {src_file.name} -> {dst_file}")
            try:
                if is_img:
                    kirkify_image(str(src_file), str(dst_file), face_analysis, face_swapper)
                else:
                    kirkify_video(str(src_file), str(dst_file), face_analysis, face_swapper, frame_step=step, workers=workers, smooth=smooth)
            except Exception as e:
                print(f"  [ERRORE] Durante l'elaborazione di {src_file.name}: {e}")

        if total_items > 1:
            print(f"\n--- Elaborazione completata! Tutti i file si trovano in: {output_dir_default.resolve()} ---")
    except KeyboardInterrupt:
        cleanup('audio.aac')
        print("\nInterrotto dall'utente. Arrivederci!")

if __name__ == "__main__":
    main()