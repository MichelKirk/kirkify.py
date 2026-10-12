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
import cv2
import insightface
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import face_align
from cv2 import imread, imwrite, VideoCapture, CAP_PROP_FPS

# Constant directories
UNPROCESSED_DIR = Path('unprocessed_frames')
PROCESSED_DIR = Path('processed_frames')
KIRKS_DIR = Path('kirks')

SUPPORTED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif'}
SUPPORTED_VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
SUPPORTED_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_VIDEO_EXTS

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
    return float(fps) if fps and fps > 0 else 30.0

def extract_frames(video_path: str):
    """Extract all frames to unprocessed_frames/ with lossless quality"""
    UNPROCESSED_DIR.mkdir(exist_ok=True)
    run([
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', video_path,
        '-q:v', '1',
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

def reconstruct_video(fps: float, audio_path: str | None, output_path: str, target_fps: float | None = None, scale: float = 1.0, crf: int = 15):
    """
    Combine processed frames with audio.
    - target_fps: ensures output has minimum 60fps (or original if higher) without lag.
    - scale: upscales resolution (e.g. 2 for 2x upscale using high-quality Lanczos).
    - crf: visually lossless x264 quality (default 15).
    """
    PROCESSED_DIR.mkdir(exist_ok=True)
    input_fps = float(fps) if fps and fps > 0 else 30.0
    if target_fps is None or target_fps <= 0:
        target_fps = max(input_fps, 60.0)

    filters = []
    if scale and scale > 1.0:
        filters.append(f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2:flags=lanczos")
    if abs(target_fps - input_fps) > 0.05:
        filters.append(f"fps=fps={target_fps:.2f}")

    vf_str = ",".join(filters) if filters else None

    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-framerate', str(input_fps),
        '-i', str(PROCESSED_DIR / 'frame_%04d.png'),
    ]

    if audio_path and os.path.exists(audio_path):
        cmd.extend(['-i', audio_path])

    if vf_str:
        cmd.extend(['-vf', vf_str])

    cmd.extend([
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
    ])

    if audio_path and os.path.exists(audio_path):
        cmd.extend(['-c:a', 'copy', '-shortest'])

    cmd.append(output_path)
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

def get_canonical_kirk_face(faceanalysis: FaceAnalysis, source: Path | str = KIRKS_DIR):
    """
    Computes an averaged, normalized identity embedding from reference face images.
    `source` can be a single image file or a directory containing one or more images.
    Supports any image format (.jpg, .jpeg, .png, .webp, .bmp).
    This guarantees that the target face identity embedding remains constant and jitter-free.
    """
    source_p = Path(source)
    candidates: list[Path] = []

    if source_p.is_file():
        candidates.append(source_p)
    elif source_p.is_dir():
        # First priority: legacy kirk_0, kirk_1, kirk_2 if present
        for i in range(10):
            p = source_p / f'kirk_{i}.jpg'
            if p.exists() and p not in candidates:
                candidates.append(p)
        # Collect all valid image files in the directory
        for p in sorted(source_p.iterdir()):
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                if p not in candidates:
                    candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"Nessuna immagine trovata in '{source_p}'. Inserisci una o piu' foto (.jpg, .png, .webp) del volto da applicare.")

    embeddings = []
    best_face = None
    best_score = -1.0

    for img_path in candidates:
        img = imread(str(img_path))
        if img is None:
            continue
        faces = faceanalysis.get(img)
        if faces:
            # Pick highest confidence detected face in the reference photo
            face = max(faces, key=lambda f: getattr(f, 'det_score', 0.0))
            embeddings.append(face.normed_embedding)
            score = getattr(face, 'det_score', 0.0)
            if score > best_score:
                best_score = score
                best_face = face

    if best_face is None or not embeddings:
        raise RuntimeError(f"Impossibile rilevare un volto nelle immagini di riferimento in '{source_p}'. Assicurati che il viso sia ben visibile e illuminato.")

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
    def __init__(self, alpha: float = 0.85, max_missed: int = 2, iou_thresh: float = 0.25):
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

                    # Responsive tracking: follow head movement directly, filter only micro-jitter
                    diff = np.linalg.norm(det.kps - track['kps'])
                    if diff > 50:
                        a = 1.0
                    elif diff > 20:
                        a = 0.90
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

def restore_face_texture(swapped_crop: np.ndarray, orig_crop: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """
    Subtly recovers natural skin texture without harsh unsharp filter halos.
    """
    if swapped_crop.shape != orig_crop.shape:
        orig_crop = cv2.resize(orig_crop, (swapped_crop.shape[1], swapped_crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    orig_blur = cv2.GaussianBlur(orig_crop, (5, 5), 0).astype(np.float32)
    orig_detail = orig_crop.astype(np.float32) - orig_blur

    enhanced = swapped_crop.astype(np.float32) + strength * orig_detail
    return np.clip(enhanced, 0, 255).astype(np.uint8)

class GFPGANEnhancer:
    """
    High-Definition 512x512 Face Restoration model (GFPGAN v1.4 ONNX).
    Restores natural clarity to eyes and mouth, blended with original skin texture to prevent over-smoothing.
    """
    def __init__(self, model_path: str = "GFPGANv1.4.onnx", providers=None):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=providers or ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def enhance_face(self, img: np.ndarray, face, fidelity: float = 0.55) -> np.ndarray:
        crop_512, M = face_align.norm_crop2(img, face.kps, image_size=512)
        blob = (cv2.cvtColor(crop_512, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0)
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        pred = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        out_512 = np.clip((np.transpose(pred, (1, 2, 0)) + 1.0) * 127.5, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_512, cv2.COLOR_RGB2BGR)

        # Blend with swapped crop according to fidelity to preserve natural skin pores and avoid the porcelain/plastic mask effect
        if fidelity < 1.0:
            out_bgr = cv2.addWeighted(out_bgr, float(fidelity), crop_512, float(1.0 - fidelity), 0)

        mask = np.zeros((512, 512), dtype=np.float32)
        cv2.ellipse(mask, (256, 256), (180, 225), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (35, 35), 0)

        IM = cv2.invertAffineTransform(M)
        h, w = img.shape[:2]
        warped_face = cv2.warpAffine(out_bgr, IM, (w, h), borderValue=0.0)
        warped_mask = cv2.warpAffine(mask, IM, (w, h), borderValue=0.0)[..., np.newaxis]

        result = (img.astype(np.float32) * (1.0 - warped_mask) + warped_face.astype(np.float32) * warped_mask)
        return np.clip(result, 0, 255).astype(np.uint8)

def swap_and_enhance(
    img: np.ndarray,
    face,
    kirk_face,
    swapper,
    enhancer: GFPGANEnhancer | None = None,
    fidelity: float = 0.55,
    sharpen: float = 0.0
) -> np.ndarray:
    """Swaps face with INSwapper and applies balanced, natural enhancement without artificial over-sharpening."""
    res = swapper.get(img, face, kirk_face, paste_back=True)
    if enhancer is not None:
        try:
            res = enhancer.enhance_face(res, face, fidelity=fidelity)
        except Exception:
            pass
    elif sharpen > 0.0:
        try:
            bbox = face.bbox.astype(int)
            h, w = img.shape[:2]
            x1 = max(0, bbox[0])
            y1 = max(0, bbox[1])
            x2 = min(w, bbox[2])
            y2 = min(h, bbox[3])
            if x2 > x1 and y2 > y1:
                res[y1:y2, x1:x2] = restore_face_texture(res[y1:y2, x1:x2], img[y1:y2, x1:x2], strength=sharpen)
        except Exception:
            pass
    return res


def kirkify_frame(frame_path: str, output_path: str, faceanalysis: FaceAnalysis, swapper, kirk_face, enhancer=None, scale: float = 1.0):
    """Process a single frame with HD face enhancement"""
    img = imread(frame_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {frame_path}")
    faces = faceanalysis.get(img)

    if faces:
        res = img.copy()
        for face in faces:
            res = swap_and_enhance(res, face, kirk_face, swapper, enhancer=enhancer)
    else:
        res = img

    if scale > 1.0:
        h, w = res.shape[:2]
        res = cv2.resize(res, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    ext = Path(output_path).suffix.lower()
    if ext in ('.jpg', '.jpeg'):
        imwrite(output_path, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif ext == '.png':
        imwrite(output_path, res, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    else:
        imwrite(output_path, res)
    return bool(faces)

def process_all_frames(
    faceanalysis: FaceAnalysis,
    swapper,
    kirk_face,
    frame_step: int = 1,
    workers: int | None = None,
    smooth: bool = True,
    enhancer: GFPGANEnhancer | None = None,
    fidelity: float = 0.55,
    tracker_alpha: float = 0.85
) -> dict:
    """Process each frame in unprocessed_frames/ with tqdm progress bar.
    - smooth: enables temporal face tracking and landmark smoothing to eliminate flickering.
    - frame_step: process 1 of every N frames (skipped frames smoothly held to keep 60fps fluid).
    - enhancer: GFPGAN neural face restoration model.
    - fidelity: blending strength of GFPGAN (0.5-0.6 keeps natural skin texture).
    - tracker_alpha: responsiveness of face tracking (0.85 = crisp and responsive).
    """
    PROCESSED_DIR.mkdir(exist_ok=True)
    all_files = sorted([f for f in os.listdir(UNPROCESSED_DIR) if f.endswith('.png')])
    total = len(all_files)
    with_faces = 0

    if total == 0:
        return {'total': 0, 'processed': 0, 'with_faces': 0}

    step = max(1, frame_step)
    last_processed_img = None

    if smooth:
        tracker = FaceTracker(alpha=tracker_alpha, max_missed=2)
        desc = "Processing frames (60fps HD)"
        for idx, filename in enumerate(tqdm(all_files, total=total, desc=desc, unit="frame")):
            output_path = str(PROCESSED_DIR / f"frame_{idx:04d}.png")
            input_path = str(UNPROCESSED_DIR / filename)

            if step > 1 and (idx % step != 0) and last_processed_img is not None:
                imwrite(output_path, last_processed_img)
                continue

            img = imread(input_path)
            if img is None:
                continue

            detected_faces = faceanalysis.get(img)
            stabilized_faces = tracker.update(detected_faces)

            if stabilized_faces:
                with_faces += 1
                res = img.copy()
                for face in stabilized_faces:
                    res = swap_and_enhance(res, face, kirk_face, swapper, enhancer=enhancer, fidelity=fidelity)
                imwrite(output_path, res)
                last_processed_img = res
            else:
                imwrite(output_path, img)
                last_processed_img = img

    else:
        desc = "Processing frames (parallel)"
        pbar = tqdm(total=total, desc=desc, unit="frame")

        def _process(idx_and_name: tuple[int, str]):
            nonlocal with_faces
            idx, filename = idx_and_name
            input_path = str(UNPROCESSED_DIR / filename)
            output_path = str(PROCESSED_DIR / f"frame_{idx:04d}.png")
            img = imread(input_path)
            if img is None:
                pbar.update(1)
                return
            faces = faceanalysis.get(img)
            if faces:
                with_faces += 1
                res = img.copy()
                for face in faces:
                    res = swap_and_enhance(res, face, kirk_face, swapper, enhancer=enhancer, fidelity=fidelity)
                imwrite(output_path, res)
            else:
                imwrite(output_path, img)
            pbar.update(1)

        workers = workers or max(1, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_process, enumerate(all_files)))
        pbar.close()

    return {'total': total, 'processed': total, 'with_faces': with_faces}

def kirkify_video(
    TARGET_PATH,
    OUTPUT_PATH,
    FACE_ANALYSIS,
    FACE_SWAPPER,
    frame_step: int = 1,
    workers: int | None = None,
    smooth: bool = True,
    enhancer: GFPGANEnhancer | None = None,
    target_fps: float | None = None,
    scale: float = 1.0,
    crf: int = 15,
    kirk_face: Face | None = None,
    source_face: Path | str = KIRKS_DIR,
    fidelity: float = 0.55,
    tracker_alpha: float = 0.85,
):
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    audio_path = None
    if kirk_face is None:
        kirk_face = get_canonical_kirk_face(FACE_ANALYSIS, source=source_face)
    try:
        print("Extracting frames (lossless)...")
        extract_frames(TARGET_PATH)
        
        print("Extracting audio...")
        audio_path = extract_audio(TARGET_PATH)
        
        fps = get_video_fps(TARGET_PATH)
        effective_fps = max(fps, 60.0) if (target_fps is None or target_fps <= 0) else target_fps
        print(f"Video input: {fps:.2f} FPS -> Target: {effective_fps:.2f} FPS" + (f" (Lanczos {scale:.1f}x Upscale)" if scale > 1.0 else "") + (" (GFPGAN HD Face Enhancer ON)" if enhancer else " (Texture Restoration ON)"))

        print("Processing frames..." + (" (temporal stabilization ON)" if smooth else " (stabilization OFF)"))
        stats = process_all_frames(
            FACE_ANALYSIS,
            FACE_SWAPPER,
            kirk_face,
            frame_step=frame_step,
            workers=workers,
            smooth=smooth,
            enhancer=enhancer,
            fidelity=fidelity,
            tracker_alpha=tracker_alpha
        )
        print(f"  Frames: total {stats['total']}, with faces {stats['with_faces']}")
        
        print("Reconstructing video (min 60fps, visually lossless CRF 15)...")
        reconstruct_video(fps, audio_path, OUTPUT_PATH, target_fps=effective_fps, scale=scale, crf=crf)
    finally:
        print("Cleaning up...")
        cleanup(audio_path)
    
    print(f"Done! Output saved to {OUTPUT_PATH}")

def kirkify_image(
    TARGET_PATH,
    OUTPUT_PATH,
    FACE_ANALYSIS,
    FACE_SWAPPER,
    enhancer: GFPGANEnhancer | None = None,
    scale: float = 1.0,
    kirk_face: Face | None = None,
    source_face: Path | str = KIRKS_DIR,
    fidelity: float = 0.55,
):
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    if kirk_face is None:
        kirk_face = get_canonical_kirk_face(FACE_ANALYSIS, source=source_face)

    print("Kirkifying image in ultra-high quality...")
    img = imread(TARGET_PATH)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {TARGET_PATH}")

    faces = FACE_ANALYSIS.get(img)
    if faces:
        res = img.copy()
        for face in faces:
            res = swap_and_enhance(res, face, kirk_face, FACE_SWAPPER, enhancer=enhancer, fidelity=fidelity)
    else:
        print("  No faces detected. Image copied unchanged.")
        res = img


    if scale > 1.0:
        h, w = res.shape[:2]
        res = cv2.resize(res, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

    ext = Path(OUTPUT_PATH).suffix.lower()
    if ext in ('.jpg', '.jpeg'):
        imwrite(OUTPUT_PATH, res, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif ext == '.png':
        imwrite(OUTPUT_PATH, res, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])
    else:
        imwrite(OUTPUT_PATH, res)

    print(f"Done! Output saved to {OUTPUT_PATH} ({res.shape[1]}x{res.shape[0]})")

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
    enable_enhance = '--no-enhance' not in args
    step = 1
    workers = None
    scale = 1.0
    custom_fps = None
    crf = 15
    source_face_path = KIRKS_DIR
    fidelity = 0.55
    tracker_alpha = 0.85

    positional = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ('--fast', '--gpu', '--cpu', '--no-smooth', '--enhance', '--no-enhance'):
            i += 1
        elif arg in ('--face', '--source', '--source-face'):
            if i + 1 < len(args) and not args[i+1].startswith('-'):
                source_face_path = Path(args[i+1])
                i += 2
            else:
                i += 1
        elif arg in ('--fidelity', '--blend', '--strength'):
            if i + 1 < len(args):
                try:
                    fidelity = max(0.0, min(1.0, float(args[i+1])))
                except ValueError:
                    pass
                i += 2
            else:
                i += 1
        elif arg == '--tracker-alpha':
            if i + 1 < len(args):
                try:
                    tracker_alpha = max(0.1, min(1.0, float(args[i+1])))
                except ValueError:
                    pass
                i += 2
            else:
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
        elif arg in ('--scale', '--upscale'):
            if i + 1 < len(args) and not args[i+1].startswith('-'):
                try:
                    scale = max(1.0, float(args[i+1]))
                    i += 2
                except ValueError:
                    scale = 2.0
                    i += 1
            else:
                scale = 2.0
                i += 1
        elif arg == '--fps':
            if i + 1 < len(args):
                try:
                    custom_fps = float(args[i+1])
                except ValueError:
                    pass
                i += 2
            else:
                i += 1
        elif arg == '--crf':
            if i + 1 < len(args):
                try:
                    crf = int(args[i+1])
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

    det = (640, 640)
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

    print("Initializing models..." + (" [GPU]" if ctx_id == 0 else " [CPU]"))
    with suppress_output():
        face_analysis, face_swapper = initialize_faceanalysis_and_swapper(
            det_size=det, ctx_id=ctx_id, swapper_providers=providers
        )

    sess = getattr(face_swapper, 'session', None) or getattr(face_swapper, 'model', None)
    active = get_session_providers(sess) if sess is not None else []
    if active:
        print(f"INSwapper active providers: {active}")

    # Initialize GFPGAN Face Enhancer if present
    enhancer = None
    gfpgan_path = Path("GFPGANv1.4.onnx")
    if enable_enhance and gfpgan_path.exists():
        try:
            print("Loading GFPGAN v1.4 HD Face Restoration model...")
            enhancer = GFPGANEnhancer(str(gfpgan_path), providers=providers)
            print("GFPGAN HD Face Restoration active! [512x512 Neural Clarity]")
        except Exception as e:
            print(f"[WARN] Could not load GFPGAN ({e}); using high-frequency texture restoration.")
    else:
        print("[INFO] High-frequency texture & unsharp detail restoration active.")

    print(f"Loading reference face from '{source_face_path}'...")
    canonical_kirk_face = get_canonical_kirk_face(face_analysis, source=source_face_path)
    print("Reference face loaded successfully!")

    total_items = len(items_to_process)
    if total_items > 1:
        print(f"\n--- Inizio elaborazione di {total_items} file (minimo 60fps per i video) ---")

    try:
        for idx, (src_file, dst_file, is_img) in enumerate(items_to_process, 1):
            if total_items > 1:
                print(f"\n[{idx}/{total_items}] Elaborazione: {src_file.name} -> {dst_file}")
            try:
                if is_img:
                    kirkify_image(
                        str(src_file),
                        str(dst_file),
                        face_analysis,
                        face_swapper,
                        enhancer=enhancer,
                        scale=scale,
                        kirk_face=canonical_kirk_face,
                        fidelity=fidelity
                    )
                else:
                    kirkify_video(
                        str(src_file),
                        str(dst_file),
                        face_analysis,
                        face_swapper,
                        frame_step=step,
                        workers=workers,
                        smooth=smooth,
                        enhancer=enhancer,
                        target_fps=custom_fps,
                        scale=scale,
                        crf=crf,
                        kirk_face=canonical_kirk_face,
                        fidelity=fidelity,
                        tracker_alpha=tracker_alpha
                    )

            except Exception as e:
                print(f"  [ERRORE] Durante l'elaborazione di {src_file.name}: {e}")


        if total_items > 1:
            print(f"\n--- Elaborazione completata! Tutti i file si trovano in: {output_dir_default.resolve()} ---")
    except KeyboardInterrupt:
        cleanup('audio.aac')
        print("\nInterrotto dall'utente. Arrivederci!")

if __name__ == "__main__":
    main()