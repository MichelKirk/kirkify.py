import os
from pathlib import Path
from typing import Optional, Callable
from app.config import settings

# Lazy model singletons
_face_analysis = None
_face_swapper = None
_enhancer = None
_initialized = False

def get_engine():
    global _face_analysis, _face_swapper, _enhancer, _initialized
    if not _initialized:
        import kirkify

        providers = None
        ctx_id = 0 if settings.USE_GPU else -1
        if settings.USE_GPU:
            ok, avail = kirkify.ort_available_and_providers()
            if ok and 'CUDAExecutionProvider' in avail:
                providers = ['CUDAExecutionProvider']
                ctx_id = 0
            else:
                ctx_id = -1
                providers = ['CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        _face_analysis, _face_swapper = kirkify.initialize_faceanalysis_and_swapper(
            det_size=(640, 640),
            ctx_id=ctx_id,
            swapper_providers=providers
        )

        gfpgan_path = settings.BASE_DIR / "GFPGANv1.4.onnx"
        if settings.ENHANCE_FACE and gfpgan_path.exists():
            try:
                _enhancer = kirkify.GFPGANEnhancer(str(gfpgan_path), providers=providers)
            except Exception as e:
                print(f"[WARN] Could not initialize GFPGAN ({e})")
                _enhancer = None

        _initialized = True

    return _face_analysis, _face_swapper, _enhancer


def process_media(
    input_path: str,
    output_path: str,
    custom_face_source: Optional[str] = None,
    scale: Optional[float] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> str:
    """
    Processes a video or image through the Kirkify engine.
    Applies Face-Swap, GFPGAN v1.4 HD enhancement, and 60 FPS reconstruction.
    """
    import kirkify

    face_analysis, face_swapper, enhancer = get_engine()

    source = custom_face_source or str(settings.KIRKS_DIR)
    canonical_face = kirkify.get_canonical_kirk_face(face_analysis, source=source)

    effective_scale = scale if scale is not None else settings.SCALE
    ext = Path(input_path).suffix.lower()

    if progress_callback:
        progress_callback(10.0, "Face analysis e preparazione...")

    if ext in kirkify.SUPPORTED_IMAGE_EXTS:
        if progress_callback:
            progress_callback(30.0, "Elaborazione immagine...")
        kirkify.kirkify_image(
            TARGET_PATH=input_path,
            OUTPUT_PATH=output_path,
            FACE_ANALYSIS=face_analysis,
            FACE_SWAPPER=face_swapper,
            enhancer=enhancer,
            scale=effective_scale,
            kirk_face=canonical_face
        )
        if progress_callback:
            progress_callback(100.0, "Completato!")
    elif ext in kirkify.SUPPORTED_DOCUMENT_EXTS:
        if progress_callback:
            progress_callback(30.0, "Elaborazione documento PDF...")
        kirkify.kirkify_pdf(
            TARGET_PATH=input_path,
            OUTPUT_PATH=output_path,
            FACE_ANALYSIS=face_analysis,
            FACE_SWAPPER=face_swapper,
            enhancer=enhancer,
            scale=effective_scale,
            kirk_face=canonical_face
        )
        if progress_callback:
            progress_callback(100.0, "Completato!")
    elif ext in kirkify.SUPPORTED_VIDEO_EXTS:
        if progress_callback:
            progress_callback(20.0, "Elaborazione video (60 FPS HD)...")
        kirkify.kirkify_video(
            TARGET_PATH=input_path,
            OUTPUT_PATH=output_path,
            FACE_ANALYSIS=face_analysis,
            FACE_SWAPPER=face_swapper,
            frame_step=1,
            workers=settings.WORKERS,
            smooth=True,
            enhancer=enhancer,
            target_fps=settings.MIN_FPS,
            scale=effective_scale,
            crf=settings.CRF,
            kirk_face=canonical_face
        )
        if progress_callback:
            progress_callback(100.0, "Completato!")
    else:
        raise ValueError(f"Formato non supportato: {ext}")


    return output_path

