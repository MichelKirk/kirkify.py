# kirkify.py

Groundbreaking engine behind the hit website, kirkify.me, which some are saying is the most innovative software since Facebook or Google.

It's a relatively simple python script to kirkify any image or video given.

## Setup

It's a bit of a pain to setup but I promise it's worth it.

First download the face swapping AI (528mb)

```bash
curl -o "inswapper_128.onnx" https://bk4vz20t6s.ufs.sh/f/5eVwDsd8R3jL5kumGF8R3jLVwUJfdOu8cQ4ymMqAFeW7zrEX
```

Next, install the requirements.

```bash
pip install -r requirements.txt
```

After, initialize the script (you don't technically have to but it saves time on first run)

```bash
python3 kirkify.py init
```

Finally, if you don't already have it installed, the video kirkifier requires `ffmpeg` to be installed

```bash
apt install ffmpeg
```

### GPU Acceleration (Optional)

To run face detection and swapping on the GPU, you need ONNX Runtime with CUDA support and a compatible NVIDIA driver:

- NVIDIA GPU with recent driver (CUDA 12 compatible recommended)
- Install ONNX Runtime GPU wheels via pip

Basic installation:

```bash
pip install onnxruntime-gpu
```

If you still see missing CUDA DLLs, install the CUDA dependency wheels explicitly (CUDA 12 series):

```bash
pip install onnxruntime-gpu nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12
```

How to verify it's working:

```bash
python kirkify.py input.mp4 output.mp4 --gpu --fast
```

You should see a line like: `INSwapper active providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']`.
If CUDA is not available, the script will fall back to CPU and print a warning.

## Usage

### 1. Batch Mode (input/ -> output/)
Place all your videos and images inside the `input/` folder, then run:

```bash
python kirkify.py --gpu --fast
```
Every supported file will be processed and saved into the `output/` folder. The AI models are loaded once into memory, making batch processing fast.

### 2. Single File Mode
You can process a single image or video. If no output path is specified, it will be saved into the `output/` folder by default:

```bash
# Saves automatically to output/photo.jpg
python kirkify.py photo.jpg --gpu

# Or specify a custom output path
python kirkify.py photo.jpg result.jpg --gpu
```

### Performance and Device Flags

The script supports several flags to speed up processing and to select the compute device. These are optional and can be combined:

- `--fast`: Use a smaller face detector size (320x320) to speed up detection (slight quality trade-off).
- `--frame-step N`: Process 1 out of every N frames (video only). Frames are renumbered to keep the output sequence contiguous.
- `--workers M`: Use M threads to process frames in parallel (used when `--no-smooth` is specified).
- `--gpu`: Prefer GPU for face detection and INSwapper when available.
- `--cpu`: Force CPU even if a GPU is present.
- `--no-smooth`: Disable temporal anti-flicker stabilization (temporal smoothing is ON by default).

Examples (cmd/PowerShell):

```bash
# Fast mode using GPU with anti-flicker stabilization (default)
python kirkify.py input.mp4 output.mp4 --fast --gpu

# Process entire input/ folder to output/ folder with GPU
python kirkify.py --gpu --fast

# Skip frames to speed up
python kirkify.py input.mp4 output.mp4 --fast --frame-step 3 --gpu

# Force CPU and keep full quality detector
python kirkify.py input.mp4 output.mp4 --cpu
```

Notes:
- **Anti-Flicker Temporal Stabilization**: Enabled by default for all video processing. It uses adaptive exponential landmark smoothing (One Euro principle), missed-detection recovery (preventing 1-frame blink glitches), and multi-Kirk embedding averaging to produce smooth, jitter-free face swaps.
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.tif`.
- Supported video formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.flv`, `.wmv`, `.webm`, `.m4v`.
- GPU mode requires ONNX Runtime GPU (`onnxruntime-gpu`) installed and a compatible NVIDIA driver. If `--gpu` is requested but CUDA is not available, the script will fall back to CPU and print a warning.
- Videos without audio streams are automatically handled without errors.
- `ffmpeg` must be installed (automatically detected if installed via `winget install Gyan.FFmpeg.Essentials`).
