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

- `--scale N`: Upscale video/image resolution by factor N (e.g. `--scale 2` upscales 480p to HD using Lanczos scaling).
- `--fps N`: Custom output framerate (default: automatic minimum 60 FPS or original framerate if higher).
- `--crf N`: Video compression quality factor (default: 15, visually lossless).
- `--no-enhance`: Disable GFPGAN neural face restoration (GFPGAN is ON by default when GFPGANv1.4.onnx is present).
- `--gpu`: Prefer GPU for face detection, INSwapper, and GFPGAN.
- `--cpu`: Force CPU even if a GPU is present.
- `--no-smooth`: Disable temporal anti-flicker stabilization (temporal smoothing is ON by default).

Examples (cmd/PowerShell):

```bash
# Ultra-High Quality 60FPS Video with GFPGAN and 2x Resolution Upscale
python kirkify.py input.mp4 output.mp4 --gpu --scale 2

# Process entire input/ folder to output/ folder in HD 60FPS
python kirkify.py --gpu

# Process single photo in 4K/HD quality
python kirkify.py photo.jpg --gpu --scale 2

# Use a custom face/filter image or folder
python kirkify.py --gpu --face path/to/my_face.jpg
```

### Changing the Applied Face / Filter (Come Cambiare il Volto Applicato)

Puoi cambiare il volto che viene applicato in due modi semplici:

1. **Metodo 1: Sostituire le immagini nella cartella `kirks/` (Consigliato per batch/default)**
   - Apri la cartella `kirks/`.
   - Inserisci una o più foto (formati `.jpg`, `.png`, `.webp`) del nuovo volto che vuoi applicare (es. `mia_faccia.jpg`).
   - Se inserisci più foto con angolazioni o espressioni diverse della stessa persona, l'algoritmo calcola una media matematica dei vettori facciali (canonical identity embedding), rendendo il face-swap ancora più preciso e stabile.

2. **Metodo 2: Usare il parametro `--face` da riga di comando o in `.bat`**
   - Puoi specificare direttamente qualsiasi file immagine o cartella senza toccare `kirks/`:
     ```bash
     python kirkify.py --gpu --face mia_foto.jpg
     # oppure
     python kirkify.py input.mp4 output.mp4 --gpu --face volti_nuovi/
     ```


Notes:
- **Minimum 60 FPS Smooth Playback**: Output videos are automatically rendered at a minimum of 60 FPS (or the original framerate if higher) using lossless frame timing, completely eliminating lag and stutter.
- **HD Face Restoration (GFPGAN v1.4)**: Automatically upscales and restores the 128x128 swapped face to crystal-clear 512x512 resolution with sharp eyes, eyelashes, teeth, and natural skin pores.
- **Resolution Upscaling**: Use `--scale 2` to double the resolution of low-quality videos or images using high-quality Lanczos interpolation.
- **Anti-Flicker Temporal Stabilization**: Enabled by default for all video processing to eliminate micro-jitter and frame-dropping flickers.
- **Supporto Documenti PDF Multi-Pagina**: Elabora automaticamente file PDF (`.pdf`), eseguendo il face-swap su ogni pagina in cui sono presenti volti e preservando intatte le pagine vettoriali originali senza volti.
- Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tiff`, `.tif`.
- Supported video formats: `.mp4`, `.mov`, `.avi`, `.mkv`, `.flv`, `.wmv`, `.webm`, `.m4v`.
- Supported document formats: `.pdf` (richiede `pymupdf`).
- `ffmpeg` must be installed (automatically detected if installed via `winget install Gyan.FFmpeg.Essentials`).

