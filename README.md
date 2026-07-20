# TRELLIS-BOX Docker Guide

This repository contains two containerized TRELLIS projects:

- `TRELLIS/`: the original TRELLIS image-to-3D app, optimized here for lower VRAM usage and GLB export. It runs a Streamlit UI on port `8501`.
- `TRELLIS.2/`: TRELLIS.2, the newer 4B O-Voxel/PBR pipeline. It runs a Gradio UI on port `7860` and includes low-VRAM flow block offload by default.

Both Docker paths assume an NVIDIA GPU and `nvidia-container-toolkit` are installed on the host.

## Requirements

- Docker with BuildKit support.
- NVIDIA driver compatible with CUDA 12.x.
- `nvidia-container-toolkit`.
- Enough disk space for CUDA layers, compiled extensions, and Hugging Face model caches.

Quick GPU check:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

Shared model caches are recommended so model downloads survive container rebuilds:

```bash
mkdir -p ~/.cache/huggingface ~/.cache/rembg
```

## Which Version Should I Run?

Use `TRELLIS/` if you want the original optimized TRELLIS workflow with the existing Streamlit UI and lower baseline VRAM requirements.

Use `TRELLIS.2/` if you want the newer TRELLIS.2 4B O-Voxel pipeline with higher-resolution PBR assets. It is heavier, but this repo enables low-VRAM flow block offload by default.

You can run both at the same time because they use different host ports by default:

- TRELLIS: http://localhost:8501
- TRELLIS.2: http://localhost:7860

## Run TRELLIS

The easiest path is Docker Compose from inside the `TRELLIS/` directory:

```bash
cd TRELLIS
cp docker.env.example .env
docker compose up --build
```

Open http://localhost:8501.

You can also run it directly with Docker:

```bash
cd TRELLIS
DOCKER_BUILDKIT=1 docker build -t trellis-v1 .

docker run --rm --gpus all \
  -p 8501:8501 \
  -v "$HOME/.cache/trellis:/home/appuser/.cache" \
  -v "$HOME/.cache/huggingface:/home/appuser/.cache/huggingface" \
  -v "$HOME/.cache/rembg:/app/rembg_cache" \
  -v "$(pwd)/outputs:/app/outputs" \
  trellis-v1
```

Useful TRELLIS helper commands:

```bash
cd TRELLIS
./trellis.sh run
./trellis.sh run --dev
./trellis.sh status
./trellis.sh stop
```

## Run TRELLIS.2

Build the TRELLIS.2 image:

```bash
cd TRELLIS.2
DOCKER_BUILDKIT=1 docker build -t trellis2-lowvram .
```

Run the image-to-3D Gradio app:

```bash
docker run --rm --gpus all \
  -p 7860:7860 \
  -v "$HOME/.cache/huggingface:/home/trellis/.cache/huggingface" \
  -v "$(pwd)/tmp:/app/tmp" \
  -v "$(pwd)/outputs:/app/outputs" \
  trellis2-lowvram
```

Open http://localhost:7860.

Run the TRELLIS.2 texturing app instead:

```bash
cd TRELLIS.2
docker run --rm --gpus all \
  -p 7860:7860 \
  -v "$HOME/.cache/huggingface:/home/trellis/.cache/huggingface" \
  -v "$(pwd)/tmp:/app/tmp" \
  -v "$(pwd)/outputs:/app/outputs" \
  trellis2-lowvram python app_texturing.py
```

## Low-VRAM Defaults

`TRELLIS.2/Dockerfile` enables these defaults:

```bash
TRELLIS2_FLOW_BLOCK_OFFLOAD=1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_MODULE_LOADING=LAZY
ATTN_BACKEND=flash_attn
```

`TRELLIS2_FLOW_BLOCK_OFFLOAD=1` streams the largest flow transformer blocks onto the GPU one at a time during low-VRAM inference. It preserves model weights, sampling settings, resolution, and output functionality, but it can make sampling slower.

To disable TRELLIS.2 block offload for speed on a large GPU:

```bash
docker run --rm --gpus all \
  -e TRELLIS2_FLOW_BLOCK_OFFLOAD=0 \
  -p 7860:7860 \
  -v "$HOME/.cache/huggingface:/home/trellis/.cache/huggingface" \
  trellis2-lowvram
```

## GPU Architecture Build Args

Both projects compile CUDA extensions. To speed builds, set `TORCH_CUDA_ARCH_LIST` for your GPU:

```bash
# RTX 3090 / A5000 class
DOCKER_BUILDKIT=1 docker build --build-arg TORCH_CUDA_ARCH_LIST="8.6" -t trellis2-lowvram .

# RTX 4090 / Ada
DOCKER_BUILDKIT=1 docker build --build-arg TORCH_CUDA_ARCH_LIST="8.9" -t trellis2-lowvram .

# A100
DOCKER_BUILDKIT=1 docker build --build-arg TORCH_CUDA_ARCH_LIST="8.0" -t trellis2-lowvram .

# H100
DOCKER_BUILDKIT=1 docker build --build-arg TORCH_CUDA_ARCH_LIST="9.0" -t trellis2-lowvram .
```

Run those commands from the project directory you are building, either `TRELLIS/` or `TRELLIS.2/`.

## Ports

Default ports:

| Project | Container port | Host URL |
| --- | ---: | --- |
| `TRELLIS/` | `8501` | http://localhost:8501 |
| `TRELLIS.2/` | `7860` | http://localhost:7860 |

To run two copies of the same app, change only the host-side port:

```bash
docker run --rm --gpus all -p 7861:7860 trellis2-lowvram
```

## Updating Images

After changing code or Dockerfiles:

```bash
cd TRELLIS
docker compose build --no-cache

cd ../TRELLIS.2
DOCKER_BUILDKIT=1 docker build --no-cache -t trellis2-lowvram .
```

Model downloads remain cached if you keep the `~/.cache/huggingface` bind mount.

## Troubleshooting

If Docker cannot see the GPU, re-check the NVIDIA runtime:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

If builds fail while compiling CUDA extensions, lower parallelism:

```bash
DOCKER_BUILDKIT=1 docker build --build-arg MAX_JOBS=2 -t trellis2-lowvram TRELLIS.2
```

If model downloads fail or are slow, pass a Hugging Face token:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/home/trellis/.cache/huggingface" \
  -p 7860:7860 \
  trellis2-lowvram
```

If TRELLIS.2 runs out of VRAM, keep `TRELLIS2_FLOW_BLOCK_OFFLOAD=1`, use the UI's lower resolution option, and close other GPU workloads. The low-VRAM path avoids quality-reducing internal shortcuts; user-selected resolution and sampling settings still control the requested output.
