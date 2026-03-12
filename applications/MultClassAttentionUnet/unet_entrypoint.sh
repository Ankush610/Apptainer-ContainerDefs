#!/usr/bin/env bash
# =============================================================
#  unet_entrypoint.sh  —  Main dispatcher for the UNet container
#
#  Dispatches to the correct mode based on first argument:
#    train      → single-GPU training
#    train_ddp  → multi-GPU training via Accelerate
#    visualize  → Gradio vessel analyzer UI
#    bash       → interactive shell
#
#  This script is set as %runscript in the Apptainer .def file.
#  It is also placed in PATH as individual commands:
#    /usr/local/bin/train
#    /usr/local/bin/train_ddp
#    /usr/local/bin/visualize
# =============================================================
set -euo pipefail

MODE="${1:-help}"
shift 2>/dev/null || true

case "$MODE" in

  # ── Single-GPU training ────────────────────────────────────────────────────
  train)
    cd /app/main
    exec python3 /app/main/train_cli.py "$@"
    ;;

  # ── Multi-GPU DDP training via HuggingFace Accelerate ────────────────────
  train_ddp)
    cd /app/main
    # accelerate launch handles torchrun / NCCL init automatically.
    # Pass --num_processes to override GPU count (default: all visible GPUs).
    exec accelerate launch /app/main/train_ddp_cli.py "$@"
    ;;

  # ── Gradio visualizer ─────────────────────────────────────────────────────
  visualize)
    exec /app/run_visualizer.sh "$@"
    ;;

  # ── Interactive shell ──────────────────────────────────────────────────────
  bash|shell)
    exec /bin/bash "$@"
    ;;

  # ── Help ──────────────────────────────────────────────────────────────────
  help|--help|-h|"")
    cat << 'EOF'
=================================================================
 MultiClass Attention-UNet Container
 Modes: train | train_ddp | visualize | bash
=================================================================

SINGLE-GPU TRAINING
  apptainer run --nv \
    --bind /data:/data --bind /results:/output \
    unet.sif train \
    --data /data --output /output \
    --epochs 500 --batch 4 --lr 5e-4

MULTI-GPU TRAINING (DDP via Accelerate)
  apptainer run --nv \
    --bind /data:/data --bind /results:/output \
    unet.sif train_ddp \
    --data /data --output /output \
    --epochs 500 --batch 4

  # Control GPU count via env var before running:
  export CUDA_VISIBLE_DEVICES=0,1,2,3

VISUALIZER (Gradio UI — requires trained checkpoint)
  apptainer run --nv \
    --bind /results/checkpoint.pth:/model/checkpoint.pth \
    unet.sif visualize

  # Override model path:
  apptainer run --nv \
    unet.sif visualize --model /path/to/checkpoint.pth

INTERACTIVE SHELL
  apptainer run unet.sif bash

MODE HELP
  apptainer run unet.sif train     --help
  apptainer run unet.sif train_ddp --help
  apptainer run unet.sif visualize --help
=================================================================
EOF
    exit 0
    ;;

  *)
    echo "[ERROR] Unknown mode: '$MODE'"
    echo "Valid modes: train | train_ddp | visualize | bash | help"
    exit 1
    ;;

esac
