#!/usr/bin/env bash
# =============================================================
#  run_visualizer.sh  —  Launch the Gradio retinal vessel analyzer
#
#  The user provides their trained .pth by bind-mounting it to
#  /model/checkpoint.pth (the canonical sentinel path).
#  Optionally they can override with --model PATH.
#
#  Usage:
#    apptainer exec --nv \
#      --bind /host/checkpoint.pth:/model/checkpoint.pth \
#      --bind /host/results:/output \
#      unet.sif visualize
#
#    # or explicit path:
#    apptainer exec --nv unet.sif visualize --model /my/model.pth --port 7860
# =============================================================
set -euo pipefail

MODEL_PATH="/model/checkpoint.pth"   # default sentinel (bind-mount target)
PORT=7860
HOST="0.0.0.0"
SHARE=false

usage() {
cat << EOF
Usage: apptainer exec [--nv] unet.sif visualize [OPTIONS]

Launches the Gradio retinal vessel analysis interface.

MODEL  (provide one):
  Bind your .pth to the sentinel path (recommended):
    --bind /host/checkpoint.pth:/model/checkpoint.pth

  Or pass explicitly:
  --model PATH    Path to trained checkpoint.pth inside container

SERVER:
  --port  N       Gradio port            [default: 7860]
  --host  ADDR    Bind address           [default: 0.0.0.0]
  --share         Enable Gradio share link

EXAMPLE:
  apptainer exec --nv \\
    --bind /results/checkpoint.pth:/model/checkpoint.pth \\
    unet.sif visualize

  # Then open http://<node-ip>:7860 in your browser
  # or use SSH port-forwarding:
  #   ssh -L 7860:localhost:7860 <hpc-login-node>
EOF
exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL_PATH="$2"; shift 2 ;;
    --port)   PORT="$2";       shift 2 ;;
    --host)   HOST="$2";       shift 2 ;;
    --share)  SHARE=true;      shift   ;;
    --help|-h) usage ;;
    *) echo "[ERROR] Unknown argument: $1"; usage ;;
  esac
done

if [[ ! -f "$MODEL_PATH" ]]; then
    echo ""
    echo "[ERROR] Model checkpoint not found at: $MODEL_PATH"
    echo ""
    echo "  Provide your trained .pth via bind-mount:"
    echo "    --bind /host/path/to/checkpoint.pth:/model/checkpoint.pth"
    echo "  or pass --model /path/inside/container/checkpoint.pth"
    echo ""
    exit 1
fi

echo ""
echo "================================================================="
echo "  Retinal Vessel Analyzer  |  Gradio UI"
echo "================================================================="
echo "  Model checkpoint : $MODEL_PATH"
echo "  Server           : http://${HOST}:${PORT}"
echo "================================================================="
echo ""
echo "  If running on HPC, forward the port to your local machine:"
echo "    ssh -L ${PORT}:localhost:${PORT} <login-node>"
echo "  Then open http://localhost:${PORT} in your browser."
echo ""

# Export for vessel_analyzer.py to pick up
export UNET_CHECKPOINT_PATH="$MODEL_PATH"
export GRADIO_SERVER_PORT="$PORT"
export GRADIO_SERVER_NAME="$HOST"

SHARE_FLAG="False"
$SHARE && SHARE_FLAG="True"

cd /app/main/visualizers

# Patch checkpoint_path at runtime via env var injection
python3 - << PYEOF
import os, sys
sys.path.insert(0, "/app/main")
sys.path.insert(0, "/app/main/visualizers")

# Override the hardcoded checkpoint_path before vessel_analyzer is imported
import vessel_analyzer
vessel_analyzer.checkpoint_path = os.environ["UNET_CHECKPOINT_PATH"]

from visualizer import create_gradio_interface
import gradio as gr

print(f"[INFO] Loading model from: {vessel_analyzer.checkpoint_path}")
demo = create_gradio_interface()
demo.launch(
    share=${SHARE_FLAG},
    server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
    server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    theme=gr.themes.Soft(),
)
PYEOF
