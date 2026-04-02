#!/usr/bin/env bash
set -e

# Guard — fail fast if key dependencies are missing
python -c "import torch, transformers, streamlit, zarr, fsspec, sklearn, skimage" 2>/dev/null || {
    echo "Missing dependencies. Set up your environment first:"
    echo "  python -m venv .venv && source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
}

gpu_stats() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu \
            --format=csv,noheader,nounits \
            | awk -F',' '{printf "  GPU: %s | VRAM: %s/%s MB | util: %s%%\n", $1,$2,$3,$4}'
    fi
}

run_step() {
    local label="$1"; shift
    echo ""
    echo "=== $label ==="
    gpu_stats
    "$@" &
    local pid=$!
    echo "  PID: $pid"
    wait $pid
    local exit_code=$?
    echo "  exit: $exit_code"
    gpu_stats
    # show the most recent log file
    local latest_log
    latest_log=$(ls -t logs/*.log 2>/dev/null | head -1)
    if [[ -n "$latest_log" ]]; then
        echo "  log: $latest_log"
    fi
    return $exit_code
}

run_step "Task 1 — Download datasets"            python download_datasets.py
run_step "Task 2 — Extract bilinear embeddings"  python extract_bilinear_embeddings.py
run_step "Task 4 — Train linear probe"           python train_linear_probe.py
run_step "Task 3 — Project embeddings"           python project_embeddings.py -p data/projection.pt

echo ""
echo "=== Streamlit dashboards ==="
streamlit run dashboard.py &
DASH_PID=$!
streamlit run visualize_embeddings.py &
VIZ_PID=$!
echo "  dashboard.py        PID: $DASH_PID"
echo "  visualize_embeddings.py PID: $VIZ_PID"
echo "  Press Ctrl+C to stop both"
wait $DASH_PID $VIZ_PID
