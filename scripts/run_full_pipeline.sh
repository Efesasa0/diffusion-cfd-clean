#!/bin/bash
# Full pipeline: Train model and run sampling
# Usage: ./run_full_pipeline.sh [--epochs N] [--quick]

set -e  # Exit on error

cd "$(dirname "$0")/.."

# Defaults
EPOCHS=100
QUICK=false
DEVICE="cuda"
CONFIG="configs/default.yaml"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --quick)
            QUICK=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Quick mode for testing (2 epochs, fewer samples)
if [ "$QUICK" = true ]; then
    EPOCHS=2
    echo "=== QUICK MODE: $EPOCHS epochs ==="
fi

echo "============================================"
echo "Physics-Informed Diffusion CFD Pipeline"
echo "============================================"
echo "Device: $DEVICE"
echo "Epochs: $EPOCHS"
echo "Config: $CONFIG"
echo ""

# Create directories
mkdir -p checkpoints logs outputs

# Step 1: Training
echo "=== STEP 1: Training ($EPOCHS epochs) ==="
python main.py \
    --config "$CONFIG" \
    --mode train \
    --device "$DEVICE" \
    --seed 42

echo ""
echo "Training complete! Checkpoint saved to checkpoints/"

# Step 2: Sampling with trained model
CHECKPOINT="checkpoints/final_model.pth"

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo ""
echo "=== STEP 2: Sampling/Reconstruction ==="
python main.py \
    --config "$CONFIG" \
    --mode sample \
    --device "$DEVICE" \
    --checkpoint "$CHECKPOINT" \
    --seed 42

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo "Results saved to:"
echo "  - Checkpoints: checkpoints/"
echo "  - Logs: logs/"
echo "  - Outputs: outputs/"
echo "    - outputs/results.npz (predictions, targets, inputs)"
echo "    - outputs/visualization_*.png"
