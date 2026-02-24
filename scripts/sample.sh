#!/bin/bash
# Sampling script for physics-informed diffusion CFD model

# Default values
CONFIG="configs/default.yaml"
DEVICE="cuda"
SEED=42
CHECKPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check checkpoint
if [ -z "$CHECKPOINT" ]; then
    echo "Error: --checkpoint is required"
    echo "Usage: ./sample.sh --checkpoint path/to/checkpoint.pth [--config CONFIG] [--device DEVICE] [--seed SEED]"
    exit 1
fi

# Build command
CMD="python main.py --config $CONFIG --mode sample --device $DEVICE --seed $SEED --checkpoint $CHECKPOINT"

echo "Running: $CMD"
$CMD
