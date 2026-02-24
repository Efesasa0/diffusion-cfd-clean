#!/bin/bash
# Training script for physics-informed diffusion CFD model

# Default values
CONFIG="configs/default.yaml"
DEVICE="cuda"
SEED=42

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

# Build command
CMD="python main.py --config $CONFIG --mode train --device $DEVICE --seed $SEED"

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

echo "Running: $CMD"
$CMD
