"""
Main entry point for physics-informed diffusion CFD model.

Usage:
    # Training
    python main.py --config configs/default.yaml --mode train

    # Sampling/Inference
    python main.py --config configs/default.yaml --mode sample --checkpoint checkpoints/final_model.pth

    # With custom device
    python main.py --config configs/default.yaml --mode train --device cuda:0
"""

import sys
from utils import get_arg_parser, Config


def main():
    parser = get_arg_parser()
    args = parser.parse_args()

    # Load config
    config = Config.from_yaml(args.config)

    if args.mode == "train":
        from train import train, set_seed
        import torch

        set_seed(args.seed)

        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = "cpu"

        train(config, device, args.checkpoint)

    elif args.mode == "sample":
        from sample import sample, set_seed
        import torch

        if args.checkpoint is None:
            print("Error: --checkpoint is required for sampling")
            sys.exit(1)

        set_seed(args.seed)

        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = "cpu"

        sample(config, device, args.checkpoint)

    else:
        print(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
