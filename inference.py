#!/usr/bin/env python3
import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add sam2-main to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2-main'))

# Import SAM2 data structures for video processing
from training.utils.data_utils import VideoDatapoint, Frame, Object

# Import the embedding model
from models.model_single import ModelEmb

# Import SAM2 image predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Import polyp_gen dataset functions via importlib
import importlib.util
polyp_gen_path = os.path.join(os.path.dirname(__file__), 'dataset', 'polyp_gen.py')
spec = importlib.util.spec_from_file_location("polyp_gen", polyp_gen_path)
polyp_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(polyp_gen)
get_polyp_gen_dataset = polyp_gen.get_polyp_gen_dataset
VideoDataLoader = polyp_gen.VideoDataLoader

# Change PyTorch cache location if needed
os.environ['TORCH_HOME'] = '/tmp/torch_cache'

# Import evaluation functions from train.py (assumes train.py is importable)
from train import evaluate, CombinedBCEDiceLoss, prepare_batch_for_sam2, sam_call,visualize_predictions
def custom_collate_fn(batch):
    return batch

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the validation dataset
    print("Loading validation dataset...")
    _, val_dataset = get_polyp_gen_dataset(sam_trans=None, base_path=args.data_dir)
    # (If you have transforms you want to apply at inference, adjust here.)
    val_dataset.transform = None  # Use original images/masks
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=custom_collate_fn
    )
    
    # Load the embedding model (ModelEmb) and load pretrained weights
    print("Loading ModelEmb...")
    model_args = {
        'order': 85,            # HarDNet-85 architecture (larger variant)
        'depth_wise': 0,        # Regular convolutions
        'nP': 8,
        'output_dim': 256
    }
    model = ModelEmb(args=model_args).to(device)
    if args.pretrained:
        print(f"Loading pretrained ModelEmb weights from {args.pretrained}")
        with torch.serialization.safe_globals([np._core.multiarray.scalar]):
            checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Load the SAM2 image model from local checkpoints
    print("Loading SAM2 image model...")
    sam2_dir = os.path.join(os.path.dirname(__file__), 'sam2-main')
    checkpoint_path = os.path.join(sam2_dir, 'checkpoints', 'sam2.1_hiera_large.pt')
    config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
    from sam2.build_sam import build_sam2
    sam2_model = build_sam2(config_file=config_path, ckpt_path=checkpoint_path, device=device)
    sam2_image = SAM2ImagePredictor(sam_model=sam2_model)
    for param in sam2_image.model.parameters():
        param.requires_grad = False
    sam2_image.model.eval()
    
    # Set up loss criterion exactly as in training
    criterion = CombinedBCEDiceLoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        pos_weight=args.pos_weight,
        empty_penalty=args.empty_penalty
    )
    
    # Evaluate the model on the entire validation set using the same evaluation logic as training
    print("Evaluating on the full validation set...")
    avg_loss, avg_dice = evaluate(model, val_loader, sam2_image, criterion, device)
    print(f"Validation Loss: {avg_loss:.4f}, Validation Dice: {avg_dice:.4f}")
    
    # Visualize predictions for the selected video using the same logic as evaluation phase
    print("Running visualization on the first video...")
    visualize_predictions(
        model=model,
        val_loader= val_loader,  # pass a list with single VideoDatapoint
        sam2=sam2_image,
        device=device,
        epoch=0,  # You can label this as "inference" or epoch 0
        output_dir=args.output_dir,
        num_samples=20,
        eval_mode=True
    )
    print("Inference and visualization complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for AutoSAM2 ModelEmb")
    parser.add_argument("--data_dir", type=str, default="/home/fodl/tomerslor/AutoSAM2/polyp_gen", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--pretrained", type=str, help="Path to pretrained ModelEmb weights")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for evaluation")
    parser.add_argument("--bce_weight", type=float, default=0.5, help="Weight for BCE loss component")
    parser.add_argument("--dice_weight", type=float, default=0.5, help="Weight for Dice loss component")
    parser.add_argument("--pos_weight", type=float, default=10.0, help="Positive class weight for BCE loss")
    parser.add_argument("--empty_penalty", type=float, default=0.5, help="Penalty for empty predictions")
    
    args = parser.parse_args()
    main(args)
