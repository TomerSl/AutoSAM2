
import os
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
import importlib.util
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from train import prepare_batch_for_sam2, sam_call



def debug_training_step(images, masks, model, sam2, criterion, batch_idx=0, output_dir="debug"):
    """
    Debug a single training step to understand what's happening.
    
    Args:
        images: Input images tensor [B, C, H, W]
        masks: Ground truth masks tensor [B, C, H, W]
        model: ModelEmb
        sam2: SAM2ImagePredictor
        criterion: Loss function
        batch_idx: Batch index for file naming
        output_dir: Directory to save debug visualizations
    """
    print("\n========== DEBUGGING INFORMATION ==========")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Check input data
    print(f"Image tensor shape: {images.shape}, dtype: {images.dtype}")
    print(f"Image min: {images.min().item()}, max: {images.max().item()}, mean: {images.mean().item()}")
    
    print(f"Mask tensor shape: {masks.shape}, dtype: {masks.dtype}")
    mask_sum = masks.sum().item()
    mask_pixels = masks.numel()
    mask_percentage = 100 * mask_sum / mask_pixels
    print(f"Mask statistics: sum={mask_sum}, pixels={mask_pixels}, foreground percentage={mask_percentage:.4f}%")
    
    if mask_percentage < 0.01:
        print("WARNING: Extremely low foreground percentage in masks!")
    
    # 2. Check model embeddings
    model.eval()  # Temporarily set to eval mode
    with torch.no_grad():
        dense_embeddings = model(images)
    model.train()  # Set back to train mode
    
    print(f"Dense embeddings shape: {dense_embeddings.shape}")
    print(f"Embeddings min: {dense_embeddings.min().item()}, max: {dense_embeddings.max().item()}")
    print(f"Embeddings mean: {dense_embeddings.mean().item()}, std: {dense_embeddings.std().item()}")
    
    # 3. Check mask predictions
    with torch.no_grad():
        mask_preds = sam_call(images, sam2, dense_embeddings)
        
        # Resize if needed
        if mask_preds.shape[2:] != masks.shape[2:]:
            mask_preds = F.interpolate(
                mask_preds,
                size=(masks.shape[2], masks.shape[3]),
                mode='bilinear',
                align_corners=False
            )
    
    print(f"Predicted mask tensor shape: {mask_preds.shape}")
    print(f"Pred logits min: {mask_preds.min().item()}, max: {mask_preds.max().item()}, mean: {mask_preds.mean().item()}")
    
    # 4. Check binary predictions (after sigmoid/threshold)
    sigmoid_preds = torch.sigmoid(mask_preds)
    print(f"After sigmoid - min: {sigmoid_preds.min().item()}, max: {sigmoid_preds.max().item()}, mean: {sigmoid_preds.mean().item()}")
    
    binary_preds = (sigmoid_preds > 0.5).float()
    binary_sum = binary_preds.sum().item()
    binary_percentage = 100 * binary_sum / binary_preds.numel()
    print(f"Binary prediction - foreground percentage: {binary_percentage:.4f}%")
    
    # 5. Check loss calculation
    with torch.no_grad():
        loss_val = criterion(mask_preds, masks).item()
    print(f"Loss value: {loss_val:.6f}")
    
    # 6. Check Dice score
    with torch.no_grad():
        dice_val = 1 - dice_loss(masks, binary_preds).item()
    print(f"Dice score: {dice_val:.6f}")
    
    # 7. Visualize images, masks and predictions
    for i in range(min(2, images.shape[0])):  # First 2 samples
        img = images[i].cpu().permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to 0-1
        
        gt_mask = masks[i, 0].cpu().numpy()
        pred_sigmoid = sigmoid_preds[i, 0].cpu().numpy()
        pred_binary = binary_preds[i, 0].cpu().numpy()
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(141)
        plt.imshow(img)
        plt.title(f"Image {i}")
        plt.axis('off')
        
        plt.subplot(142)
        plt.imshow(gt_mask, cmap='gray')
        plt.title(f"GT Mask (sum={gt_mask.sum():.1f})")
        plt.axis('off')
        
        plt.subplot(143)
        plt.imshow(pred_sigmoid, cmap='jet')
        plt.colorbar()
        plt.title(f"Pred Sigmoid")
        plt.axis('off')
        
        plt.subplot(144)
        plt.imshow(pred_binary, cmap='gray')
        plt.title(f"Pred Binary (sum={pred_binary.sum():.1f})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/debug_batch{batch_idx}_sample{i}.png")
        plt.close()
    
    # 8. Calculate pixel-wise statistics
    max_diff = (sigmoid_preds - masks).abs().max().item()
    print(f"Max absolute difference between prediction and ground truth: {max_diff:.6f}")
    
    positive_pixels_gt = (masks > 0.5).sum().item()
    positive_pixels_pred = (sigmoid_preds > 0.5).sum().item()
    positive_match = ((masks > 0.5) & (sigmoid_preds > 0.5)).sum().item()
    
    if positive_pixels_gt > 0:
        true_positive_rate = 100 * positive_match / positive_pixels_gt
        print(f"True positive rate: {true_positive_rate:.2f}%")
    else:
        print("No positive pixels in ground truth")
    
    if positive_pixels_pred > 0:
        precision = 100 * positive_match / positive_pixels_pred
        print(f"Precision: {precision:.2f}%")
    else:
        print("No positive pixels in prediction")
    
    # 9. Check histogram of predictions
    with torch.no_grad():
        pred_values = sigmoid_preds.cpu().numpy().flatten()
        plt.figure(figsize=(10, 5))
        plt.hist(pred_values, bins=50)
        plt.title("Histogram of Prediction Values")
        plt.xlabel("Prediction Value")
        plt.ylabel("Count")
        plt.savefig(f"{output_dir}/debug_batch{batch_idx}_histogram.png")
        plt.close()
    
    print("===========================================\n")
    return {
        "loss": loss_val,
        "dice": dice_val,
        "mask_percentage": mask_percentage,
        "pred_percentage": binary_percentage,
        "max_diff": max_diff,
    }


def check_gradients(model, step=0, print_every=10):
    """
    Check if gradients are flowing properly in the model.
    
    Args:
        model: The model to check gradients for
        step: Current training step (for logging)
        print_every: Print details every N steps
    
    Returns:
        gradient_stats: Dictionary with gradient statistics
    """
    # Track gradient stats
    stats = {
        'total_norm': 0.0,
        'max_norm': 0.0,
        'min_norm': float('inf'),
        'num_zero_grads': 0,
        'num_params_with_grad': 0
    }
    
    # Iterate through parameters
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            stats['total_norm'] += grad_norm
            stats['max_norm'] = max(stats['max_norm'], grad_norm)
            stats['min_norm'] = min(stats['min_norm'], grad_norm) if grad_norm > 0 else stats['min_norm']
            stats['num_params_with_grad'] += 1
            
            if grad_norm == 0:
                stats['num_zero_grads'] += 1
                
            # Print detailed info every N steps
            if step % print_every == 0 and grad_norm > 0:
                print(f"  Param: {name}, Grad norm: {grad_norm:.6f}")
    
    # Overall stats
    if stats['num_params_with_grad'] > 0:
        stats['avg_norm'] = stats['total_norm'] / stats['num_params_with_grad']
        if step % print_every == 0:
            print(f"Gradient Stats (Step {step}):")
            print(f"  Total norm: {stats['total_norm']:.6f}")
            print(f"  Max norm: {stats['max_norm']:.6f}")
            print(f"  Min norm (non-zero): {stats['min_norm']:.6f}")
            print(f"  Avg norm: {stats['avg_norm']:.6f}")
            print(f"  Zero grads: {stats['num_zero_grads']}/{stats['num_params_with_grad']} params")
    else:
        print("WARNING: No parameters have gradients!")
        
    return stats


def visualize_predictions(model, val_loader, sam2, device, epoch, output_dir):
    """
    Visualize predictions for 4 random samples from the validation set
    
    Args:
        model: ModelEmb to evaluate
        val_loader: Validation dataloader
        sam2: SAM2ImagePredictor (frozen)
        device: Device
        epoch: Current epoch
        output_dir: Output directory
    """
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations', f'epoch_{epoch}')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Instead of loading all samples (which causes memory issues), 
    # select 4 random indices from the dataset directly
    dataset = val_loader.dataset
    dataset_size = len(dataset)
    
    # Ensure we have enough samples
    if dataset_size < 4:
        print("Not enough validation samples for visualization")
        return
        
    # Select 4 random indices
    random_indices = random.sample(range(dataset_size), 4)
    
    model.eval()
    sam2.model.eval()
    
    with torch.no_grad():
        for i, idx in enumerate(random_indices):
            # Get sample directly from dataset
            sample = dataset[idx]
            
            # Prepare sample
            images, masks, _ = prepare_batch_for_sam2([sample], device, output_format='image')
            
            # Get model predictions
            dense_embeddings = model(images)
            mask_preds = sam_call(images, sam2, dense_embeddings)
            
            # Upscale predictions if needed
            if mask_preds.shape[2:] != masks.shape[2:]:
                mask_preds = F.interpolate(
                    mask_preds,
                    size=(masks.shape[2], masks.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Binarize predictions
            pred_masks = (mask_preds > 0).float()
            
            # Convert to numpy for visualization
            image_np = images[0].cpu().permute(1, 2, 0).numpy()
            gt_mask_np = masks[0, 0].cpu().numpy()
            pred_mask_np = pred_masks[0, 0].cpu().numpy()
            
            # Normalize image for visualization
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
            
            # Create figure with 3 subplots
            plt.figure(figsize=(15, 5))
            
            # Plot original image
            plt.subplot(1, 3, 1)
            plt.imshow(image_np)
            plt.title('Original Image')
            plt.axis('off')
            
            # Plot ground truth mask
            plt.subplot(1, 3, 2)
            plt.imshow(gt_mask_np, cmap='gray')
            plt.title('Ground Truth Mask')
            plt.axis('off')
            
            # Plot predicted mask
            plt.subplot(1, 3, 3)
            plt.imshow(pred_mask_np, cmap='gray')
            plt.title(f'Predicted Mask (Epoch {epoch})')
            plt.axis('off')
            
            # Save figure
            plt.savefig(os.path.join(vis_dir, f'sample_{i}.png'), dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"Saved validation visualizations to {vis_dir}")


def analyze_dataset(dataset, name, num_samples=10):
    """
    Analyze the dataset structure and content before training
    
    Args:
        dataset: Dataset to analyze
        name: Name of the dataset (for logging)
        num_samples: Number of samples to analyze
    """
    print(f"\n{'='*20} ANALYZING {name} DATASET {'='*20}")
    print(f"Dataset length: {len(dataset)}")
    
    # Create debug directory
    debug_dir = f"debug/{name}_dataset"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Track mask statistics
    mask_stats = {
        'sum': [],
        'mean': [],
        'foreground_percent': []
    }
    
    # Sample random indices
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    # Plot some samples
    for i, idx in enumerate(indices):
        sample = dataset[idx]
        
        # Extract image and mask
        frames = sample.frames
        if len(frames) == 0 or len(frames[0].objects) == 0:
            print(f"Sample {idx} has no frames or objects")
            continue
            
        image = frames[0].data
        mask = frames[0].objects[0].segment
        
        # Convert to numpy for plotting
        image_np = image.permute(1, 2, 0).numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min() + 1e-8)
        
        # Fix: Handle mask dimensions properly
        if isinstance(mask, torch.Tensor):
            # Remove any extra dimensions to get a 2D array
            if mask.dim() > 2:
                mask_np = mask.float().squeeze().numpy()  # Squeeze out all singleton dimensions
            else:
                mask_np = mask.float().numpy()
        else:
            mask_np = np.array(mask).astype(float)
            # Also ensure numpy array has correct dimensions
            if mask_np.ndim > 2:
                mask_np = np.squeeze(mask_np)  # Remove singleton dimensions
            
        # Calculate statistics
        mask_sum = float(mask_np.sum())
        mask_size = float(mask_np.size)
        mask_mean = mask_sum / mask_size
        mask_percent = 100.0 * mask_sum / mask_size
        
        mask_stats['sum'].append(mask_sum)
        mask_stats['mean'].append(mask_mean)
        mask_stats['foreground_percent'].append(mask_percent)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(121)
        plt.imshow(image_np)
        plt.title(f"Image {idx}")
        plt.axis('off')
        
        plt.subplot(122)
        plt.imshow(mask_np, cmap='gray')
        plt.title(f"Mask {idx} (FG: {mask_percent:.2f}%)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{debug_dir}/sample_{i}.png")
        plt.close()
        
        print(f"Sample {idx}: Mask sum={mask_sum}, mean={mask_mean:.4f}, foreground={mask_percent:.2f}%")
    
    # Summarize mask statistics
    if mask_stats['sum']:
        avg_sum = sum(mask_stats['sum']) / len(mask_stats['sum'])
        avg_percent = sum(mask_stats['foreground_percent']) / len(mask_stats['foreground_percent'])
        
        print(f"Average mask sum: {avg_sum:.2f}")
        print(f"Average foreground percentage: {avg_percent:.2f}%")
        
        # Plot histogram of foreground percentages
        plt.figure(figsize=(10, 6))
        plt.hist(mask_stats['foreground_percent'], bins=20)
        plt.xlabel('Foreground Percentage')
        plt.ylabel('Count')
        plt.title(f'{name} Dataset - Foreground Percentage Distribution')
        plt.savefig(f"{debug_dir}/foreground_histogram.png")
        plt.close()
    
    print(f"{'='*20} END OF DATASET ANALYSIS {'='*20}\n")
    return mask_stats


def visualize_gradients(model, epoch, batch_idx, output_dir="debug/gradients"):
    """
    Visualize gradients for each layer in the model
    
    Args:
        model: The model to visualize gradients for
        epoch: Current epoch
        batch_idx: Current batch index
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for gradient norms
    plt.figure(figsize=(15, 10))
    
    # Track gradient information by layer type
    layer_types = {}
    max_norm = 0
    
    # Collect gradient data
    grad_data = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Calculate gradient norm
            grad_norm = param.grad.data.norm(2).item()
            max_norm = max(max_norm, grad_norm)
            
            # Categorize by layer type
            layer_type = name.split('.')[-2] if '.' in name else 'other'
            if layer_type not in layer_types:
                layer_types[layer_type] = []
            
            layer_types[layer_type].append(grad_norm)
            
            # Add to gradient data
            grad_data.append({
                'name': name,
                'norm': grad_norm,
                'mean': param.grad.data.abs().mean().item(),
                'std': param.grad.data.std().item(),
                'shape': list(param.shape),
                'layer_type': layer_type
            })
    
    # Sort by gradient norm
    grad_data.sort(key=lambda x: x['norm'], reverse=True)
    
    # Plot top 30 gradient norms
    plt.subplot(2, 1, 1)
    plt.bar(
        range(min(30, len(grad_data))), 
        [item['norm'] for item in grad_data[:30]],
        tick_label=[item['name'].split('.')[-1] for item in grad_data[:30]]
    )
    plt.xticks(rotation=90)
    plt.title(f'Top 30 Gradient Norms (Epoch {epoch}, Batch {batch_idx})')
    plt.ylabel('Gradient Norm')
    plt.tight_layout()
    
    # Plot gradient norms by layer type (mean)
    plt.subplot(2, 1, 2)
    layer_means = {k: sum(v)/len(v) for k, v in layer_types.items()}
    plt.bar(
        range(len(layer_means)),
        list(layer_means.values()),
        tick_label=list(layer_means.keys())
    )
    plt.xticks(rotation=45)
    plt.title('Mean Gradient Norm by Layer Type')
    plt.ylabel('Mean Gradient Norm')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gradients_e{epoch}_b{batch_idx}.png")
    plt.close()
    
    # Create histogram of all gradient values for top 5 layers with highest norm
    plt.figure(figsize=(15, 5 * min(5, len(grad_data))))
    
    for i, item in enumerate(grad_data[:5]):
        param = dict(model.named_parameters())[item['name']]
        grad_values = param.grad.data.cpu().numpy().flatten()
        
        plt.subplot(min(5, len(grad_data)), 1, i+1)
        plt.hist(grad_values, bins=50)
        plt.title(f"{item['name']} - Norm: {item['norm']:.6f}, Mean: {item['mean']:.6f}, Std: {item['std']:.6f}")
        plt.xlabel("Gradient Value")
        plt.ylabel("Count")
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/grad_hist_e{epoch}_b{batch_idx}.png")
    plt.close()
    
    # Save detailed gradient stats to file
    with open(f"{output_dir}/grad_stats_e{epoch}_b{batch_idx}.txt", 'w') as f:
        f.write(f"Gradient Statistics for Epoch {epoch}, Batch {batch_idx}\n")
        f.write("-" * 50 + "\n\n")
        
        for i, item in enumerate(grad_data):
            f.write(f"{i+1}. {item['name']}\n")
            f.write(f"   Shape: {item['shape']}\n")
            f.write(f"   Layer Type: {item['layer_type']}\n")
            f.write(f"   Norm: {item['norm']:.6f}\n")
            f.write(f"   Mean: {item['mean']:.6f}\n")
            f.write(f"   Std: {item['std']:.6f}\n\n")
            
    return grad_data
