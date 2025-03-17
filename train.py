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
from debug import *
# Add sam2-main to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sam2-main'))

# Import SAM2 data structures for video processing
from training.utils.data_utils import VideoDatapoint, Frame, Object

# Import ModelEmb from models/model_single.py
from models.model_single import ModelEmb

# Import SAM2 components
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.transforms import SAM2Transforms

# Import polyp_gen module directly using importlib
polyp_gen_path = os.path.join(os.path.dirname(__file__), 'dataset', 'polyp_gen.py')
spec = importlib.util.spec_from_file_location("polyp_gen", polyp_gen_path)
polyp_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(polyp_gen)
get_polyp_gen_dataset = polyp_gen.get_polyp_gen_dataset
# Import VideoDataLoader for video evaluation
VideoDataLoader = polyp_gen.VideoDataLoader

# Change PyTorch cache location
os.environ['TORCH_HOME'] = '/tmp/torch_cache'

def norm_batch(x):
    """Normalize a batch of masks or predictions"""
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss


def dice_loss(y_true, y_pred, smooth=1):
    """Calculate Dice loss"""
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return 1 - dice


def prepare_batch_for_sam2(batch, device, output_format='video'):
    """
    Process batch of VideoDatapoints for SAM2 model processing
    
    Args:
        batch: List of VideoDatapoint objects
        device: Torch device to move tensors to
        output_format: 'video' for [B, C, T, H, W] format, 'image' for [B, C, H, W]
    
    Returns:
        images: Tensor of images [B, C, T, H, W] or [B, C, H, W]
        masks: Tensor of masks [B, C, T, H, W] or [B, C, H, W]
        ids: Tensor of video/image IDs
    """
    try:
        images = []
        masks = []
        ids = []
        
        # Set a fixed target size for all images and masks
        # This ensures all tensors have the same dimensions
        TARGET_SIZE = (1024, 1024)  # Common size for SAM models
        
        for dp in batch:
            if len(dp.frames) > 0:
                # Get the image tensor from the frame
                img = dp.frames[0].data
                
                # Resize image to the target size
                img_resized = F.interpolate(img.unsqueeze(0), size=TARGET_SIZE, 
                                            mode='bilinear', align_corners=False).squeeze(0)
                
                images.append(img_resized)
                
                # Get the mask tensor from the first object in the frame
                if len(dp.frames[0].objects) > 0 and dp.frames[0].objects[0].segment is not None:
                    mask = dp.frames[0].objects[0].segment.float()
                    
                    # Ensure mask is at least 2D
                    if mask.dim() == 1:
                        # If mask is 1D, try to reshape it based on image dimensions
                        h, w = dp.frames[0].data.shape[1:]
                        mask = mask.reshape(h, w)
                    
                    # Add channel dimension if missing (create [C, H, W] tensor)
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(0)
                    
                    # Resize mask to match the target size
                    mask_resized = F.interpolate(mask.unsqueeze(0), size=TARGET_SIZE, 
                                                mode='nearest').squeeze(0)
                    
                    masks.append(mask_resized)
                else:
                    # Create empty mask with target dimensions
                    masks.append(torch.zeros((1, TARGET_SIZE[0], TARGET_SIZE[1])))
                
                # Add identifier
                ids.append(dp.video_id)
        
        # Stack resized images and masks
        images = torch.stack(images).to(device)
        masks = torch.stack(masks).to(device)
        
        # Handle video data format if needed (5D tensor)
        if len(images.shape) == 5:  # [B, T, C, H, W]
            # Adjust for video format
            images = images.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            if len(masks.shape) == 4:  # [B, C, H, W]
                # Add time dimension to masks to match video format
                masks = masks.unsqueeze(2)  # [B, C, 1, H, W]
        
        ids = torch.tensor(ids).to(device)
        
        if output_format == 'image':
            return images, masks, ids
        else:
            return images, masks, ids
            
    except Exception as e:
        print(f"Error in prepare_batch_for_sam2: {e}")
        import traceback
        traceback.print_exc()
        raise


def sam_call(images, sam2, dense_embeddings, state=None):
    """
    Generate masks using SAM2 with dense embeddings from our ModelEmb
    
    Args:
        images: Input images (B, C, H, W)
        sam2: SAM2ImagePredictor instance with model loaded
        dense_embeddings: Dense embeddings from our model (B, C, H, W)
        state: State for prompting (default: None)
        
    Returns:
        low_res_masks: Generated masks
    """
    B = images.shape[0]
    
    # We need gradients for dense_embeddings, so don't use no_grad for the whole function
    
    # Extract image features (note: the API is nested differently in the predictor)
    with torch.no_grad():  # No gradients for SAM2 image encoder
        image_embeddings = sam2.model.image_encoder(images)
        
        # Process backbone features
        image_embeddings["backbone_fpn"][0] = sam2.model.sam_mask_decoder.conv_s0(
            image_embeddings["backbone_fpn"][0]
        )
        image_embeddings["backbone_fpn"][1] = sam2.model.sam_mask_decoder.conv_s1(
            image_embeddings["backbone_fpn"][1]
        )

        # Prepare features for the decoder
        tokens, vision_feats, vision_pos_embeds, feat_sizes = sam2.model._prepare_backbone_features(image_embeddings)

        # Extract high resolution features if available
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        # Default empty prompt (we're using dense embeddings)
        sam_point_coords = torch.zeros(B, 1, 2, device=images.device)
        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=images.device)
        
        # Get sparse embeddings (empty)
        sparse_embeddings, _ = sam2.model.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=None
        )
        
        # Create a dummy output_dict for conditioning features
        # This is a single-frame version, so we're treating each image independently
        # Note that for video we would need to maintain this structure across frames
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {}
        }
        
        # Prepare pixel features with conditioning
        # For single images, we treat it as the first frame (t=0, is_init_cond_frame=True)
        pix_feat = sam2.model._prepare_memory_conditioned_features(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=vision_feats[-1:],
            current_vision_pos_embeds=vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=output_dict,
            num_frames=1
        )
    
    # This part needs to track gradients for dense_embeddings
    # We still need to prevent SAM2 parameters from being updated, but that's handled by .requires_grad = False
    # and not by torch.no_grad()
    low_res_masks, ious, sam_output_tokens, object_score_logits = sam2.model.sam_mask_decoder(
        image_embeddings=pix_feat,  # Use conditioned features, not tokens directly
        image_pe=sam2.model.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,  # Our dense embeddings need gradients
        multimask_output=False,
        repeat_image=False,
        high_res_features=high_res_features,
    )
        
    return low_res_masks


class CombinedBCEDiceLoss(nn.Module):
    """
    Combines Binary Cross Entropy and Dice Loss with improved handling for imbalanced data.
    
    Args:
        bce_weight: Weight of BCE loss component (default: 0.5)
        dice_weight: Weight of Dice loss component (default: 0.5)
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0, pos_weight=10.0, empty_penalty=5.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.empty_penalty = empty_penalty  # Additional penalty for empty predictions
        
        # Increase positive class weight to handle class imbalance
        self.pos_weight = torch.tensor([pos_weight])
        self.bce_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
    def forward(self, inputs, targets):
        if self.pos_weight.device != inputs.device:
            self.pos_weight = self.pos_weight.to(inputs.device)
            self.bce_criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        
        # BCE Loss
        bce_loss = self.bce_criterion(inputs, targets)
        
        # Dice Loss with sigmoid activation
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        union = inputs_flat.sum() + targets_flat.sum()
        
        # Add epsilon to prevent division by zero
        eps = 1e-6
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth + eps)
        dice_loss = 1 - dice_score
        
        # Add penalty for empty predictions
        pred_sum = inputs_sigmoid.sum()
        if pred_sum < eps:
            empty_loss = self.empty_penalty
        else:
            empty_loss = 0.0
        
        # Combined loss with gradient clipping
        combined_loss = self.bce_weight * torch.clamp(bce_loss, 0, 10) + self.dice_weight * dice_loss + empty_loss
        
        return combined_loss


def train_one_epoch(model, train_loader, sam2, optimizer, criterion, device, epoch):
    """
    Train model for one epoch with enhanced debugging
    
    Args:
        model: ModelEmb to train
        train_loader: Training dataloader
        sam2: SAM2ImagePredictor (frozen)
        optimizer: Optimizer for ModelEmb
        criterion: Loss function
        device: Device
        epoch: Current epoch
        
    Returns:
        avg_loss: Average loss
        avg_dice: Average Dice score
    """
    # Ensure ModelEmb is in training mode
    model.train()
    sam2.model.eval()
    
    running_loss = 0.0
    dice_scores = []
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(progress_bar):
        try:
            images, masks, _ = prepare_batch_for_sam2(batch, device, output_format='image')
            
            # Skip batches with no foreground or all foreground
            mask_sum = masks.sum().item()
            mask_size = masks.numel()
            if mask_sum == 0 or mask_sum == mask_size:
                continue
            
            # Increase minimum foreground percentage threshold
            fg_percent = 100.0 * mask_sum / mask_size
            if fg_percent < 0.5:  # Increased from 0.1% to 0.5%
                continue
            
            # Skip if mask is too sparse (less than 100 foreground pixels)
            if mask_sum < 100:
                continue
            
            optimizer.zero_grad()
            
            # Forward pass
            dense_embeddings = model(images)
            mask_preds = sam_call(images, sam2, dense_embeddings)
            
            # Upscale masks to match target size if needed
            if mask_preds.shape[2:] != masks.shape[2:]:
                mask_preds = F.interpolate(
                    mask_preds,
                    size=(masks.shape[2], masks.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Calculate loss
            loss = criterion(mask_preds, masks)
            
            # Skip if loss is too high
            if loss.item() > 10:
                continue
                
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            pred_masks = (torch.sigmoid(mask_preds) > 0.5).float()
            batch_dice = 1 - dice_loss(masks, pred_masks).item()
            dice_scores.append(batch_dice)
            
            progress_bar.set_postfix(loss=running_loss/(batch_idx+1), dice=np.mean(dice_scores))
            
        except Exception as e:
            continue
    
    avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
    avg_dice = np.mean(dice_scores) if len(dice_scores) > 0 else 0.0
    
    return avg_loss, avg_dice


def evaluate(model, val_loader, sam2, criterion, device):
    """
    Evaluate model on validation set
    
    Args:
        model: ModelEmb to evaluate
        val_loader: Validation dataloader
        sam2: SAM2ImagePredictor (frozen)
        criterion: Loss function
        device: Device
        
    Returns:
        avg_loss: Average loss
        avg_dice: Average Dice score
    """
    # Set model to evaluation mode
    model.eval()
    
    # Ensure SAM2 is in eval mode
    sam2.model.eval()
    
    running_loss = 0.0
    dice_scores = []
    progress_bar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare batch
            images, masks, _ = prepare_batch_for_sam2(batch, device, output_format='image')
            
            # Forward pass through our embedding model
            dense_embeddings = model(images)
            
            # Generate masks using SAM2
            mask_preds = sam_call(images, sam2, dense_embeddings)
            
            # Upscale masks to match target size if needed
            if mask_preds.shape[2:] != masks.shape[2:]:
                mask_preds = F.interpolate(
                    mask_preds,
                    size=(masks.shape[2], masks.shape[3]),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Calculate loss
            loss = criterion(mask_preds, masks)
            running_loss += loss.item()
            
            # Calculate Dice score
            pred_masks = (mask_preds > 0).float()
            batch_dice = 1 - dice_loss(masks, pred_masks).item()
            dice_scores.append(batch_dice)
            
            progress_bar.set_postfix(loss=running_loss/(batch_idx+1), dice=np.mean(dice_scores))
    
    return running_loss / len(val_loader), np.mean(dice_scores)


def evaluate_video(model, val_dataset, sam2_video, device, output_dir):
    """
    Evaluate model on video data using SAM2VideoPredictor.
    Processes one video at a time with consistent frame dimensions.
    
    Args:
        model: ModelEmb model to evaluate
        val_dataset: Dataset containing video data
        sam2_video: SAM2VideoPredictor instance
        device: Device to run evaluation on
        output_dir: Directory to save evaluation results
        
    Returns:
        avg_loss: Average loss across all videos
    """
    # Set models to evaluation mode
    model.eval()
    sam2_video.image_encoder.eval()
    sam2_video.memory_attention.eval()
    sam2_video.memory_encoder.eval()
    
    running_loss = 0.0
    dice_scores = []
    total_videos = 0
    successful_videos = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the VideoDataLoader from polyp_gen.py
    video_loader = VideoDataLoader(val_dataset, device)
    progress_bar = tqdm(video_loader, desc='Video Evaluation')
    
    with torch.no_grad():
        for video_datapoint in progress_bar:
            try:
                total_videos += 1
                video_id = video_datapoint.video_id
                frames = video_datapoint.frames
                
                # Skip empty videos
                if not frames:
                    print(f"Skipping empty video {video_id}")
                    continue
                
                # Choose a fixed size for all frames in this video
                TARGET_SIZE = (1024, 1024)
                    
                # Prepare tensors for the entire video - ensure all frames have EXACTLY the same shape
                processed_images = []
                processed_masks = []
                
                # First pass: extract and resize all frames to standard size
                for f in frames:
                    # Extract image and always resize to target size
                    img = f.data
                    if len(img.shape) > 3:  # If it's [B, C, H, W], squeeze the batch dimension
                        img = img.squeeze(0)
                    
                    # Always resize to the fixed target size
                    img = F.interpolate(
                        img.unsqueeze(0), 
                        size=TARGET_SIZE,
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze(0)
                    
                    # Extract mask and always resize to target size
                    mask = f.objects[0].segment.float()
                    if mask.dim() == 2:  # If it's [H, W], add channel dimension
                        mask = mask.unsqueeze(0)
                    
                    # Always resize to the fixed target size
                    mask = F.interpolate(
                        mask.unsqueeze(0),
                        size=TARGET_SIZE,
                        mode='nearest'
                    ).squeeze(0)
                    
                    processed_images.append(img)
                    processed_masks.append(mask)
                
                # Verify shapes before stacking
                C = processed_images[0].shape[0]
                H, W = TARGET_SIZE
                T = len(processed_images)
                
                # Check all processed frames for consistency
                shapes_consistent = True
                for i, (img, mask) in enumerate(zip(processed_images, processed_masks)):
                    if img.shape != (C, H, W):
                        print(f"Warning: Image {i} has wrong shape {img.shape}, expected {(C, H, W)}")
                        shapes_consistent = False
                    if mask.shape != (1, H, W):
                        print(f"Warning: Mask {i} has wrong shape {mask.shape}, expected {(1, H, W)}")
                        shapes_consistent = False
                
                if not shapes_consistent:
                    print(f"Skipping video {video_id} due to inconsistent frame shapes")
                    continue
                
                # Stack processed frames
                images = torch.stack(processed_images).unsqueeze(0).to(device)  # [1, T, C, H, W]
                masks = torch.stack(processed_masks).unsqueeze(0).to(device)    # [1, T, 1, H, W]
                
                # Verify stacked tensor shapes
                if images.shape != (1, T, C, H, W):
                    print(f"Error: Stacked images shape {images.shape} doesn't match expected {(1, T, C, H, W)}")
                    continue
                if masks.shape != (1, T, 1, H, W):
                    print(f"Error: Stacked masks shape {masks.shape} doesn't match expected {(1, T, 1, H, W)}")
                    continue
                    
                # Process frames in chunks to manage memory
                chunk_size = 32
                all_preds = []
                
                for chunk_start in range(0, T, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, T)
                    chunk_images = images[:, chunk_start:chunk_end]
                    chunk_masks = []
                    
                    # Memory dictionary for current chunk
                    output_dict = {
                        "cond_frame_outputs": {},
                        "non_cond_frame_outputs": {}
                    }
                    
                    # Process each frame in chunk
                    for t in range(chunk_end - chunk_start):
                        frame = chunk_images[:, t]  # [1, C, H, W]
                        dense_embeddings = model(frame)
                        
                        if t == 0 and chunk_start == 0:
                            # First frame of video - initialize memory
                            frame_embedding = sam2_video.image_encoder(frame)
                            frame_embedding["backbone_fpn"][0] = sam2_video.sam_mask_decoder.conv_s0(
                                frame_embedding["backbone_fpn"][0]
                            )
                            frame_embedding["backbone_fpn"][1] = sam2_video.sam_mask_decoder.conv_s1(
                                frame_embedding["backbone_fpn"][1]
                            )
                            
                            frame_embedding, vision_feats, vision_pos_embeds, feat_sizes = sam2_video._prepare_backbone_features(frame_embedding)
                            
                            # Default empty prompt
                            sam_point_coords = torch.zeros(1, 1, 2, device=device)
                            sam_point_labels = -torch.ones(1, 1, dtype=torch.int32, device=device)
                            sparse_embeddings, _ = sam2_video.sam_prompt_encoder(
                                points=(sam_point_coords, sam_point_labels),
                                boxes=None,
                                masks=None
                            )
                            
                            # Generate first frame mask
                            low_res_masks, ious, sam_output_tokens, object_score_logits = sam2_video.sam_mask_decoder(
                                image_embeddings=frame_embedding,
                                multi_scale_features=None,
                                vision_features=vision_feats[-1].permute(1, 0, 2),
                                vision_pos_embeds=vision_pos_embeds[-1].permute(1, 0, 2),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                            )
                            
                            # Store memory
                            sam_output_token = sam_output_tokens[:, 0]
                            obj_ptr = sam2_video.obj_ptr_proj(sam_output_token)
                            maskmem_features, maskmem_pos_enc = sam2_video._encode_new_memory(
                                vision_feats,
                                feat_sizes,
                                low_res_masks,
                                object_score_logits,
                                False,
                            )
                            output_dict["cond_frame_outputs"][t] = {
                                "obj_ptr": obj_ptr,
                                "maskmem_features": maskmem_features,
                                "maskmem_pos_enc": maskmem_pos_enc
                            }
                            mask_pred = low_res_masks
                            
                        else:
                            # Subsequent frames - use memory conditioning
                            frame_embedding = sam2_video.image_encoder(frame)
                            frame_embedding["backbone_fpn"][0] = sam2_video.sam_mask_decoder.conv_s0(
                                frame_embedding["backbone_fpn"][0]
                            )
                            frame_embedding["backbone_fpn"][1] = sam2_video.sam_mask_decoder.conv_s1(
                                frame_embedding["backbone_fpn"][1]
                            )
                            
                            frame_embedding, vision_feats, vision_pos_embeds, feat_sizes = sam2_video._prepare_backbone_features(frame_embedding)
                            
                            pix_feat = sam2_video._prepare_memory_conditioned_features(
                                frame_idx=t + chunk_start,
                                is_init_cond_frame=False,
                                current_vision_feats=vision_feats[-1:],
                                current_vision_pos_embeds=vision_pos_embeds[-1:],
                                feat_sizes=feat_sizes[-1:],
                                output_dict=output_dict,
                                num_frames=T,
                            )
                            
                            sam_point_coords = torch.zeros(1, 1, 2, device=device)
                            sam_point_labels = -torch.ones(1, 1, dtype=torch.int32, device=device)
                            sparse_embeddings, _ = sam2_video.sam_prompt_encoder(
                                points=(sam_point_coords, sam_point_labels),
                                boxes=None,
                                masks=None
                            )
                            
                            low_res_masks, ious, sam_output_tokens, object_score_logits = sam2_video.sam_mask_decoder(
                                image_embeddings=pix_feat,
                                multi_scale_features=None,
                                vision_features=vision_feats[-1].permute(1, 0, 2),
                                vision_pos_embeds=vision_pos_embeds[-1].permute(1, 0, 2),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                            )
                            
                            # Update memory
                            sam_output_token = sam_output_tokens[:, 0]
                            obj_ptr = sam2_video.obj_ptr_proj(sam_output_token)
                            maskmem_features, maskmem_pos_enc = sam2_video._encode_new_memory(
                                vision_feats,
                                feat_sizes,
                                low_res_masks,
                                object_score_logits,
                                False,
                            )
                            output_dict["non_cond_frame_outputs"][t + chunk_start] = {
                                "obj_ptr": obj_ptr,
                                "maskmem_features": maskmem_features,
                                "maskmem_pos_enc": maskmem_pos_enc
                            }
                            mask_pred = low_res_masks
                        
                        chunk_masks.append(mask_pred)
                    
                    # Stack chunk predictions
                    chunk_masks = torch.stack(chunk_masks, dim=1)  # [1, chunk_size, 1, H, W]
                    all_preds.append(chunk_masks)
                
                # Concatenate all chunks
                all_preds = torch.cat(all_preds, dim=1)  # [1, T, 1, H, W]
                
                # Calculate metrics for the video
                video_dice = []
                video_loss = 0.0
                
                for t in range(T):
                    pred = (all_preds[0, t] > 0).float()
                    target = masks[0, t]
                    frame_dice = 1 - dice_loss(target, pred).item()
                    video_dice.append(frame_dice)
                    
                    frame_loss = dice_loss(target, pred)
                    video_loss += frame_loss.item()
                
                # Average metrics for this video
                avg_video_dice = np.mean(video_dice)
                avg_video_loss = video_loss / T
                
                dice_scores.append(avg_video_dice)
                running_loss += avg_video_loss
                successful_videos += 1
                
                # Save results
                video_dir = os.path.join(output_dir, f'video_{video_id}')
                os.makedirs(video_dir, exist_ok=True)
                
                metrics = {
                    'video_id': video_id,
                    'dice': avg_video_dice,
                    'loss': avg_video_loss,
                    'dice_per_frame': video_dice
                }
                
                with open(os.path.join(video_dir, 'metrics.json'), 'w') as f:
                    json.dump(metrics, f)
                
                # Update progress bar
                progress_bar.set_postfix(
                    avg_loss=running_loss/max(successful_videos, 1),
                    avg_dice=np.mean(dice_scores) if dice_scores else 0.0,
                    successful=f"{successful_videos}/{total_videos}"
                )
                
            except Exception as e:
                print(f"Error processing video {video_id}: {str(e)}")
                continue
    
    # Calculate final average loss
    avg_loss = running_loss / successful_videos if successful_videos > 0 else float('inf')
    avg_dice = np.mean(dice_scores) if dice_scores else 0.0
    
    # Save overall metrics
    overall_metrics = {
        'avg_loss': avg_loss,
        'avg_dice': avg_dice,
        'total_videos': total_videos,
        'successful_videos': successful_videos
    }
    
    with open(os.path.join(output_dir, 'overall_metrics.json'), 'w') as f:
        json.dump(overall_metrics, f)
    
    print(f'\nVideo Evaluation Complete:')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Average Dice: {avg_dice:.4f}')
    print(f'Total Videos Processed: {total_videos}')
    print(f'Successful Videos: {successful_videos}')
    
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, dice_score, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'dice_score': dice_score
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth'))


def main(args):
    # Suppress warnings about skipping RandomAffine for zero-area masks
    import warnings
    import logging
    warnings.filterwarnings("ignore", message="Skip RandomAffine")
    logging.getLogger().setLevel(logging.ERROR)  # Only show ERROR messages
    
    # Setup directories
    debug_dir = "debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        print(f"Created debug directory: {debug_dir}")
        
    # Use CUDA if available, with deterministic mode for reproducibility
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    
    # Set random seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seeds set to {seed} for reproducibility")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load SAM2 image model ONLY from local checkpoints
    print("Loading SAM2 image and video models from local checkpoints...")

    # Use checkpoint in sam2-main/checkpoints
    sam2_dir = os.path.join(os.path.dirname(__file__), 'sam2-main')
    checkpoint_path = os.path.join(sam2_dir, 'checkpoints', 'sam2.1_hiera_large.pt')
    video_checkpoint_path = os.path.join(sam2_dir, 'checkpoints', 'sam2.1_hiera_large.pt')  # Video checkpoint
    config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Relative path used by SAM2
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Please ensure the sam2-main/checkpoints directory contains the model files.")
    if not os.path.exists(video_checkpoint_path):
        raise FileNotFoundError(f"Video checkpoint file not found at {config_path}.")

    # Import the required components
    from sam2.build_sam import build_sam2

    print(f"Loading SAM2 image model from checkpoint: {checkpoint_path}")
    print(f"Loading SAM2 video model from checkpoint: {video_checkpoint_path}")
    print(f"Using configs: {config_path}")

    # Set up the environment for Hydra to find configs
    os.environ["PYTHONPATH"] = f"{sam2_dir}:{os.environ.get('PYTHONPATH', '')}"

    # Change to sam2-main directory temporarily for correct config paths
    original_dir = os.getcwd()
    os.chdir(sam2_dir)

    try:
        # Build the image model
        sam2_model = build_sam2(config_file=config_path, ckpt_path=checkpoint_path, device=device)
        sam2_image = SAM2ImagePredictor(sam_model=sam2_model)
        
        # Build the video model
        sam2_video_model = build_sam2(config_file=config_path, ckpt_path=video_checkpoint_path, device=device)
        # Initialize video predictor with the correct parameters
        sam2_video = SAM2VideoPredictor(
            image_encoder=sam2_video_model.image_encoder,
            memory_attention=sam2_video_model.memory_attention,
            memory_encoder=sam2_video_model.memory_encoder
        )
    finally:
        # Return to original directory
        os.chdir(original_dir)

    # Freeze all SAM2 parameters for both models
    for param in sam2_image.model.parameters():
        param.requires_grad = False
    
    # Freeze video predictor parameters
    for param in sam2_video.image_encoder.parameters():
        param.requires_grad = False
    for param in sam2_video.memory_attention.parameters():
        param.requires_grad = False
    for param in sam2_video.memory_encoder.parameters():
        param.requires_grad = False

    # Set both to evaluation mode
    sam2_image.model.eval()
    sam2_video.image_encoder.eval()
    sam2_video.memory_attention.eval()
    sam2_video.memory_encoder.eval()

    # Print model structures
    print("\nSAM2 Image Model Summary:")
    sam_params = sum(p.numel() for p in sam2_image.model.parameters())
    sam_trainable = sum(p.numel() for p in sam2_image.model.parameters() if p.requires_grad)
    print(f"Total parameters: {sam_params:,}")
    print(f"Trainable parameters: {sam_trainable:,}")
    print(f"Frozen parameters: {sam_params - sam_trainable:,}")

    print("\nSAM2 Video Model Summary:")
    video_params = (
        sum(p.numel() for p in sam2_video.image_encoder.parameters()) +
        sum(p.numel() for p in sam2_video.memory_attention.parameters()) +
        sum(p.numel() for p in sam2_video.memory_encoder.parameters())
    )
    video_trainable = (
        sum(p.numel() for p in sam2_video.image_encoder.parameters() if p.requires_grad) +
        sum(p.numel() for p in sam2_video.memory_attention.parameters() if p.requires_grad) +
        sum(p.numel() for p in sam2_video.memory_encoder.parameters() if p.requires_grad)
    )
    print(f"Total parameters: {video_params:,}")
    print(f"Trainable parameters: {video_trainable:,}")
    print(f"Frozen parameters: {video_params - video_trainable:,}")
    
    # Disable transformations for debugging - use None instead
    transform = None
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset, val_dataset = get_polyp_gen_dataset(sam_trans=transform, base_path=args.data_dir)
    # Define custom collate function that simply returns the batch list
    # This is needed because DataLoader's default collate can't handle VideoDatapoint objects
    def custom_collate_fn(batch):
        return batch
    
    # Create dataloaders with smaller batch size for stability
    actual_batch_size = args.batch_size  # Cap batch size at 8
    print(f"\nUsing batch size: {actual_batch_size} (requested: {args.batch_size})")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=2,  # Reduce workers to 2
        pin_memory=False,  # Disable pin_memory for stability
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=2,  # Reduce workers to 2
        pin_memory=False,  # Disable pin_memory
        collate_fn=custom_collate_fn
    )
    
    # Initialize our model
    print("\nCreating AutoSAM2 ModelEmb...")
    
    # Create args dictionary for ModelEmb
    model_args = {
        'order': 85,            # HarDNet-85 architecture (larger model variant)
        'depth_wise': 0,        # Use regular convolutions (0 = False)
        'nP': 8,                # Number of points parameter (not directly used in ModelEmb)
        'output_dim': 256       # Output dimension for embeddings
    }
    
    model = ModelEmb(args=model_args).to(device)
    
    # Print model structure
    model_params = sum(p.numel() for p in model.parameters())
    print(f"ModelEmb parameters: {model_params:,}")
    
    # Load pretrained weights if specified
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
    
    # Set up optimizer with lower learning rate and higher weight decay
    initial_lr = args.lr
    weight_decay = args.weight_decay
    print(f"Using learning rate: {initial_lr} and weight decay: {weight_decay}")
    
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    # Set up criterion with increased weights for non-empty predictions
    pos_weight = 10.0  # Increased from 5.0
    criterion = CombinedBCEDiceLoss(
        bce_weight=0.7,
        dice_weight=0.3,
        pos_weight=pos_weight,
        empty_penalty=5.0  # Add penalty for empty predictions
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        # Additional video validation if requested
        if True:
            video_loss = evaluate_video(
                model=model,
                val_dataset=val_dataset,
                sam2_video=sam2_video,  # Use the video predictor
                device=device,
                output_dir=os.path.join(args.output_dir, 'video_eval')
            )
        # Train
        train_loss, train_dice = train_one_epoch(
            model=model,
            train_loader=train_loader,
            sam2=sam2_image,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch
        )
        
        # Validate with both image and video models
        val_loss, val_dice = evaluate(
            model=model,
            val_loader=val_loader,
            sam2=sam2_image,
            criterion=criterion,
            device=device
        )
        
        # Additional video validation if requested
        if args.eval_video:
            video_loss = evaluate_video(
                model=model,
                val_dataset=val_dataset,
                sam2_video=sam2_video,  # Use the video predictor
                device=device,
                output_dir=os.path.join(args.output_dir, 'video_eval')
            )
            print(f"Video Validation - Loss: {video_loss:.4f}")
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        
        # Visualize predictions
        visualize_predictions(
            model=model,
            val_loader=val_loader,
            sam2=sam2_image,
            device=device,
            epoch=epoch+1,
            output_dir=args.output_dir
        )
        
        # Save checkpoint based on image validation score
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,
            dice_score=val_dice,
            checkpoint_dir=checkpoint_dir
        )
        
        # Save best model based on image validation
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_dir, 'best_model.pth')
            )
            print(f"Saved best model with Dice score: {best_dice:.4f}")
    
    print("\nTraining completed!")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AutoSAM2 ModelEmb")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="polyp_gen", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--eval_video", action="store_true", help="Evaluate on video data after training")
    
    # Distributed training parameters
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    
    args = parser.parse_args()
    main(args)