import os
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from sam2.build_sam import get_sam2_model

# Import dataset utilities
from dataset.polyp_gen import get_polyp_gen_dataset


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


def prepare_batch_for_sam2(batch, device, output_format='image'):
    """
    Prepare batch data for SAM2 processing
    
    Args:
        batch: Batch from dataloader containing VideoDatapoint objects
        device: Device to use
        output_format: Output format ('image' or 'video')
        
    Returns:
        images: Images tensor
        masks: Ground truth masks tensor
        video_ids: Optional video IDs
    """
    # Handle VideoDatapoint objects
    if hasattr(batch[0], 'frames'):
        # We have a batch of VideoDatapoint objects
        batch_size = len(batch)
        
        # Extract images and masks from VideoDatapoint objects
        images = []
        masks = []
        video_ids = []
        
        for i in range(batch_size):
            video_datapoint = batch[i]
            video_ids.append(video_datapoint.video_id)
            
            # Extract frames
            frames = video_datapoint.frames
            
            for frame in frames:
                # Add image data
                images.append(frame.data)
                
                # Extract mask from the first object in the frame
                if frame.objects and len(frame.objects) > 0:
                    # Convert boolean mask to float
                    mask = frame.objects[0].segment.float()
                    masks.append(mask)
                else:
                    # Create empty mask if no objects
                    mask_shape = list(frame.data.shape)
                    mask_shape[0] = 1  # Single channel for mask
                    masks.append(torch.zeros(mask_shape))
        
        # Stack images and masks
        images = torch.stack(images).to(device)
        masks = torch.stack(masks).to(device)
        
        # Handle video data format if needed (5D tensor)
        if len(images.shape) == 5:  # [B, T, C, H, W]
            B, T, C, H, W = images.shape
            if output_format == 'image':
                # For training, treat each frame as a separate image
                images = images.reshape(B*T, C, H, W)
                masks = masks.reshape(B*T, 1, H, W)
            # else, keep the video format
        
        return images, masks, video_ids
    
    elif isinstance(batch, dict) and 'image' in batch:
        # Format with 'image', 'mask' keys
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        video_ids = batch.get('video_id', None)
    else:
        # Handle other formats
        images = batch[0].to(device)
        masks = batch[1].to(device)
        video_ids = None
    
    return images, masks, video_ids


def sam_call(images, sam2, dense_embeddings, state=None):
    """
    Generate masks using SAM2 with dense embeddings from our ModelEmb
    
    Args:
        images: Input images (B, C, H, W)
        sam2: SAM2ImagePredictor instance
        dense_embeddings: Dense embeddings from our model (B, C, H, W)
        state: State for prompting (default: None)
        
    Returns:
        low_res_masks: Generated masks
    """
    B = images.shape[0]
    
    with torch.no_grad():  # Ensure no gradients flow through SAM2
        # Extract image features
        image_embeddings = sam2.image_encoder(images)
        
        # Process backbone features
        image_embeddings["backbone_fpn"][0] = sam2.sam_mask_decoder.conv_s0(
            image_embeddings["backbone_fpn"][0]
        )
        image_embeddings["backbone_fpn"][1] = sam2.sam_mask_decoder.conv_s1(
            image_embeddings["backbone_fpn"][1]
        )

        # Prepare features for the decoder
        image_embeddings, vision_feats, vision_pos_embeds, feat_sizes = sam2._prepare_backbone_features(image_embeddings)

        # Extract high resolution features if available
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        # Default empty prompt (we're using dense embeddings)
        sam_point_coords = torch.zeros(B, 1, 2, device=sam2.device)
        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=sam2.device)
        
        # Get sparse embeddings (empty)
        sparse_embeddings, _ = sam2.sam_prompt_encoder(
            points=(sam_point_coords, sam_point_labels),
            boxes=None,
            masks=None
        )
        
        # Generate masks using the decoder
        low_res_masks, _ = sam2.sam_mask_decoder(
            image_embeddings=image_embeddings,
            multi_scale_features=high_res_features,
            vision_features=vision_feats[-1].permute(1, 0, 2),
            vision_pos_embeds=vision_pos_embeds[-1].permute(1, 0, 2),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,  # Use our dense embeddings here
            multimask_output=False,
        )
        
    return low_res_masks


def train_one_epoch(model, train_loader, sam2, optimizer, criterion, device, epoch):
    """
    Train model for one epoch
    
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
    
    # Ensure SAM2 is in eval mode and frozen
    sam2.model.eval()
    
    running_loss = 0.0
    dice_scores = []
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(progress_bar):
        # Prepare batch
        images, masks, _ = prepare_batch_for_sam2(batch, device, output_format='image')
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass through our embedding model (only this will be trained)
        dense_embeddings = model(images)
        
        # Generate masks using SAM2 (with no_grad to ensure SAM2 is not trained)
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
        
        # Backward pass (only ModelEmb will be updated)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        pred_masks = (mask_preds > 0).float()
        batch_dice = 1 - dice_loss(masks, pred_masks).item()
        dice_scores.append(batch_dice)
        
        # Update progress bar
        progress_bar.set_postfix(loss=running_loss/(batch_idx+1), dice=np.mean(dice_scores))
    
    return running_loss / len(train_loader), np.mean(dice_scores)


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


def evaluate_video(model, val_loader, sam2_video, device, output_dir):
    """
    Evaluate model on video data using SAM2VideoPredictor
    
    Args:
        model: ModelEmb model (to be evaluated)
        val_loader: Validation dataloader with video data
        sam2_video: SAM2VideoPredictor (frozen)
        device: Device
        output_dir: Output directory for results
        
    Returns:
        avg_dice: Average Dice score
    """
    # Set model to evaluation mode
    model.eval()
    
    # Ensure SAM2 is in eval mode
    sam2_video.model.eval()
    
    dice_scores = []
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc='Video Evaluation')):
            # Handle VideoDatapoint objects from PolypGenDataset
            if hasattr(batch[0], 'frames'):
                # Process each VideoDatapoint separately
                for video_datapoint in batch:
                    video_id = video_datapoint.video_id
                    frames = video_datapoint.frames
                    
                    # Extract images and masks
                    images = torch.stack([frame.data for frame in frames]).unsqueeze(0).to(device)  # [1, T, C, H, W]
                    masks = torch.stack([frame.objects[0].segment.float() for frame in frames]).unsqueeze(0).to(device)  # [1, T, 1, H, W]
                    
                    B, T, C, H, W = images.shape
                    
                    # Create memory dictionary for SAM2 video predictor
                    output_dict = {
                        "cond_frame_outputs": {},
                        "non_cond_frame_outputs": {}
                    }
                    
                    # Process each frame
                    all_preds = []
                    
                    for t in range(T):
                        # Get current frame
                        frame = images[:, t]  # [B, C, H, W]
                        
                        # Get embeddings from our model
                        dense_embeddings = model(frame)
                        
                        # Process first frame with SAM2
                        if t == 0:
                            # Generate mask for first frame
                            mask_pred = sam_call(frame, sam2_video, dense_embeddings)
                            
                            # Store in memory structure
                            frame_embedding = sam2_video.image_encoder(frame)
                            
                            # Process the backbone feature pyramid network outputs
                            frame_embedding["backbone_fpn"][0] = sam2_video.sam_mask_decoder.conv_s0(
                                frame_embedding["backbone_fpn"][0]
                            )
                            frame_embedding["backbone_fpn"][1] = sam2_video.sam_mask_decoder.conv_s1(
                                frame_embedding["backbone_fpn"][1]
                            )
                            
                            # Prepare features
                            frame_embedding, vision_feats, vision_pos_embeds, feat_sizes = sam2_video._prepare_backbone_features(frame_embedding)
                            
                            # Extract object pointer
                            B_frame = frame.shape[0]
                            
                            # Default empty prompt
                            sam_point_coords = torch.zeros(B_frame, 1, 2, device=device)
                            sam_point_labels = -torch.ones(B_frame, 1, dtype=torch.int32, device=device)
                            
                            # Get sparse embeddings
                            sparse_embeddings, _ = sam2_video.sam_prompt_encoder(
                                points=(sam_point_coords, sam_point_labels),
                                boxes=None,
                                masks=None
                            )
                            
                            # Generate masks
                            low_res_masks, ious, sam_output_tokens, object_score_logits = sam2_video.sam_mask_decoder(
                                image_embeddings=frame_embedding,
                                multi_scale_features=None,
                                vision_features=vision_feats[-1].permute(1, 0, 2),
                                vision_pos_embeds=vision_pos_embeds[-1].permute(1, 0, 2),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                            )
                            
                            # Extract output tokens
                            sam_output_token = sam_output_tokens[:, 0]
                            obj_ptr = sam2_video.obj_ptr_proj(sam_output_token)
                            
                            # Encode memory for first frame
                            maskmem_features, maskmem_pos_enc = sam2_video._encode_new_memory(
                                vision_feats,
                                feat_sizes,
                                low_res_masks,
                                object_score_logits,
                                False,
                            )
                            
                            # Store in memory
                            output_dict["cond_frame_outputs"][t] = {
                                "obj_ptr": obj_ptr,
                                "maskmem_features": maskmem_features,
                                "maskmem_pos_enc": maskmem_pos_enc
                            }
                            
                            mask_pred = low_res_masks
                            
                        else:
                            # For subsequent frames, use the memory conditioning
                            frame_embedding = sam2_video.image_encoder(frame)
                            
                            # Process features
                            frame_embedding["backbone_fpn"][0] = sam2_video.sam_mask_decoder.conv_s0(
                                frame_embedding["backbone_fpn"][0]
                            )
                            frame_embedding["backbone_fpn"][1] = sam2_video.sam_mask_decoder.conv_s1(
                                frame_embedding["backbone_fpn"][1]
                            )
                            
                            # Prepare features
                            frame_embedding, vision_feats, vision_pos_embeds, feat_sizes = sam2_video._prepare_backbone_features(frame_embedding)
                            
                            # Use memory conditioning
                            pix_feat = sam2_video._prepare_memory_conditioned_features(
                                frame_idx=t,
                                is_init_cond_frame=False,  # Not the first frame
                                current_vision_feats=vision_feats[-1:],
                                current_vision_pos_embeds=vision_pos_embeds[-1:],
                                feat_sizes=feat_sizes[-1:],
                                output_dict=output_dict,
                                num_frames=T,
                            )
                            
                            # Default empty prompt
                            sam_point_coords = torch.zeros(B_frame, 1, 2, device=device)
                            sam_point_labels = -torch.ones(B_frame, 1, dtype=torch.int32, device=device)
                            
                            # Get sparse embeddings
                            sparse_embeddings, _ = sam2_video.sam_prompt_encoder(
                                points=(sam_point_coords, sam_point_labels),
                                boxes=None,
                                masks=None
                            )
                            
                            # Generate masks using memory conditioning
                            low_res_masks, ious, sam_output_tokens, object_score_logits = sam2_video.sam_mask_decoder(
                                image_embeddings=pix_feat,
                                multi_scale_features=None,
                                vision_features=vision_feats[-1].permute(1, 0, 2),
                                vision_pos_embeds=vision_pos_embeds[-1].permute(1, 0, 2),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=False,
                            )
                            
                            # Extract output tokens
                            sam_output_token = sam_output_tokens[:, 0]
                            obj_ptr = sam2_video.obj_ptr_proj(sam_output_token)
                            
                            # Encode memory for this frame
                            maskmem_features, maskmem_pos_enc = sam2_video._encode_new_memory(
                                vision_feats,
                                feat_sizes,
                                low_res_masks,
                                object_score_logits,
                                False,
                            )
                            
                            # Store in non-conditioning memory
                            output_dict["non_cond_frame_outputs"][t] = {
                                "obj_ptr": obj_ptr,
                                "maskmem_features": maskmem_features,
                                "maskmem_pos_enc": maskmem_pos_enc
                            }
                            
                            mask_pred = low_res_masks
                        
                        # Store prediction
                        all_preds.append(mask_pred)
                    
                    # Stack predictions
                    all_preds = torch.stack(all_preds, dim=1)  # [B, T, 1, H, W]
                    
                    # Upscale if needed
                    if all_preds.shape[-2:] != (H, W):
                        # Reshape for interpolation
                        all_preds_flat = all_preds.reshape(B*T, 1, all_preds.shape[-2], all_preds.shape[-1])
                        all_preds_flat = F.interpolate(all_preds_flat, size=(H, W), mode='bilinear', align_corners=False)
                        all_preds = all_preds_flat.reshape(B, T, 1, H, W)
                    
                    # Calculate metrics for the video
                    video_dice = []
                    for t in range(T):
                        pred = (all_preds[0, t] > 0).float()
                        target = masks[0, t]
                        frame_dice = 1 - dice_loss(target, pred).item()
                        video_dice.append(frame_dice)
                    
                    # Average dice for video
                    avg_video_dice = np.mean(video_dice)
                    dice_scores.append(avg_video_dice)
                    
                    # Save some frames as visualization
                    video_dir = os.path.join(output_dir, f'video_{video_id}')
                    os.makedirs(video_dir, exist_ok=True)
                    
                    # Save metrics
                    with open(os.path.join(video_dir, 'metrics.json'), 'w') as f:
                        json.dump({'dice': avg_video_dice}, f)
            else:
                # Get video data (original implementation for non-VideoDatapoint)
                if isinstance(batch, dict) and 'image' in batch:
                    videos = batch['image'].to(device)
                    masks = batch['mask'].to(device)
                else:
                    videos = batch[0].to(device)
                    masks = batch[1].to(device)
                
                # Ensure data is in video format [B, T, C, H, W]
                if len(videos.shape) != 5:
                    videos = videos.unsqueeze(1)  # Add time dimension
                    masks = masks.unsqueeze(1)
                
                B, T, C, H, W = videos.shape
                
                # Create memory dictionary for SAM2 video predictor
                output_dict = {
                    "cond_frame_outputs": {},
                    "non_cond_frame_outputs": {}
                }
                
                # Process each frame
                all_preds = []
                
                for t in range(T):
                    # Get current frame
                    frame = videos[:, t]  # [B, C, H, W]
                    
                    # Get embeddings from our model
                    dense_embeddings = model(frame)
                    
                    # Process first frame with SAM2
                    if t == 0:
                        # Generate mask for first frame
                        mask_pred = sam_call(frame, sam2_video, dense_embeddings)
                        
                        # Store in memory structure
                        frame_embedding = sam2_video.image_encoder(frame)
                        
                        # Process the backbone feature pyramid network outputs
                        frame_embedding["backbone_fpn"][0] = sam2_video.sam_mask_decoder.conv_s0(
                            frame_embedding["backbone_fpn"][0]
                        )
                        frame_embedding["backbone_fpn"][1] = sam2_video.sam_mask_decoder.conv_s1(
                            frame_embedding["backbone_fpn"][1]
                        )
                        
                        # Prepare features
                        frame_embedding, vision_feats, vision_pos_embeds, feat_sizes = sam2_video._prepare_backbone_features(frame_embedding)
                        
                        # Extract object pointer
                        B_frame = frame.shape[0]
                        
                        # Default empty prompt
                        sam_point_coords = torch.zeros(B_frame, 1, 2, device=device)
                        sam_point_labels = -torch.ones(B_frame, 1, dtype=torch.int32, device=device)
                        
                        # Get sparse embeddings
                        sparse_embeddings, _ = sam2_video.sam_prompt_encoder(
                            points=(sam_point_coords, sam_point_labels),
                            boxes=None,
                            masks=None
                        )
                        
                        # Generate masks
                        low_res_masks, ious, sam_output_tokens, object_score_logits = sam2_video.sam_mask_decoder(
                            image_embeddings=frame_embedding,
                            multi_scale_features=None,
                            vision_features=vision_feats[-1].permute(1, 0, 2),
                            vision_pos_embeds=vision_pos_embeds[-1].permute(1, 0, 2),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )
                        
                        # Extract output tokens
                        sam_output_token = sam_output_tokens[:, 0]
                        obj_ptr = sam2_video.obj_ptr_proj(sam_output_token)
                        
                        # Encode memory for first frame
                        maskmem_features, maskmem_pos_enc = sam2_video._encode_new_memory(
                            vision_feats,
                            feat_sizes,
                            low_res_masks,
                            object_score_logits,
                            False,
                        )
                        
                        # Store in memory
                        output_dict["cond_frame_outputs"][t] = {
                            "obj_ptr": obj_ptr,
                            "maskmem_features": maskmem_features,
                            "maskmem_pos_enc": maskmem_pos_enc
                        }
                        
                        mask_pred = low_res_masks
                        
                    else:
                        # For subsequent frames, use the memory conditioning
                        frame_embedding = sam2_video.image_encoder(frame)
                        
                        # Process features
                        frame_embedding["backbone_fpn"][0] = sam2_video.sam_mask_decoder.conv_s0(
                            frame_embedding["backbone_fpn"][0]
                        )
                        frame_embedding["backbone_fpn"][1] = sam2_video.sam_mask_decoder.conv_s1(
                            frame_embedding["backbone_fpn"][1]
                        )
                        
                        # Prepare features
                        frame_embedding, vision_feats, vision_pos_embeds, feat_sizes = sam2_video._prepare_backbone_features(frame_embedding)
                        
                        # Use memory conditioning
                        pix_feat = sam2_video._prepare_memory_conditioned_features(
                            frame_idx=t,
                            is_init_cond_frame=False,  # Not the first frame
                            current_vision_feats=vision_feats[-1:],
                            current_vision_pos_embeds=vision_pos_embeds[-1:],
                            feat_sizes=feat_sizes[-1:],
                            output_dict=output_dict,
                            num_frames=T,
                        )
                        
                        # Default empty prompt
                        sam_point_coords = torch.zeros(B_frame, 1, 2, device=device)
                        sam_point_labels = -torch.ones(B_frame, 1, dtype=torch.int32, device=device)
                        
                        # Get sparse embeddings
                        sparse_embeddings, _ = sam2_video.sam_prompt_encoder(
                            points=(sam_point_coords, sam_point_labels),
                            boxes=None,
                            masks=None
                        )
                        
                        # Generate masks using memory conditioning
                        low_res_masks, ious, sam_output_tokens, object_score_logits = sam2_video.sam_mask_decoder(
                            image_embeddings=pix_feat,
                            multi_scale_features=None,
                            vision_features=vision_feats[-1].permute(1, 0, 2),
                            vision_pos_embeds=vision_pos_embeds[-1].permute(1, 0, 2),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=False,
                        )
                        
                        # Extract output tokens
                        sam_output_token = sam_output_tokens[:, 0]
                        obj_ptr = sam2_video.obj_ptr_proj(sam_output_token)
                        
                        # Encode memory for this frame
                        maskmem_features, maskmem_pos_enc = sam2_video._encode_new_memory(
                            vision_feats,
                            feat_sizes,
                            low_res_masks,
                            object_score_logits,
                            False,
                        )
                        
                        # Store in non-conditioning memory
                        output_dict["non_cond_frame_outputs"][t] = {
                            "obj_ptr": obj_ptr,
                            "maskmem_features": maskmem_features,
                            "maskmem_pos_enc": maskmem_pos_enc
                        }
                        
                        mask_pred = low_res_masks
                    
                    # Store prediction
                    all_preds.append(mask_pred)
                
                # Stack predictions
                all_preds = torch.stack(all_preds, dim=1)  # [B, T, 1, H, W]
                
                # Upscale if needed
                if all_preds.shape[-2:] != (H, W):
                    # Reshape for interpolation
                    all_preds_flat = all_preds.reshape(B*T, 1, all_preds.shape[-2], all_preds.shape[-1])
                    all_preds_flat = F.interpolate(all_preds_flat, size=(H, W), mode='bilinear', align_corners=False)
                    all_preds = all_preds_flat.reshape(B, T, 1, H, W)
                
                # Calculate metrics for the video
                for b in range(B):
                    video_dice = []
                    for t in range(T):
                        pred = (all_preds[b, t] > 0).float()
                        target = masks[b, t]
                        frame_dice = 1 - dice_loss(target, pred).item()
                        video_dice.append(frame_dice)
                    
                    # Average dice for video
                    avg_video_dice = np.mean(video_dice)
                    dice_scores.append(avg_video_dice)
                    
                    # Save some frames as visualization
                    video_dir = os.path.join(output_dir, f'video_{batch_idx}_{b}')
                    os.makedirs(video_dir, exist_ok=True)
                    
                    # Save metrics
                    with open(os.path.join(video_dir, 'metrics.json'), 'w') as f:
                        json.dump({'dice': avg_video_dice}, f)
    
    # Return average dice across all videos
    avg_dice = np.mean(dice_scores)
    print(f'Video Evaluation - Average Dice: {avg_dice:.4f}')
    return avg_dice


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
    # Set up device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f'Using device: {device}')
    
    # Create directories for results
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load SAM2 model using from_pretrained method
    print("Loading SAM2.1 model with Hiera Large backbone...")
    
    # Create SAM2 Image Predictor for training using from_pretrained
    sam2_image = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
    
    # Freeze all SAM2 components
    for param in sam2_image.model.parameters():
        param.requires_grad = False
    
    # Ensure SAM2 is in eval mode
    sam2_image.model.eval()
    print("SAM2 model loaded and frozen (set to evaluation mode)")
    
    # Set up transformations
    transform = SAM2Transforms(sam2_image.model.image_size)
    
    # Load datasets
    print(f"Loading {args.dataset} dataset...")
    train_dataset, val_dataset = get_polyp_gen_dataset(sam_trans=transform, base_path=args.dataset)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize our model
    print("Creating AutoSAM2 ModelEmb...")
    model = ModelEmb(image_size=sam2_image.model.image_size, output_dim=256).to(device)
    
    # Load pretrained weights if specified
    if args.pretrained:
        print(f"Loading pretrained weights from {args.pretrained}")
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
    
    # Set up optimizer - only optimize ModelEmb parameters
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Set up criterion
    criterion = FocalLoss() if args.focal_loss else nn.BCEWithLogitsLoss()
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    best_dice = 0.0
    
    for epoch in range(args.epochs):
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
        
        # Validate
        val_loss, val_dice = evaluate(
            model=model,
            val_loader=val_loader,
            sam2=sam2_image,
            criterion=criterion,
            device=device
        )
        
        # Print metrics
        print(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Epoch {epoch} - Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=val_loss,
            dice_score=val_dice,
            checkpoint_dir=checkpoint_dir
        )
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with Dice score: {best_dice:.4f}")
    
    # Evaluate on video data if requested
    if args.eval_video:
        print("Evaluating on video data...")
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
        
        # Create SAM2 Video Predictor using from_pretrained
        sam2_video = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large", device=device)
        
        # Freeze SAM2 Video Predictor
        for param in sam2_video.model.parameters():
            param.requires_grad = False
        sam2_video.model.eval()
        
        # Create video output directory
        video_output_dir = os.path.join(args.output_dir, 'video_eval')
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Evaluate on video data
        video_dice = evaluate_video(
            model=model,
            val_loader=val_loader,  # Assuming val_loader has video data
            sam2_video=sam2_video,
            device=device,
            output_dir=video_output_dir
        )
        
        print(f"Video Evaluation - Dice Score: {video_dice:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AutoSAM2 model')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='polyp_gen', help='Dataset name')
    
    # Model parameters
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained ModelEmb weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--focal_loss', action='store_true', help='Use focal loss instead of BCE')
    
    # Hardware parameters
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    
    # Evaluation parameters
    parser.add_argument('--eval_video', action='store_true', help='Evaluate on video data')
    
    args = parser.parse_args()
    
    main(args)