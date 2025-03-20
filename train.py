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
NO_OBJ_SCORE = -1024.0 # Score for empty object predictions
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
        batch: List of VideoDatapoint objects or single VideoDatapoint
        device: Torch device to move tensors to
        output_format: 'video' for [B, T, C, H, W] format, 'image' for [B, C, H, W]
    
    Returns:
        images: Tensor of images [B, T, C, H, W] or [B, C, H, W]
        masks: Tensor of masks [B, T, C, H, W] or [B, C, H, W]
        ids: Tensor of video/image IDs
    """
    try:
        
        # Convert single VideoDatapoint to list for consistent processing
        if isinstance(batch, VideoDatapoint):
            print("Converting single VideoDatapoint to list")
            batch = [batch]
        elif isinstance(batch, dict):
        
            images = []
            masks = []
            ids = []
            
            # Set a fixed target size for all images and masks
            TARGET_SIZE = (1024, 1024)  # Common size for SAM models
            if output_format == 'video':
                # Extract frames and masks directly (since we printed their format)
                images = batch['frames'].to(device)  # [B, T, C, H, W]
                masks = batch['masks'].to(device)  # [B, T, H, W]
                ids = torch.tensor(batch['video_id']).to(device)

                # 🔹 Ensure masks have a channel dimension [B, T, 1, H, W]
                masks = masks.unsqueeze(2)  # Add channel dimension

                B, T, C, H, W = images.shape
                images = images.view(B * T, C, H, W)  # Flatten batch for resizing
                images = F.interpolate(images, size=TARGET_SIZE, mode='bilinear', align_corners=False)
                images = images.view(B, T, C, TARGET_SIZE[0], TARGET_SIZE[1])  # Restore batch & time

                masks = masks.view(B * T, 1, H, W)  # Flatten batch
                masks = F.interpolate(masks, size=TARGET_SIZE, mode='nearest')
                masks = masks.view(B, T, 1, TARGET_SIZE[0], TARGET_SIZE[1])  # Restore batch & time

                print(f"Final images shape: {images.shape}")  # [B, T, C, 1024, 1024]
                print(f"Final masks shape: {masks.shape}")  # [B, T, 1, 1024, 1024]
                images = torch.stack(images).to(device)
                masks = torch.stack(masks).to(device)
                ids = torch.tensor(batch['video_id']).to(device)
                return images, masks, ids
            
            else:# Process frames from the dictionary
                for frame, mask in zip(batch['frames'], batch['masks']):
                    print(f"Processing frame shape: {frame.shape}, mask shape: {mask.shape}")
                    
                    # Resize image to the target size
                    img_resized = F.interpolate(frame.unsqueeze(0), size=TARGET_SIZE, 
                                            mode='bilinear', align_corners=False).squeeze(0)
                    images.append(img_resized)
                    
                    # Resize mask to match the target size
                    mask_resized = F.interpolate(mask.unsqueeze(0), size=TARGET_SIZE, 
                                            mode='nearest').squeeze(0)
                    masks.append(mask_resized)
            
            # Stack resized images and masks
            images = torch.stack(images).to(device)
            masks = torch.stack(masks).to(device)
            ids = torch.tensor(batch['video_id']).to(device)
            
            
            # For video format, ensure 5D tensor [B, T, C, H, W]
            if output_format == 'video':
                if len(images.shape) == 4:  # [B, C, H, W]
                    print("Adding time dimension to images")
                    images = images.unsqueeze(1)  # Add time dimension [B, 1, C, H, W]
                if len(masks.shape) == 4:  # [B, C, H, W]
                    print("Adding time dimension to masks")
                    masks = masks.unsqueeze(1)  # Add time dimension [B, 1, C, H, W]
            
            print(f"Final images shape: {images.shape}")
            print(f"Final masks shape: {masks.shape}")
            return images, masks, ids
        
        images = []
        masks = []
        ids = []
        
        # Set a fixed target size for all images and masks
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
        

        
        # For video format, ensure 5D tensor [B, T, C, H, W]
        if output_format == 'video':
            if len(images.shape) == 4:  # [B, C, H, W]
                print("Adding time dimension to images")
                images = images.unsqueeze(1)  # Add time dimension [B, 1, C, H, W]
            if len(masks.shape) == 4:  # [B, C, H, W]
                print("Adding time dimension to masks")
                masks = masks.unsqueeze(1)  # Add time dimension [B, 1, C, H, W]
        
        ids = torch.tensor(ids).to(device)
        

        
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


import torch
import torch.nn.functional as F

def sam_video_call(videos, sam2, dense_embeddings, output_dict, t, shape):
    """
    Perform video segmentation on a specific frame using the SAM2 model.
    This function always uses the provided `dense_embeddings` (equivalent to `state == 'autosam2'`).

    Args:
        videos (torch.Tensor): The input video tensor of shape [B, T, C, H, W].
        sam2 (SAM2VideoPredictor): The SAM2 video predictor model.
        dense_embeddings (torch.Tensor): The dense embeddings used for segmentation.
        output_dict (dict): Dictionary to store frame-wise outputs.
        t (int): The current frame index.
        shape (tuple): Shape of the input tensor (B, T, C, H, W).

    Returns:
        torch.Tensor: Low-resolution segmentation masks for the frame `t`.
    """
    B, T, C, H, W = shape  # Unpack the input shape
    current_out = {}  # Dictionary to store the current frame's outputs
    
    with torch.no_grad():
        # Extract image features using the SAM2 image encoder for the current frame
        image_embeddings = sam2.image_encoder(videos[:, t])
        
        # Process features through the SAM2 mask decoder's convolution layers
        image_embeddings["backbone_fpn"][0] = sam2.sam_mask_decoder.conv_s0(
            image_embeddings["backbone_fpn"][0]
        )
        image_embeddings["backbone_fpn"][1] = sam2.sam_mask_decoder.conv_s1(
            image_embeddings["backbone_fpn"][1]
        )
        
        # Prepare feature maps for segmentation
        image_embeddings, vision_feats, vision_pos_embeds, feat_sizes = sam2._prepare_backbone_features(image_embeddings)
        
        # Extract high-resolution features if available
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        
        # Prepare default empty point inputs (not used, but required by API)
        sam_point_coords = torch.zeros(B, 1, 2, device=sam2.device)
        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=sam2.device)
        
        # Encode the input prompt embeddings (always using dense embeddings provided)
        sparse_embeddings, _ = sam2.sam_prompt_encoder(points=(sam_point_coords, sam_point_labels), boxes=None, masks=None)
    
        # Process the memory-conditioned features for tracking
        pix_feat = sam2._prepare_memory_conditioned_features(
            frame_idx=t,
            is_init_cond_frame=(t == 0),  # First frame is a conditioning frame
            current_vision_feats=vision_feats[-1:],
            current_vision_pos_embeds=vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=output_dict,
            num_frames=T,
        )    

        # Ensure dense embeddings match spatial dimensions of pix_feat
        _, _, target_height, target_width = pix_feat.shape
        dense_embeddings_resized = F.interpolate(
            dense_embeddings,  
            size=(target_height, target_width),  
            mode="bilinear",  
            align_corners=False
        )

        # Generate segmentation masks using the SAM2 mask decoder
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = sam2.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings_resized,  # Use resized embeddings
            multimask_output=False,
            repeat_image=False,  
            high_res_features=high_res_features,
        )   

        # Determine object presence and update mask scores accordingly
        is_obj_appearing = object_score_logits > 0
        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            torch.tensor(-1024.0, device=low_res_multimasks.device),  # Replace NO_OBJ_SCORE with actual tensor
        )
        
        # Convert masks to float for further processing
        low_res_multimasks = low_res_multimasks.float()
        
        # Upsample masks to match original image resolution dynamically
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(sam2.image_size, sam2.image_size),  # Dynamically resize to original input image dimensions
            mode="bilinear",
            align_corners=False,
        )
        
        # Extract object pointers from SAM output tokens
        sam_output_token = sam_output_tokens[:, 0]
        obj_ptr = sam2.obj_ptr_proj(sam_output_token)
        
        # Store predictions in current frame output dictionary
        current_out["pred_masks"] = low_res_multimasks
        current_out["pred_masks_high_res"] = high_res_multimasks
        current_out["obj_ptr"] = obj_ptr

        # Encode memory for future frames
        maskmem_features, maskmem_pos_enc = sam2._encode_new_memory(
            vision_feats,
            feat_sizes,
            high_res_multimasks,
            object_score_logits,
            False,
        )
        
        # Store frame output in appropriate dictionary (conditioning vs. non-conditioning)
        if t == 0:
            output_dict["cond_frame_outputs"][t] = {
                "obj_ptr": obj_ptr,
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": maskmem_pos_enc
            }
        else:
            output_dict["non_cond_frame_outputs"][t] = {
                "obj_ptr": obj_ptr,
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": maskmem_pos_enc
            }
    
    return low_res_multimasks

def evaluate_video(model, val_loader, sam2_video, criterion, device):
    """
    Evaluate the model on a video dataset using the SAM2VideoPredictor.

    Args:
        model: The embedding model (ModelEmb instance).
        val_loader: Video validation DataLoader.
        sam2_video: The SAM2VideoPredictor instance.
        criterion: Loss function (e.g., Dice loss, BCE).
        device: Device to run inference (CUDA/CPU).

    Returns:
        avg_loss (float): Average loss across all videos.
        avg_dice (float): Average Dice score across all videos.
    """
    # Set models to evaluation mode
    model.eval()
    sam2_video.image_encoder.eval()
    sam2_video.memory_attention.eval()
    sam2_video.memory_encoder.eval()

    running_loss = 0.0
    dice_scores = []

    # Initialize progress bar
    progress_bar = tqdm(val_loader, desc='Video Validation')

    with torch.no_grad():
        for batch_idx, (video_tensor, mask_tensor, video_id) in enumerate(progress_bar):
            try:
                print(f"[DEBUG] Processing batch {batch_idx} - Video ID: {video_id}")
                print(f"[DEBUG] Video Shape: {video_tensor.shape} (Expected: [B, T, C, H, W])")
                print(f"[DEBUG] Mask Shape: {mask_tensor.shape} (Expected: [B, T, 1, H, W])")

                B, T, C, H, W = video_tensor.shape

                # Wrap `video_tensor` and `mask_tensor` into a VideoDatapoint
                frames = []
                for t in range(T):
                    frame_data = video_tensor[:, t]  # Shape: [B, C, H, W]
                    mask_data = mask_tensor[:, t]  # Shape: [B, 1, H, W]
                    frame = Frame(data=frame_data, objects=[Object(object_id=0, frame_index=t, segment=mask_data)])
                    frames.append(frame)

                # Create a VideoDatapoint object
                batch_video_dp = VideoDatapoint(frames=frames, video_id=video_id, size=(W, H))

                # Now call `prepare_batch_for_sam2` with the correct input
                videos, masks, _ = prepare_batch_for_sam2(batch_video_dp, device, output_format='video')

                print(f"[DEBUG] Prepared batch for SAM2 - Video Shape: {videos.shape}, Mask Shape: {masks.shape}")

                # Get dense embeddings
                B, T, C, H, W = videos.shape
                videos_reshaped = videos.view(-1, C, H, W)  # Flatten for embedding model
                dense_embeddings = model(videos_reshaped)
                dense_embeddings = dense_embeddings.view(B, T, -1, H, W)  # Reshape back to video format

                print(f"[DEBUG] Dense embeddings shape: {dense_embeddings.shape}")

                # Initialize memory dictionary for SAM2 processing
                output_dict = {
                    "cond_frame_outputs": {},
                    "non_cond_frame_outputs": {}
                }

                all_preds = []

                for t in range(T):
                    current_embeddings = dense_embeddings[:, t]  # Corresponding dense embedding
                    mask_preds = sam_video_call(
                        videos, sam2_video, current_embeddings, output_dict, t, (B, T, C, H, W)
                    )
                    all_preds.append(mask_preds)

                all_preds = torch.stack(all_preds, dim=1)
                print(f"[DEBUG] Stacked predictions shape: {all_preds.shape}")

                # Ensure prediction mask size matches ground truth mask size
                if all_preds.shape[-2:] != masks.shape[-2:]:  # Compare only H and W
                    all_preds = all_preds.view(-1, all_preds.shape[2], all_preds.shape[3], all_preds.shape[4])
                    all_preds = F.interpolate(
                        all_preds, size=(masks.shape[-2], masks.shape[-1]), mode='bilinear', align_corners=False
                    )
                    all_preds = all_preds.view(B, T, C, masks.shape[2], masks.shape[3])

                # Compute loss
                loss = criterion(all_preds, masks)
                running_loss += loss.item()

                # Compute Dice score
                pred_masks = (all_preds > 0).float()
                batch_dice = 1 - dice_loss(masks, pred_masks).item()
                dice_scores.append(batch_dice)

                progress_bar.set_postfix(loss=running_loss / (batch_idx + 1), dice=np.mean(dice_scores))

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

    avg_loss = running_loss / len(val_loader)
    avg_dice = np.mean(dice_scores)

    print(f"\nEvaluation Complete - Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}")

    return avg_loss, avg_dice




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
        combined_loss = self.bce_weight * torch.clamp(bce_loss, 0, 100) + self.dice_weight * dice_loss + empty_loss
        
        return combined_loss


def train_one_epoch(model, train_loader, sam2, optimizer, criterion, device, epoch, accumulation_steps=8):
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
        accumulation_steps: Number of steps to accumulate gradients before performing optimization step
        
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
                print(f"Skipping batch with no/only foreground: {mask_sum} / {mask_size} pixels")
                continue
            
            # Increase minimum foreground percentage threshold
            fg_percent = 100.0 * mask_sum / mask_size
            if fg_percent < 0.5:  # Increased from 0.1% to 0.5%
                print(f"Skipping batch with low foreground: {fg_percent:.2f}%")
                continue
            
            # Skip if mask is too sparse (less than 100 foreground pixels)
            if mask_sum < 100:
                print(f"Skipping batch with sparse mask: {mask_sum} pixels")
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
            
            # Accumulate gradients
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
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



def debug_video_dataloader(dataset, device="cpu"):
    """
    Debug function for checking the output of the VideoDataLoader.

    Args:
        dataset: The video dataset.
        device: The device (CPU/GPU).
    """
    video_loader = VideoDataLoader(dataset, device=device)
    
    print(f"[DEBUG] Total videos in dataset: {len(video_loader)}\n")

    for video_idx, (video_tensor, mask_tensor, video_id) in enumerate(video_loader):
        print(f"[DEBUG] Processing Video {video_idx+1}/{len(video_loader)} - Video ID: {video_id}")
        print(f"[DEBUG] Video Shape: {video_tensor.shape} (Expected: [B, T, C, H, W])")
        print(f"[DEBUG] Mask Shape: {mask_tensor.shape} (Expected: [B, T, 1, H, W])")

        T = video_tensor.shape[1]  # Number of frames

        for t in range(min(T, 10)):  # Limit prints to the first 10 frames
            print(f"[DEBUG] Frame {t}: Image Shape: {video_tensor[:, t].shape} (Expected: [B, C, H, W])")
            print(f"[DEBUG] Frame {t}: Mask Shape: {mask_tensor[:, t].shape} (Expected: [B, 1, H, W])")

        print(f"[DEBUG] Successfully verified video {video_id} ✅\n")

        # Only check the first few videos
        if video_idx >= 2:
            break

def visualize_predictions(model, val_loader, sam2, device, epoch, output_dir,num_samples=4,eval_mode=False):
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
    if eval_mode:
        random_indices = [i for i in range(num_samples)]
    else:
        random_indices = random.sample(range(dataset_size), num_samples)
    
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
    from sam2.build_sam import build_sam2, build_sam2_video_predictor

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
        sam2_video = build_sam2_video_predictor(config_path, video_checkpoint_path)

        
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
    # debug_video_dataloader(val_dataset, device="cuda" if torch.cuda.is_available() else "cpu")
    video_loader = VideoDataLoader(val_dataset, device)
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
        with torch.serialization.safe_globals([np._core.multiarray.scalar]):
            checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    
    # Set up optimizer with lower learning rate and higher weight decay
    initial_lr = args.lr
    weight_decay = args.weight_decay
    print(f"Using learning rate: {initial_lr} and weight decay: {weight_decay}")
    
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    
    # Set up criterion with increased weights for non-empty predictions
    pos_weight = 10.0  # Increased from 5.0
    criterion = CombinedBCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=pos_weight,
        empty_penalty=0.5  # Add penalty for empty predictions
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs")
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
        
        
        # Validate with both image and video models
        val_loss, val_dice = evaluate(
            model=model,
            val_loader=val_loader,
            sam2=sam2_image,
            criterion=criterion,
            device=device
        )
        
        # Additional video validation if requested
        if False:#args.eval_video:
            video_loss, video_dice = evaluate_video(
                model=model,
                val_loader=video_loader,  # ✅ Corrected to video_loader
                sam2_video=sam2_video.eval(),  
                criterion=criterion,  # ✅ Added missing argument
                device=device
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
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights")
    parser.add_argument("--eval_video", action="store_true", help="Evaluate on video data after training")
    
    # Distributed training parameters
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    
    args = parser.parse_args()
    main(args)