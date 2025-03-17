import os
import torch
import numpy as np
import cv2
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.transforms import SAM2Transforms
from datasets import get_dataset
import torch.nn.functional as F
from models.model import AutoSAM2
import traceback

def calculate_metrics(pred_mask, gt_mask):
    """
    Calculate evaluation metrics for segmentation
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert tensors to numpy if needed
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.cpu().numpy()
    
    # Ensure binary masks
    pred_mask = (pred_mask > 0.5).astype(np.float32)
    gt_mask = (gt_mask > 0.5).astype(np.float32)
    
    # Calculate metrics
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
    
    # Dice Coefficient
    dice = (2.0 * intersection + 1e-6) / (union + intersection + 1e-6)
    
    # IoU / Jaccard Index
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return {
        'dice': dice,
        'iou': iou
    }

def save_visualization(image, gt_mask, pred_mask, output_path):
    """
    Save visualization of the prediction results
    
    Args:
        image: Input image
        gt_mask: Ground truth mask
        pred_mask: Predicted mask
        output_path: Path to save visualization
    """
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Ground truth
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.imshow(gt_mask, alpha=0.5, cmap='jet')
    plt.title('Ground Truth')
    plt.axis('off')
    
    # Prediction
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(pred_mask, alpha=0.5, cmap='jet')
    plt.title('Prediction')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def prepare_batch_for_inference(batch, device):
    """
    Prepares a batch from our custom dataset format for inference with video data
    
    Args:
        batch: Batch from dataloader
        device: Device to load tensors to
        
    Returns:
        dict: Dictionary with images and masks tensors in video format [B, T, C, H, W]
    """
    if isinstance(batch, dict) and 'image' in batch:
        # Direct format with 'image', 'mask' keys
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Check if this is already in video format [B, T, C, H, W]
        is_video = len(images.shape) == 5
        
        if not is_video:
            print("Converting image batch to video format with single frame")
            # Convert to video format with a single frame
            images = images.unsqueeze(1)  # [B, 1, C, H, W]
            masks = masks.unsqueeze(1)    # [B, 1, 1, H, W]
        
        # Generate video IDs
        if 'video_id' in batch:
            video_ids = batch['video_id']
        else:
            # If no video IDs provided, generate them
            B = images.shape[0]
            video_ids = [f"video_{i}" for i in range(B)]
        
        return {
            'images': images,
            'masks': masks,
            'video_ids': video_ids
        }
    
    elif isinstance(batch, dict) and 'video_datapoint' in batch:
        # VideoDatapoint format
        video_datapoints = batch['video_datapoint']
        
        videos_list = []
        masks_list = []
        video_ids = []
        
        # Process each video datapoint
        for vdp in video_datapoints:
            # Get frames for this video
            frames = [frame.data for frame in vdp.frames]
            
            # Get masks for this video if available
            masks = []
            for frame in vdp.frames:
                if frame.objects:
                    masks.append(frame.objects[0].segment)
                else:
                    # Create an empty mask if no objects
                    masks.append(torch.zeros_like(frame.data[0:1]))
            
            # Stack frames and masks for this video
            video_tensor = torch.stack(frames)  # [T, C, H, W]
            masks_tensor = torch.stack(masks)   # [T, 1, H, W]
            
            videos_list.append(video_tensor)
            masks_list.append(masks_tensor)
            video_ids.append(vdp.video_id)
        
        # Stack all videos into batch dimension
        # Each video may have different number of frames, so we'll pad if needed
        max_frames = max(v.shape[0] for v in videos_list)
        B = len(videos_list)
        
        # Get dimensions from first video
        _, C, H, W = videos_list[0].shape
        
        # Create tensors of appropriate size
        padded_videos = torch.zeros(B, max_frames, C, H, W, device=device)
        padded_masks = torch.zeros(B, max_frames, 1, H, W, device=device)
        
        # Fill tensors with actual data
        for i, (video, masks) in enumerate(zip(videos_list, masks_list)):
            T = video.shape[0]
            padded_videos[i, :T] = video
            padded_masks[i, :T] = masks
            
        print(f"Loaded {B} videos with max {max_frames} frames each")
        
        return {
            'images': padded_videos,
            'masks': padded_masks,
            'video_ids': video_ids
        }
    
    else:
        # Handle other formats if needed
        raise ValueError("Unsupported batch format. Expected 'image' or 'video_datapoint' in batch dictionary.")

def sam_call(inputs, sam2, dense_embeddings, output_dict, t, state=None, input=None):
    """
    Process a single video with SAM2 model to generate masks using the memory-based temporal consistency.
    
    This function processes a specific frame (t) within a video sequence using SAM2's memory
    conditioning mechanism for temporal consistency. It extracts features, applies prompt
    encodings, and generates masks while managing temporal information flow between frames.
    
    Args:
        inputs: Video data tensor with shape [B, T, C, H, W]
               B: batch size, T: number of frames, C: channels, H: height, W: width
        sam2: SAM2VideoPredictor instance that provides the model for mask generation
        dense_embeddings: Dense embeddings from AutoSAM2 model to be used with SAM2
        output_dict: Dictionary that tracks frame outputs for memory conditioning
                     Has structure {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        t: Current frame index being processed (0-indexed)
        state: Type of prompt to use for mask generation. Options:
               - 'autosam2': Use AutoSAM2 embeddings without additional prompts
               - 'sam2_point': Use point prompts
               - 'sam2_gt': Use ground truth mask prompts
               - 'sam2_bb': Use bounding box prompts
               - None: Defaults to 'autosam2'
        input: Additional inputs based on state (points, masks, or boxes)
               For 'sam2_point': [coordinates, labels]
               For 'sam2_gt': mask tensor
               For 'sam2_bb': bounding box coordinates
    
    Returns:
        torch.Tensor: Low-resolution mask predictions for the current frame
                     with shape [B, 1, H_mask, W_mask]
    """
    # Get dimensions from input video
    B, T, C, H, W = inputs.shape
    
    # Initialize dictionary to store current frame outputs
    current_out = {}
    
    with torch.no_grad():
        # Step 1: Extract image features for the current frame
        # Extract the frame t from each video in the batch
        current_frames = inputs[:, t]  # Shape: [B, C, H, W]
        
        # Generate image embeddings using SAM2's image encoder
        image_embeddings = sam2.image_encoder(current_frames)
  
        # Apply convolutions to process feature maps at different scales
        image_embeddings["backbone_fpn"][0] = sam2.sam_mask_decoder.conv_s0(
            image_embeddings["backbone_fpn"][0]
        )
        image_embeddings["backbone_fpn"][1] = sam2.sam_mask_decoder.conv_s1(
            image_embeddings["backbone_fpn"][1]
        )

        # Step 2: Prepare backbone features for the decoder
        # Extract and format features for the mask decoder
        image_embeddings, vision_feats, vision_pos_embeds, feat_sizes = sam2._prepare_backbone_features(image_embeddings)

        # Step 3: Extract high-resolution features if available
        # These are used for more detailed mask generation
        if len(vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        # Step 4: Set up default point coordinates for no-prompt case
        # Default to negative prompts (no point specified)
        sam_point_coords = torch.zeros(B, 1, 2, device=sam2.device)
        sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=sam2.device)

        # Step 5: Process different prompt types based on state
        if state == 'autosam2' or state is None:
            # Use AutoSAM2 embeddings with no additional prompts
            # Generate sparse embeddings with default negative points
            sparse_embeddings, dense_embeddings_none = sam2.sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels), 
                boxes=None, 
                masks=None
            )
            # Note: dense_embeddings are provided from AutoSAM2 model
            
        elif state == 'sam2_point' and input is not None:
            # Use point prompts provided in input
            # Extract coordinates and labels
            coords, labels = input[0], input[1]
            
            # Convert to tensors on the correct device if they aren't already
            sam_point_coords = torch.tensor(coords, dtype=torch.float32, device=sam2.device).unsqueeze(0)  
            sam_point_labels = torch.tensor(labels, dtype=torch.int32, device=sam2.device).unsqueeze(0)
            
            # Generate embeddings from point prompts
            sparse_embeddings, dense_embeddings = sam2.sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels), 
                boxes=None, 
                masks=None
            )
            
        elif state == 'sam2_gt' and input is not None:
            # Use ground truth mask as prompt
            # Generate embeddings from mask
            sparse_embeddings, dense_embeddings = sam2.sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels), 
                boxes=None, 
                masks=input.to(sam2.device)
            )
            
        elif state == 'sam2_bb' and input is not None:
            # Use bounding box prompts
            # Convert to tensor on the correct device
            boxes = torch.tensor(input, dtype=torch.float32, device=sam2.device).unsqueeze(0)  
            sparse_embeddings_none, dense_embeddings = sam2.sam_prompt_encoder(points=(sam_point_coords,sam_point_labels), boxes=boxes, masks=None)
            
        # Step 6: Apply memory conditioning for temporal consistency
        pix_feat = sam2._prepare_memory_conditioned_features(
            frame_idx=t,
            is_init_cond_frame=t==0,
            current_vision_feats=vision_feats[-1:],
            current_vision_pos_embeds=vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=output_dict,
            num_frames=T,
        )    

        # Step 7: Generate masks using the SAM2 mask decoder
        low_res_multimasks, ious, sam_output_tokens, object_score_logits = sam2.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=sam2.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings_none,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        repeat_image=False,  
        high_res_features=high_res_features,
        )   

        # Step 8: Apply object score thresholding for better mask quality
        # Handle cases where no object is detected
        NO_OBJ_SCORE = -1e4  # Very negative value for "no object"
        is_obj_appearing = object_score_logits > 0  # Binary indicator of object presence

        # Apply the object score to improve mask quality
        # Replace mask values with NO_OBJ_SCORE where object isn't detected
        low_res_masks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_masks,
            NO_OBJ_SCORE,
        )
        low_res_masks = low_res_masks.float()
        
        # Step 9: Generate high-resolution masks by upscaling
        high_res_masks = F.interpolate(
            low_res_masks,
            size=(sam2.image_size, sam2.image_size),
            mode="bilinear",
            align_corners=False,
        )
        
        # Step 10: Process SAM output tokens for memory tracking
        sam_output_token = sam_output_tokens[:, 0]  # Take first token
        obj_ptr = sam2.obj_ptr_proj(sam_output_token)  # Project to object pointer space

        # Step 11: Apply object appearance score handling
        if sam2.pred_obj_scores:
            # Calculate object appearance probability
            if sam2.soft_no_obj_ptr:
                lambda_is_obj_appearing = object_score_logits.sigmoid()  # Soft probability
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()  # Hard binary

            # Apply fixed no-object pointer if needed
            if sam2.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
                
            # Combine with no-object pointer based on appearance probability
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * sam2.no_obj_ptr

        # Step 12: Store current frame outputs for memory tracking
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Step 13: Encode current frame as memory for subsequent frames
        maskmem_features, maskmem_pos_enc = sam2._encode_new_memory(
            vision_feats,
            feat_sizes,
            high_res_masks,
            object_score_logits,
            False,  # Don't clear previous memory
        )
        
        # Step 14: Update memory structure based on frame type
        # First frame is treated as conditioning frame, others as non-conditioning
        if t == 0:
            # Store first frame as conditioning frame
            output_dict["cond_frame_outputs"][t] = {
                "obj_ptr": obj_ptr,
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": maskmem_pos_enc
            }
        else:
            # Store subsequent frames as non-conditioning frames
            output_dict["non_cond_frame_outputs"][t] = {
                "obj_ptr": obj_ptr,
                "maskmem_features": maskmem_features,
                "maskmem_pos_enc": maskmem_pos_enc
            }

    # Return the generated masks for the current frame
    return low_res_masks

