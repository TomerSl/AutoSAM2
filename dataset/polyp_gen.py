import os
import sys
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import random
from typing import Iterable, Dict, List, Tuple, Optional, Union

# Add sam2-main to the Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sam2-main'))

# Import from training module directly
from training.utils.data_utils import VideoDatapoint, Frame, Object
# Import transforms from SAM2
from training.dataset.transforms import (
    ComposeAPI, 
    NormalizeAPI,
    RandomHorizontalFlip,
    RandomGrayscale,
    ColorJitter,
    RandomAffine,
    RandomResizeAPI
)
import torchvision.transforms.functional as F


def get_transforms(resolution=1024):
    """
    Get transformations for training and validation datasets
    
    Args:
        resolution: The resolution to resize images to
    
    Returns:
        transform_train: Transformations for training dataset
        transform_val: Transformations for validation dataset
    """
    print("\n[DEBUG] All transformations disabled for debugging purposes\n")
    
    # Return None for both train and val transformations
    # This effectively disables all data augmentations
    transform_train = None
    transform_val = None
    
    return transform_train, transform_val


class ImageMaskTransformWrapper:
    """
    Wrapper that applies color transforms to images only and geometric transforms to both.
    This follows best practices for segmentation tasks.
    """
    def __init__(self, transforms: Optional[Dict[str, ComposeAPI]]):
        self.transforms = transforms
        
    def __call__(self, image, mask):
        if self.transforms is None:
            return image, mask
        
        # Step 1: Apply color transforms (image only)
        if 'color' in self.transforms and self.transforms['color'] is not None:
            # Convert image to tensor for SAM2 transforms
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()  # HWC -> CHW
            
            # Create a single frame VideoDatapoint with just the image (no mask)
            frame = Frame(data=image_tensor, objects=[])
            
            # Get image dimensions
            if hasattr(image, 'width') and hasattr(image, 'height'):
                w, h = int(image.width), int(image.height)
            else:
                h, w = int(image.shape[0]), int(image.shape[1])
                
            # Create VideoDatapoint
            video_dp = VideoDatapoint(frames=[frame], video_id=0, size=(w, h))
            
            # Apply color transforms to image only
            transformed_dp = self.transforms['color'](video_dp)
            
            # Convert back to PIL for further processing
            transformed_image = transformed_dp.frames[0].data.permute(1, 2, 0).numpy().astype(np.uint8)
            transformed_image = Image.fromarray(transformed_image)
        else:
            transformed_image = image
            
        # Step 2: Apply geometric transforms (both image and mask)
        if 'geometric' in self.transforms and self.transforms['geometric'] is not None:
            # Convert PIL images to tensors
            image_tensor = torch.from_numpy(np.array(transformed_image)).permute(2, 0, 1).float()  # HWC -> CHW
            
            # Process mask based on its type
            if isinstance(mask, np.ndarray):
                if len(mask.shape) == 2:
                    mask_np = mask
                else:
                    mask_np = mask[:, :, 0]
                mask_tensor = torch.from_numpy(mask_np).float()
            else:  # PIL Image
                mask_tensor = torch.from_numpy(np.array(mask)).float()
            
            # Binarize mask
            mask_tensor[mask_tensor > 0.5] = 1
            mask_tensor[mask_tensor <= 0.5] = 0
            
            # Convert to boolean tensor
            mask_tensor_bool = mask_tensor.bool()
            
            # Create Frame object
            frame = Frame(data=image_tensor, objects=[])
            frame.objects.append(Object(object_id=0, frame_index=0, segment=mask_tensor_bool))
            
            # Get image dimensions again (in case they changed from color transforms)
            if hasattr(transformed_image, 'width') and hasattr(transformed_image, 'height'):
                w, h = int(transformed_image.width), int(transformed_image.height)
            else:
                h, w = int(transformed_image.shape[0]), int(transformed_image.shape[1])
                
            # Create VideoDatapoint
            video_dp = VideoDatapoint(frames=[frame], video_id=0, size=(w, h))
            
            # Store original mask values to use after transform
            original_mask_values = video_dp.frames[0].objects[0].segment.clone()
            
            # Apply geometric transforms to both image and mask
            transformed_dp = self.transforms['geometric'](video_dp)
            
            # Check if mask has been zeroed out
            transformed_mask = transformed_dp.frames[0].objects[0].segment
            
            if transformed_mask.sum() == 0 and original_mask_values.sum() > 0:
                # If mask has been zeroed but was not originally zero, restore values
                transformed_dp.frames[0].objects[0].segment = original_mask_values
            
            # Convert back to PIL for further processing
            transformed_image = transformed_dp.frames[0].data.permute(1, 2, 0).numpy().astype(np.uint8)
            transformed_image = Image.fromarray(transformed_image)
            
            # Get the mask and ensure it has proper dimensions
            transformed_mask = transformed_dp.frames[0].objects[0].segment.float()
            
            # Ensure mask has spatial dimensions (not just a 1D vector)
            if transformed_mask.dim() == 1:
                # If somehow mask is 1D, reshape it back to 2D based on transformed image size
                h, w = transformed_image.shape[:2] if isinstance(transformed_image, np.ndarray) else (transformed_image.height, transformed_image.width)
                transformed_mask = transformed_mask.reshape(h, w)
            
            # Add channel dimension for F.interpolate
            transformed_mask = transformed_mask.unsqueeze(0)
        else:
            # If no geometric transforms, convert mask to tensor format for consistency
            if isinstance(mask, np.ndarray):
                if len(mask.shape) == 2:
                    mask_np = mask
                else:
                    mask_np = mask[:, :, 0]
                mask_tensor = torch.from_numpy(mask_np).float()
            else:  # PIL Image
                mask_tensor = torch.from_numpy(np.array(mask)).float()
            
            # Binarize mask
            mask_tensor[mask_tensor > 0.5] = 1
            mask_tensor[mask_tensor <= 0.5] = 0
            
            # Add channel dimension
            transformed_mask = mask_tensor.unsqueeze(0)
        
        return transformed_image, transformed_mask


class PolypGenDataset(Dataset):
    def __init__(self, root, file_list, transform=None, sam_trans=None):
        """
        Args:
            root (str): Directory containing 'images' and 'masks' subfolders.
            file_list (str): Path to a text file listing sequence names
            transform: Custom transformations to apply to images and masks
            sam_trans: SAM model transforms to apply
        """
        self.root = os.path.expanduser(root)
        self.transform = ImageMaskTransformWrapper(transform) if transform is not None else None
        self.sam_trans = sam_trans
        
        # Read sequence names from file_list
        with open(file_list, 'r') as f:
            self.sequences = [line.strip() for line in f.readlines()]
        
        # Build sequence to frames mapping
        self.sequence_frames = {}
        for seq in self.sequences:
            seq_img_dir = os.path.join(self.root, "images", seq)
            # Check if directory exists
            if not os.path.exists(seq_img_dir):
                print(f"Warning: Directory {seq_img_dir} does not exist")
                continue
                
            frames = [f.split('.')[0] for f in sorted(os.listdir(seq_img_dir)) 
                    if f.endswith('.jpg') or f.endswith('.png')]
            self.sequence_frames[seq] = frames
        
        # Create flat list of all frames for indexing
        self.entries = []
        for seq in self.sequences:
            if seq in self.sequence_frames:  # Only add if sequence exists
                for frame in self.sequence_frames[seq]:
                    self.entries.append((seq, frame))
                    
        print(f"Loaded {len(self.entries)} frames from {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        seq_name, frame_name = self.entries[index]
        
        # Construct paths using sequence directory structure
        img_path = os.path.join(self.root, "images", seq_name, f"{frame_name}.jpg")
        mask_path = os.path.join(self.root, "masks", seq_name, f"{frame_name}.jpg")
        
        # Try alternative extension if the file doesn't exist
        if not os.path.exists(img_path):
            img_path = os.path.join(self.root, "images", seq_name, f"{frame_name}.png")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.root, "masks", seq_name, f"{frame_name}.png")
        
        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Get original size as simple integers
        w, h = int(image.width), int(image.height)
        
        # Apply custom transforms if specified
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Convert to numpy arrays for SAM transforms
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Apply SAM transforms if specified
        if self.sam_trans:
            # Use the SAM2Transforms directly by calling it on the image
            image_tensor = self.sam_trans(image_np)
            
            # For the mask, we need to handle it differently - keep as 2D
            if len(mask_np.shape) > 2:
                # If mask is multi-channel, take first channel
                mask_np = mask_np[:, :, 0]
            
            # Convert mask to tensor but keep as 2D (H, W)
            mask_tensor = torch.from_numpy(mask_np).float()
            
            # Binarize mask
            mask_tensor[mask_tensor > 0.5] = 1
            mask_tensor[mask_tensor <= 0.5] = 0
        else:
            # Convert PIL images to tensors
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()  # HWC -> CHW
            # Keep mask as 2D tensor (H, W) without channel dimension for SAM2 transforms
            mask_np = np.array(mask)
            mask_tensor = torch.from_numpy(mask_np).float()
            
            # Binarize mask
            mask_tensor[mask_tensor > 0.5] = 1
            mask_tensor[mask_tensor <= 0.5] = 0
        
        # Create Frame object with the image tensor
        frame = Frame(data=image_tensor, objects=[])
        
        # Add the mask as an object to the frame
        # Convert to boolean tensor but keep as 2D (H, W) for SAM2 compatibility
        mask_tensor_bool = mask_tensor.bool()
        frame.objects.append(Object(object_id=0, frame_index=0, segment=mask_tensor_bool))
        
        # Convert sequence name to a numeric ID
        try:
            video_id = int(seq_name)
        except ValueError:
            video_id = hash(seq_name) % 10000000
            
        # Create VideoDatapoint with simple size tuple
        return VideoDatapoint(frames=[frame], video_id=video_id, size=(w, h))


def get_polyp_gen_dataset(sam_trans=None, base_path="polyp_gen"):
    """
    Get training and validation datasets for the polyp_gen dataset
    
    Args:
        sam_trans: SAM model transforms to apply
        base_path: Base path to the polyp_gen dataset
    
    Returns:
        train_dataset, val_dataset: Dataset objects for training and validation
    """
    # Directly use None for transforms to disable all transformations
    print("\n[DEBUG] Using None for all transformations - disabled for debugging\n")
    # Define color transforms for the image only
    # Define color transforms for the image only.
    color_transforms = ComposeAPI([
        ColorJitter(
            consistent_transform=True,
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )
    ])
    
    # Define geometric transforms that include only a rotation.
    # We set translate, scale, and shear to None so only rotation is applied.
    geometric_transforms = ComposeAPI([
        RandomHorizontalFlip(consistent_transform=True, p=0.5)
        # RandomAffine(
        #     degrees=15,                # Rotate by up to ±15 degrees.
        #     consistent_transform=True, # Apply the same transform to all frames (and masks).
        #     translate=None,
        #     scale=None,
        #     shear=None,
        #     image_interpolation="bilinear"  # Changed from "bicubic" to "bilinear"
        # )
    ])

    # Create a dictionary for training transforms
    transform_train = {
        'color': color_transforms,
        'geometric': None  # Disable geometric transforms for now
    }
    # transform_train = None
    transform_val = None
    
    # Paths for the file lists
    train_list = os.path.join(base_path, "train", "train_list.txt")
    val_list = os.path.join(base_path, "valid", "val_list.txt")
    
    # Create datasets
    train_dataset = PolypGenDataset(
        root=os.path.join(base_path, "train"),  # Update to use the train subdirectory
        file_list=train_list,
        transform=transform_train,
        sam_trans=None  # Force sam_trans to None to prevent mask processing issues
    )
    
    val_dataset = PolypGenDataset(
        root=os.path.join(base_path, "valid"),  # Update to use the valid subdirectory
        file_list=val_list,
        transform=transform_val,
        sam_trans=None  # Force sam_trans to None to prevent mask processing issues
    )
    
    return train_dataset, val_dataset 

class VideoDataLoader:
    """
    Custom data loader for video evaluation that loads one video at a time
    and ensures all frames within a video have consistent dimensions.
    
    This loader groups frames by video_id, ensuring that all frames from the same
    video are processed together with consistent dimensions.
    """
    def __init__(self, dataset, device=None):
        self.dataset = dataset
        self.device = device
        self.video_indices = []
        
        # Group data points by video ID
        self.videos = {}
        
        # Process each item in the dataset
        for idx in range(len(dataset)):
            # Get the VideoDatapoint
            dp = dataset[idx]
            video_id = dp.video_id
            
            # Initialize video entry if not exists
            if video_id not in self.videos:
                self.videos[video_id] = []
                self.video_indices.append(video_id)
            
            # Add this datapoint to the video group
            self.videos[video_id].append(dp)
        
        print(f"Grouped {len(dataset)} frames into {len(self.video_indices)} videos")
    
    def __len__(self):
        return len(self.video_indices)
    
    def __iter__(self):
        for video_id in self.video_indices:
            datapoints = self.videos[video_id]
            if not datapoints:
                continue
                
            # Collect all frames from all datapoints for this video
            all_frames = []
            all_masks = []

            for dp in datapoints:
                for frame in dp.frames:
                    all_frames.append(frame.data)  # Extract image tensor
                    mask_tensor = frame.objects[0].segment.float() if frame.objects else torch.zeros_like(frame.data[0:1])  # Default to empty mask
                    all_masks.append(mask_tensor)  # Extract mask tensor

            if not all_frames:
                continue

            # Convert to tensors
            video_tensor = torch.stack(all_frames, dim=0)  # (T, C, H, W)
            mask_tensor = torch.stack(all_masks, dim=0)  # (T, 1, H, W)

            # Add batch dimension (B=1 for evaluation)
            video_tensor = video_tensor.unsqueeze(0)  # (1, T, C, H, W)
            mask_tensor = mask_tensor.unsqueeze(0)  # (1, T, 1, H, W)

            print(f"[DEBUG] Yielding Video {video_id}: Shape {video_tensor.shape}, Mask Shape {mask_tensor.shape}")

            yield video_tensor, mask_tensor, video_id

