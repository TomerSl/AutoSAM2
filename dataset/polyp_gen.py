import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from sam2.training.utils.data_utils import VideoDatapoint, Frame, Object

class ImageMaskTransformWrapper:
    """Simple wrapper for transform that applies to both image and mask"""
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, image, mask):
        if self.transform is None:
            return image, mask
        
        result = self.transform({"image": image, "mask": mask})
        return result["image"], result["mask"]

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
        
        # Get original size before any transforms
        orig_size = (image.width, image.height)  # (width, height)
        
        # Apply custom transforms if specified
        if self.transform:
            image, mask = self.transform(image, mask)
        
        # Convert to numpy arrays for SAM transforms
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Apply SAM transforms if specified
        if self.sam_trans:
            image_np = self.sam_trans.apply_image(image_np)
            mask_np = self.sam_trans.apply_image(mask_np if len(mask_np.shape) == 3 else mask_np[..., None])
            
            # Convert to tensor and normalize
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()  # HWC -> CHW
            mask_tensor = torch.from_numpy(mask_np).permute(2, 0, 1).float() if len(mask_np.shape) == 3 else torch.from_numpy(mask_np).unsqueeze(0).float()
            
            # Binarize mask
            mask_tensor[mask_tensor > 0.5] = 1
            mask_tensor[mask_tensor <= 0.5] = 0
        else:
            # Convert PIL images to tensors
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()  # HWC -> CHW
            mask_tensor = torch.from_numpy(np.array(mask)).unsqueeze(0).float()
            
            # Binarize mask
            mask_tensor[mask_tensor > 0.5] = 1
            mask_tensor[mask_tensor <= 0.5] = 0
        
        # Create Frame object with the image tensor
        frame = Frame(data=image_tensor, objects=[])
        
        # Add the mask as an object to the frame
        # Convert to boolean tensor as expected by SAM2
        mask_tensor_bool = mask_tensor.bool()
        frame.objects.append(Object(object_id=0, frame_index=0, segment=mask_tensor_bool))
        
        # Convert sequence name to a numeric ID (SAM2 VideoDatapoint expects an int for video_id)
        try:
            # Try to convert directly if the sequence name is numeric
            video_id = int(seq_name)
        except ValueError:
            # Otherwise, hash the string for a numeric ID
            video_id = hash(seq_name) % 10000000  # Limit to a reasonable range
            
        # Create and return a VideoDatapoint as expected by SAM2
        return VideoDatapoint(frames=[frame], video_id=video_id, size=orig_size)


def get_transforms():
    """
    Define transformations for the dataset
    """
    # Simple transforms for now, you can extend this with more custom transformations
    transform_train = None  # Identity transformation
    transform_val = None    # Identity transformation
    
    return transform_train, transform_val


def get_polyp_gen_dataset(sam_trans=None, base_path="polyp_gen"):
    """
    Get training and validation datasets for the polyp_gen dataset
    
    Args:
        sam_trans: SAM model transforms to apply
        base_path: Base path to the polyp_gen dataset
    
    Returns:
        train_dataset, val_dataset: Dataset objects for training and validation
    """
    transform_train, transform_val = get_transforms()
    
    # Paths for the file lists
    train_list = os.path.join(base_path, "train", "train_list.txt")
    val_list = os.path.join(base_path, "valid", "val_list.txt")
    
    # Create datasets
    train_dataset = PolypGenDataset(
        root=base_path,
        file_list=train_list,
        transform=transform_train,
        sam_trans=sam_trans
    )
    
    val_dataset = PolypGenDataset(
        root=base_path,
        file_list=val_list,
        transform=transform_val,
        sam_trans=sam_trans
    )
    
    return train_dataset, val_dataset 