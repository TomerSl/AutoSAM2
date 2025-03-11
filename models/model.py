import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_single import ModelEmb, Model
from models.hardnet import HarDNet
from models.base import *

def get_autosam2_model(image_size, embed_dim=256, model_type="embed"):
    """
    Creates and returns the AutoSAM2 model
    
    Args:
        image_size (int): Input image size for SAM2
        embed_dim (int): Dimension of embeddings
        model_type (str): Type of model to use ("embed" or "full")
        
    Returns:
        nn.Module: AutoSAM2 model
    """
    # Simple dictionary for args to match model_single.py expectation
    args = {
        'task': 'polyp',
        'epoch': 200,
        'lr': 0.0001,
        'step': 100,
        'decay': 0.5,
        'Idim': 352  # Default image dimension for AutoSAM models
    }
    
    if model_type == "embed":
        # Use ModelEmb which outputs embeddings for SAM2
        model = ModelEmb(args)
    else:
        # Use full Model with decoder
        model = Model(args)
        
    # Initialize model weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    return model

class AutoSAM2(nn.Module):
    """
    AutoSAM2 model that combines the embedding model with additional layers
    to match the expected dense embedding format for SAM2.
    Supports both image and video inputs.
    """
    def __init__(self, image_size=1024, embed_dim=256):
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        
        # Create the base embedding model
        self.embed_model = get_autosam2_model(image_size, embed_dim, "embed")
        
        # Additional layers to match SAM2 expected format if needed
        self.adapter = nn.Conv2d(128, embed_dim, kernel_size=1)
        
    def forward(self, x):
        original_shape = x.shape
        is_video = len(original_shape) == 5
        
        if is_video:
            # Handle video data [B, T, C, H, W]
            b, t, c, h, w = original_shape
            x = x.reshape(b*t, c, h, w)
        else:
            # Handle image data [B, C, H, W]
            b, c, h, w = original_shape
        
        # If input image size doesn't match model's expected size, resize
        if h != 352 or w != 352:
            x = F.interpolate(x, size=(352, 352), mode='bilinear', align_corners=False)
            
        # Get embeddings from base model
        embeddings = self.embed_model(x)
        
        # Process embeddings if needed
        if embeddings.shape[1] != self.embed_dim:
            embeddings = self.adapter(embeddings)
            
        # If we need to resize back to original image size
        if h != 352 or w != 352:
            embeddings = F.interpolate(embeddings, size=(h, w), mode='bilinear', align_corners=False)
            
        # Reshape back to video format if input was video
        if is_video:
            # Current shape: [B*T, embed_dim, H, W]
            # Target shape: [B, T, embed_dim, H, W]
            embeddings = embeddings.reshape(b, t, embeddings.shape[1], embeddings.shape[2], embeddings.shape[3])
            
        return embeddings 