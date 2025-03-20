import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

class PolypDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'images')
        self.mask_dir = os.path.join(root_dir, 'masks')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name.replace('.jpg', '.png'))
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)
        
        return image, mask, img_name

def analyze_mask_examples(num_samples=5):
    # Create output directory
    output_dir = 'mask_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = PolypDataset(root_dir='data/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Collect statistics
    foreground_percentages = []
    
    for i, (image, mask, img_name) in enumerate(dataloader):
        if i >= num_samples:
            break
            
        # Convert to numpy for visualization
        image_np = image[0].permute(1, 2, 0).numpy()
        image_np = (image_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        mask_np = mask[0, 0].numpy()
        
        # Calculate statistics
        mask_sum = mask_np.sum()
        mask_size = mask_np.size
        foreground_percentage = (mask_sum / mask_size) * 100
        foreground_percentages.append(foreground_percentage)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask_np, cmap='gray')
        axes[1].set_title(f'Mask (Foreground: {foreground_percentage:.2f}%)')
        axes[1].axis('off')
        
        # Overlay visualization
        overlay = image_np.copy()
        overlay[mask_np > 0.5] = [1, 0, 0]  # Red overlay for mask
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{i+1}.png'))
        plt.close()
        
        print(f"\nAnalysis for sample {i+1} ({img_name[0]}):")
        print(f"Mask sum: {mask_sum:.2f}")
        print(f"Mask size: {mask_size}")
        print(f"Foreground percentage: {foreground_percentage:.2f}%")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Mean foreground percentage: {np.mean(foreground_percentages):.2f}%")
    print(f"Min foreground percentage: {np.min(foreground_percentages):.2f}%")
    print(f"Max foreground percentage: {np.max(foreground_percentages):.2f}%")
    
    # Plot histogram of foreground percentages
    plt.figure(figsize=(10, 5))
    plt.hist(foreground_percentages, bins=20)
    plt.title('Distribution of Foreground Percentages')
    plt.xlabel('Foreground Percentage')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'foreground_distribution.png'))
    plt.close()

if __name__ == '__main__':
    analyze_mask_examples()