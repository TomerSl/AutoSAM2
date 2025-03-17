# This file marks the directory as a Python package
# It can be empty

from dataset.polyp_gen import get_polyp_gen_dataset

def get_dataset(dataset_name, transform=None, base_path=None):
    """
    Factory function to get the appropriate dataset based on name
    
    Args:
        dataset_name: Name of the dataset
        transform: SAM2 transforms to apply to the dataset
        base_path: Base path to the dataset directory
    
    Returns:
        train_dataset, val_dataset
    """
    if dataset_name.lower() == 'polyp_gen':
        base_path = base_path or 'polyp_gen'
        return get_polyp_gen_dataset(sam_trans=transform, base_path=base_path)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")