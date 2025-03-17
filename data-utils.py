"""
Utility functions for data loading and preprocessing
"""

import os
import torch
from PIL import Image
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader

def get_transforms():
    """
    Get the image and sketch transforms
    """
    transform_image = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_sketch = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return transform_image, transform_sketch

def load_images(image_folder_path, transform):
    """
    Load images from a folder
    
    Args:
        image_folder_path: Path to the folder containing images
        transform: Transformation to apply to the images
        
    Returns:
        image_file_names: List of image file names
        images: Tensor of images
    """
    image_file_names = sorted(os.listdir(image_folder_path))
    images_list = []
    
    for image_name in image_file_names:
        if image_name.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
            image_path = os.path.join(image_folder_path, image_name)
            image = Image.open(image_path)
            image = transform(image)
            images_list.append(image.unsqueeze(0))
    
    if not images_list:
        raise ValueError(f"No valid images found in {image_folder_path}")
        
    images = torch.cat(images_list, dim=0)
    return image_file_names, images

def denorm(img_tensors):
    """
    Denormalize image tensors from [-1, 1] to [0, 1]
    """
    return img_tensors * 0.5 + 0.5

def load_data(config):
    """
    Load data from the specified paths in config
    
    Args:
        config: Configuration object
        
    Returns:
        train_data: Training images
        train_sketch: Training sketches
        test_data: Test images
        test_sketch: Test sketches
        train_label: Training labels
        test_label: Test labels
        train_loader: Training data loader
    """
    transform_image, transform_sketch = get_transforms()
    
    # Load images
    train_data_name, train_data = load_images(config.TRAIN_DATA_PATH, transform_image)
    train_sketch_name, train_sketch = load_images(config.TRAIN_SKETCH_PATH, transform_sketch)
    test_data_name, test_data = load_images(config.TEST_DATA_PATH, transform_image)
    test_sketch_name, test_sketch = load_images(config.TEST_SKETCH_PATH, transform_sketch)
    
    # Load labels
    train_labels_df = pd.read_csv(config.TRAIN_LABELS_PATH)
    test_labels_df = pd.read_csv(config.TEST_LABELS_PATH)
    
    train_label = train_labels_df.iloc[:, 1:]
    train_label = torch.tensor(train_label.values, dtype=torch.float32)
    
    test_label = test_labels_df.iloc[:, 1:]
    test_label = torch.tensor(test_label.values, dtype=torch.float32)
    
    return {
        'train_data': train_data,
        'train_data_name': train_data_name,
        'train_sketch': train_sketch,
        'test_data': test_data,
        'test_sketch': test_sketch,
        'train_label': train_label,
        'test_label': test_label,
        'train_labels_df': train_labels_df,
        'test_labels_df': test_labels_df
    }

def prepare_data_for_training(train_data, train_label, embedding_model, config):
    """
    Prepare data for training
    
    Args:
        train_data: Training images
        train_label: Training labels
        embedding_model: Label embedding model
        config: Configuration object
        
    Returns:
        train_data: Training data with embedded labels
        train_loader: Training data loader
    """
    # Embed labels
    embedded_labels = embedding_model(train_label)
    
    # Combine data with embedded labels
    train_data = torch.cat((train_data, embedded_labels), dim=1)
    
    # Create data loader
    train_loader = DataLoader(
        train_data, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=3, 
        pin_memory=True
    )
    
    return train_data, train_loader
