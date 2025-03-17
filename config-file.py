"""
Configuration file for Conditional Sketch-to-Image GAN
"""

import torch
import os

# Paths
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TRAIN_DATA_PATH = os.path.join(DATA_ROOT, 'Train_data')
TRAIN_SKETCH_PATH = os.path.join(DATA_ROOT, 'Train/Train_sketch/Contours')
TEST_DATA_PATH = os.path.join(DATA_ROOT, 'Test/Test_data/Test')
TEST_SKETCH_PATH = os.path.join(DATA_ROOT, 'Test/Test_sketch/Test_contours')
TRAIN_LABELS_PATH = os.path.join(DATA_ROOT, 'Train_labels.csv')
TEST_LABELS_PATH = os.path.join(DATA_ROOT, 'Test_Labels.csv')

# Output directories
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
GENERATED_IMAGES_DIR = os.path.join(OUTPUT_DIR, 'generated_images')
ORIGINAL_SAMPLES_DIR = os.path.join(OUTPUT_DIR, 'Original_Samples')

# Model parameters
IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
NUM_CLASSES = 7

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wandb configuration
WANDB_PROJECT = "DL_Assignment_4_2"
WANDB_ENTITY = "your_wandb_username"  # Replace with your username

# Model paths
CLASSIFIER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'weights', 'classifier.pth')
GENERATOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'weights', 'generator.pth')
DISCRIMINATOR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'weights', 'discriminator.pth')

# Ensure directories exist
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'weights'), exist_ok=True)
os.makedirs(GENERATED_IMAGES_DIR, exist_ok=True)
os.makedirs(ORIGINAL_SAMPLES_DIR, exist_ok=True)
