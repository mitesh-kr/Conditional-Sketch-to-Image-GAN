# -*- coding: utf-8 -*-
"""evaluation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1WPhdF_CzQ8DhiAlADh5v-ET27qxHvg5j
"""

"""
Evaluation script for conditional GAN
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from PIL import Image
from torchvision import transforms

from models import Generator, Classifier, LabelEmbedding
from utils import load_data, denorm
from config import *

def preprocess_images(images):
    """
    Preprocess images for inception model
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return preprocess(images).unsqueeze(0)

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import inception_v3
import numpy as np
from torchvision import utils as vutils
import os

# Function to calculate the inception score
def inception_score(images, inception_model, splits=10):
    preds = torch.nn.functional.softmax(inception_model(images), dim=1)
    scores = []
    for i in range(splits):
        part = preds[i * (images.shape[0] // splits): (i + 1) * (images.shape[0] // splits), :]
        kl = part * (torch.log(part) - torch.log(torch.mean(part, dim=0)))
        kl = torch.mean(torch.sum(kl, dim=1))
        scores.append(torch.exp(kl))
    return torch.mean(torch.tensor(scores)), torch.std(torch.tensor(scores))

# Function to calculate the FID
def calculate_fid(real_features, generated_features):
    mu_real, mu_gen = torch.mean(real_features, dim=0), torch.mean(generated_features, dim=0)
    sigma_real = torch_cov(real_features)
    sigma_gen = torch_cov(generated_features)
    sqrt_diff = torch.sqrt(sigma_real.mm(sigma_gen))
    if torch.any(torch.isnan(sqrt_diff)):
        sqrt_diff[torch.isnan(sqrt_diff)] = 0
    fid = torch.norm(mu_real - mu_gen)**2 + torch.trace(sigma_real + sigma_gen - 2 * sqrt_diff)
    return fid

def torch_cov(x):
    mean_x = torch.mean(x, dim=0)
    x = x - mean_x.unsqueeze(0)
    if x.size(0) <= 1:
        return torch.zeros_like(torch.mm(x.t(), x))
    return 1 / (x.size(0) - 1) * x.t().mm(x)

# Function to preprocess images
def preprocess_images(images):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])
    return preprocess(images).unsqueeze(0)

def denorm(img_tensors, stats=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))):
    return img_tensors * stats[1][0] + stats[0][0]

def save_original_samples(test_data, test_label, directory='Original Samples'):
    """Save original samples for comparison"""
    num_class = 7
    grid = []
    for c in range(num_class):
        images = test_data[torch.max(test_label.data, 1)[1] == c][:10]
        grid_image = vutils.make_grid(images, nrow=10, padding=2)
        grid.append(grid_image)
    grid = torch.cat(grid, dim=1)
    os.makedirs(directory, exist_ok=True)
    vutils.save_image(denorm(grid), f"{directory}/Original_sample.png")
    return f"{directory}/Original_sample.png"

def evaluate_generator(generator, test_sketch, classifier, device, num_samples=10):
    """Evaluate generator performance and save generated images"""
    num_class = 7
    grid = []
    classi_loss = 0
    classi_accuarcy = 0

    # Create Embedding object if needed (you might need to import this from your model definition)
    from your_model_file import LabelEmbedding
    Embedding = LabelEmbedding()

    for c in range(num_class):
        samples = test_sketch[torch.randperm(test_sketch.shape[0])[:num_samples]]
        label = torch.eye(num_class)[c]
        labels = label.unsqueeze(0).repeat(samples.shape[0], 1)
        labels_classification = torch.max(labels, 1)[1]
        labels_embedded = Embedding(labels)
        samples = torch.cat([samples, labels_embedded], dim=1)
        images = generator(samples.to(device))

        y_pred_cla = classifier(images)
        classification_loss = torch.nn.functional.cross_entropy(y_pred_cla, labels_classification.to(device))
        classi_loss += classification_loss.item()

        classification_accuarcy = (((torch.max(y_pred_cla, 1)[1] == labels_classification.to(device))).sum().item()) / num_samples
        classi_accuarcy += classification_accuarcy

        grid_image = vutils.make_grid(images, nrow=num_samples, padding=2)
        grid.append(grid_image)

    grid = torch.cat(grid, dim=1)
    directory = 'generated_images'
    os.makedirs(directory, exist_ok=True)
    output_path = f"{directory}/generated_samples.png"
    vutils.save_image(denorm(grid), output_path)

    print(f'Classification Loss: {classi_loss / num_class:.4f} | '
          f'Classification Accuracy: {((classi_accuarcy / num_class) * 100):.4f}%')

    return output_path, classi_loss / num_class, (classi_accuarcy / num_class) * 100

def compute_metrics(gen_image_path, orig_image_path):
    """Compute Inception Score and FID for generated images"""
    # Load the images
    gen_image = Image.open(gen_image_path)
    orig_image = Image.open(orig_image_path)

    # Load pre-trained Inception model
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()

    # Preprocess images
    generated_images_preprocessed = preprocess_images(gen_image)
    real_images_preprocessed = preprocess_images(orig_image)

    # Compute Inception Score
    is_mean, is_std = inception_score(generated_images_preprocessed, inception_model)
    print(f"Inception Score - Mean: {is_mean.item():.4f}, Standard Deviation: {is_std.item():.4f}")

    # Compute FID
    real_features = inception_model(real_images_preprocessed).detach().cpu()
    generated_features = inception_model(generated_images_preprocessed).detach().cpu()
    fid = calculate_fid(real_features, generated_features)
    print(f"FID: {fid.item():.4f}")

    return {
        "inception_score_mean": is_mean.item(),
        "inception_score_std": is_std.item(),
        "fid": fid.item()
    }

if __name__ == "__main__":
    # Example usage:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your models
    from your_model_file import Generator, Classifier
    generator = Generator().to(device)
    classifier = Classifier().to(device)

    # Load saved model weights
    generator.load_state_dict(torch.load('generator.pth'))
    classifier.load_state_dict(torch.load('classifier.pth'))

    # Set models to evaluation mode
    generator.eval()
    classifier.eval()

    # Load your test data
    # This would need to be adjusted based on your data loading approach
    import torch
    from torchvision import transforms
    from PIL import Image
    import os

    # Example of loading test data - replace with your actual data loading
    test_data_path = 'path/to/test_data'
    test_sketch_path = 'path/to/test_sketch'

    # Your data loading code here...
    # test_data, test_sketch, test_label = load_your_data()

    with torch.no_grad():
        # Generate and save original samples
        orig_path = save_original_samples(test_data, test_label)

        # Evaluate generator and save generated samples
        gen_path, cls_loss, cls_accuracy = evaluate_generator(generator, test_sketch, classifier, device)

        # Compute metrics
        metrics = compute_metrics(gen_path, orig_path)

        # Print overall evaluation results
        print("\nOverall Evaluation Results:")
        print(f"Classification Loss: {cls_loss:.4f}")
        print(f"Classification Accuracy: {cls_accuracy:.2f}%")
        print(f"Inception Score: {metrics['inception_score_mean']:.4f} ± {metrics['inception_score_std']:.4f}")
        print(f"FID Score: {metrics['fid']:.4f}")