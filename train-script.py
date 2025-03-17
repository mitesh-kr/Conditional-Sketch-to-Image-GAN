"""
Main training script for conditional GAN
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.utils as vutils
import wandb

from models import Discriminator, Generator, Classifier, LabelEmbedding
from utils import load_data, prepare_data_for_training, denorm
from config import *

def train():
    # Initialize wandb
    wandb.login(key=os.environ.get('WANDB_API_KEY'))
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    
    # Load data
    data_dict = load_data(config=globals())
    train_data = data_dict['train_data']
    train_sketch = data_dict['train_sketch']
    test_data = data_dict['test_data']
    test_sketch = data_dict['test_sketch']
    train_label = data_dict['train_label']
    test_label = data_dict['test_label']
    
    # Initialize models
    discriminator = Discriminator().to(DEVICE)
    generator = Generator().to(DEVICE)
    classifier = Classifier().to(DEVICE)
    
    # Load pre-trained classifier if available
    try:
        classifier.load_state_dict(torch.load(CLASSIFIER_PATH))
        print(f"Loaded pre-trained classifier from {CLASSIFIER_PATH}")
    except (FileNotFoundError, RuntimeError):
        print(f"Pre-trained classifier not found at {CLASSIFIER_PATH}")
    
    # Initialize label embedding
    embedding_model = LabelEmbedding().to(DEVICE)
    
    # Prepare data for training
    train_data, train_loader = prepare_data_for_training(
        train_data, train_label, embedding_model, config=globals()
    )
    
    # Initialize optimizers
    loss_function = nn.BCELoss()
    loss_function2 = nn.CrossEntropyLoss()
    
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(BETA1, BETA2))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA1, BETA2))
    
    # Training loop
    for epoch in range(EPOCHS):
        d_loss_real_list = []
        d_loss_fake_list = []
        d_loss_total = 0
        g_loss_total = 0
        d_real_score = 0
        d_fake_score = 0
        classi_loss = 0
        classi_accuarcy = 0
        
        for X in train_loader:
            # Train discriminator
            d_optimizer.zero_grad()
            
            # Real data
            real_data = X
            real_label = torch.ones(X.shape[0], 1)
            d_real_output = discriminator(real_data.to(DEVICE))
            d_loss_real = loss_function(d_real_output, real_label.to(DEVICE))
            d_loss_real_list.append(d_loss_real.cpu().item())
            d_real_score += torch.mean(d_real_output).item()
            
            # Fake data
            X_sketch = train_sketch[torch.randperm(train_sketch.shape[0])[:X.shape[0]]]
            random_label = embedding_model(torch.eye(NUM_CLASSES)[torch.randint(0, NUM_CLASSES, size=(X.shape[0],))])
            X_sketch = torch.cat([X_sketch, random_label], dim=1)
            fake_data = generator(X_sketch.to(DEVICE))
            fake_data = torch.cat([fake_data, random_label.to(DEVICE)], dim=1)
            fake_label_d = torch.zeros(X.shape[0], 1)
            
            d_fake_output = discriminator(fake_data)
            d_loss_fake = loss_function(d_fake_output, fake_label_d.to(DEVICE))
            d_loss_fake_list.append(d_loss_fake.cpu().item())
            d_fake_score += torch.mean(d_fake_output).item()
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            d_loss_total += d_loss.item()
            
            # Train generator
            g_optimizer.zero_grad()
            X_sketch = train_sketch[torch.randperm(train_sketch.shape[0])[:X.shape[0]]]
            random_label = embedding_model(torch.eye(NUM_CLASSES)[torch.randint(0, NUM_CLASSES, size=(X.shape[0],))])
            X_sketch = torch.cat([X_sketch, random_label], dim=1)
            fake_data = generator(X_sketch.to(DEVICE))
            fake_data = torch.cat([fake_data, random_label.to(DEVICE)], dim=1)
            
            g_output = discriminator(fake_data)
            fake_label_g = torch.ones(X.shape[0], 1)
            g_loss = loss_function(g_output, fake_label_g.to(DEVICE))
            g_loss.backward()
            g_optimizer.step()
            g_loss_total += g_loss.item()
            
            # Train generator with classification loss
            g_optimizer.zero_grad()
            X_sketch = train_sketch[torch.randperm(train_sketch.shape[0])[:X.shape[0]]]
            
            classification_true_labels = torch.eye(NUM_CLASSES)[torch.randint(0, NUM_CLASSES, size=(X.shape[0],))]
            
            random_label = embedding_model(classification_true_labels)
            X_sketch = torch.cat([X_sketch, random_label], dim=1)
            fake_data = generator(X_sketch.to(DEVICE))
            y_pred_classification = classifier(fake_data)
            
            classification_loss = loss_function2(y_pred_classification, torch.max(classification_true_labels.to(DEVICE), 1)[1])
            classification_acc = ((torch.max(y_pred_classification, 1)[1] == torch.max(classification_true_labels.to(DEVICE), 1)[1]).sum().item()) / X.shape[0]
            classification_loss.backward()
            g_optimizer.step()
            classi_loss += classification_loss.item()
            classi_accuarcy += classification_acc
        
        # Calculate epoch metrics
        d_real_loss = np.mean(d_loss_real_list)
        d_fake_loss = np.mean(d_loss_fake_list)
        
        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "D_Loss_Real": d_real_loss,
            "D_Loss_Fake": d_fake_loss,
            "D_Loss": d_loss_total / len(train_loader),
            "G_Loss": g_loss_total / len(train_loader),
            "D_Real_Score": d_real_score / len(train_loader),
            "D_Fake_Score": d_fake_score / len(train_loader),
            "C_Loss": classi_loss / len(train_loader),
            "C_Accuracy": (classi_accuarcy * 100) / len(train_loader)
        })
        
        # Print epoch metrics
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"D_Loss_Real: {d_real_loss:.4f} | "
              f"D_Loss_Fake: {d_fake_loss:.4f} | "
              f"D_Real_Score: {d_real_score / len(train_loader):.4f} | "
              f"D_Fake_Score: {d_fake_score / len(train_loader):.4f} | "
              f"D_Loss: {d_loss_total / len(train_loader):.4f} | "
              f"G_Loss: {g_loss_total / len(train_loader):.4f} | "
              f"C_Loss: {classi_loss / len(train_loader):.4f}  | "
              f"C_Accuracy: {(classi_accuarcy * 100) / len(train_loader):.4f}")
        
        # Generate and save sample images for each class
        generate_class_samples(generator, embedding_model, test_sketch, classifier, epoch)
        
        # Save model checkpoints
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(generator.state_dict(), os.path.join(os.path.dirname(GENERATOR_PATH), f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(os.path.dirname(DISCRIMINATOR_PATH), f'discriminator_epoch_{epoch+1}.pth'))
    
    # Save final models
    torch.save(generator.state_dict(), GENERATOR_PATH)
    torch.save(discriminator.state_dict(), DISCRIMINATOR_PATH)
    
    # Generate original samples for comparison
    generate_original_samples(test_data, test_label)
    
    wandb.finish()

def generate_class_samples(generator, embedding_model, test_sketch, classifier, epoch):
    """
    Generate and save sample images for each class
    """
    num_class = NUM_CLASSES
    grid = []
    classi_loss = 0
    classi_accuarcy = 0
    
    for c in range(num_class):
        samples = test_sketch[torch.randperm(test_sketch.shape[0])[:10]]
        label = torch.eye(num_class)[c]
        labels = label.unsqueeze(0).repeat(samples.shape[0], 1)
        labels_classification = torch.max(labels, 1)[1]
        labels = embedding_model(labels)
        samples = torch.cat([samples, labels], dim=1)
        images = generator(samples.to(DEVICE))
        
        y_pred_cla = classifier(images)
        classification_loss = F.cross_entropy(y_pred_cla, labels_classification.to(DEVICE))
        classi_loss += classification_loss.item()
        
        classification_accuarcy = (((torch.max(y_pred_cla, 1)[1] == labels_classification.to(DEVICE))).sum().item()) / 10
        classi_accuarcy += classification_accuarcy
        
        grid_image = vutils.make_grid(images, nrow=10, padding=2)
        grid.append(grid_image)
    
    grid = torch.cat(grid, dim=1)
    directory = os.path.join(GENERATED_IMAGES_DIR, f'lr_{LR}')
    os.makedirs(directory, exist_ok=True)
    vutils.save_image(denorm(grid), f"{directory}/epoch_{epoch}.png")
    
    print(f'classification_loss: {classi_loss / num_class} | '
          f'classification_accuracy: {((classi_accuarcy / num_class) * 100):.4f}')

def generate_original_samples(test_data, test_label):
    """
    Generate and save original samples for comparison
    """
    num_class = NUM_CLASSES
    grid = []
    
    for c in range(num_class):
        images = test_data[torch.max(test_label.data, 1)[1] == c][:10]
        grid_image = vutils.make_grid(images, nrow=10, padding=2)
        grid.append(grid_image)
    
    grid = torch.cat(grid, dim=1)
    os.makedirs(ORIGINAL_SAMPLES_DIR, exist_ok=True)
    vutils.save_image(denorm(grid), f"{ORIGINAL_SAMPLES_DIR}/Original_sample.png")

if __name__ == "__main__":
    train()
