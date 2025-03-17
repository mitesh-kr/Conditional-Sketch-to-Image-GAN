"""
Discriminator model for conditional GAN
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Discriminator network that classifies images as real or fake
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, 4, 128, 128]
               (Image concatenated with label embedding)
            
        Returns:
            Tensor of shape [batch_size, 1] with values between 0 and 1
        """
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
