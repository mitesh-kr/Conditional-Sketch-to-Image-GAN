"""
Label Embedding model for conditional GAN
"""

import torch
import torch.nn as nn

class LabelEmbedding(nn.Module):
    """
    Embeds class labels into a spatial representation that can be used by the generator
    """
    def __init__(self):
        super(LabelEmbedding, self).__init__()
        self.fc1 = nn.Linear(7, 256)
        self.fc2 = nn.Linear(256, 1024)
        self.fc3 = nn.Linear(1024, 128*128)

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape [batch_size, 7]
            
        Returns:
            Embedded label tensor of shape [batch_size, 1, 128, 128]
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = x.view(-1, 128, 128)
        x = x.unsqueeze(1)
        return x.detach()
