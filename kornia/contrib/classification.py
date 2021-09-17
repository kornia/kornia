"""Module containing utilities for image classification."""
import torch
from torch import nn


class ImageClassificationHead(nn.Module):
    """Module to be used as an image classification head.
    
    Args:
        embed_size: the logits tensor coming from the networks.
        num_classes: an integer representing the numbers of classes to classify.
    
    Example:
        >>> feat = torch.rand(1, 256, 256)
        >>> head = ImageClassificationHead(256, 10)
        >>> head(feat).shape
        torch.Size([1, 10])
    """
    def __init__(self, embed_size: int = 768, num_classes: int = 10) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_size) 
        self.linear = nn.Linear(embed_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x.mean(-2)
        return self.linear(self.norm(out))
