"""Module that implements DETR for object detection using transformers.

Inspired by:
- https://medium.com/swlh/object-detection-with-transformers-437217a3d62e
- https://github.com/facebookresearch/detr/blob/main/models/detr.py
"""
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet50  # we do not want to depend on this


class DETR(nn.Module):
    """Minimal Example of the Detection Transformer model with learned positional embedding"""
    
    def __init__(self, num_classes, hidden_dim, num_heads, num_enc_layers, num_dec_layers) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        
        # CNN Backbone
        # TODO: find a better way to use any backbone
        self.backbone = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        
        # Transformer
        self.transformer = nn.Transformer(hidden_dim, num_heads,
            num_enc_layers, num_dec_layers)
            
        # Prediction Heads
        self.to_classes = nn.Linear(hidden_dim, num_classes+1)
        self.to_bbox = nn.Linear(hidden_dim, 4)
        
        # Positional Encodings
        self.object_query = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(x)
        h = self.conv(x)
        H, W = h.shape[-2:]
        
        pos_enc = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H,1,1),
            self.row_embed[:H].unsqueeze(1).repeat(1,W,1)],
        dim=-1).flatten(0,1).unsqueeze(1)
        
        h = self.transformer(pos_enc + h.flatten(2).permute(2,0,1),
        self.object_query.unsqueeze(1))
        
        class_pred = self.to_classes(h)
        bbox_pred = self.to_bbox(h).sigmoid()

        # return batch first
        class_pred = class_pred.permute(1, 0, 2)
        bbox_pred = bbox_pred.permute(1, 0, 2)

        return class_pred, bbox_pred
