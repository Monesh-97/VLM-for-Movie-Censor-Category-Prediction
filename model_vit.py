import torch
import torch.nn as nn
from transformers import ViTModel

class VideoViTClassifier(nn.Module):
    def __init__(self, num_classes=3, num_frames=16, vit_name='google/vit-base-patch16-224'):
        """
        Custom Vision model for Videos.
        Uses a pretrained Spatial ViT on each frame, pools the temporal axis, and classifies.
        """
        super().__init__()
        
        # Load Pretrained Image ViT backbone
        self.vit = ViTModel.from_pretrained(vit_name)
        
        # Hidden dimension size (768 for ViT-Base)
        hidden_dim = self.vit.config.hidden_size
        
        # Classifier Head
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        x: Input video tensor of shape [Batch, Frames, Channels, Height, Width]
        Returns: logits of shape [Batch, NumClasses]
        """
        B, F, C, H, W = x.shape
        
        # Collapse Batch and Frame dimension to process frames independently through ViT
        # Shape: [B*F, C, H, W]
        x_flat = x.view(B * F, C, H, W)
        
        # Get ViT CLS tokens. Shape: [B*F, seq_len, 768] Ensure return_dict=True
        outputs = self.vit(pixel_values=x_flat)
        # We take the [CLS] token which is at sequence index 0
        cls_tokens = outputs.last_hidden_state[:, 0, :] # Shape: [B*F, 768]
        
        # Unflatten back to [B, F, 768]
        features = cls_tokens.view(B, F, -1)
        
        # --- Temporal Aggregation ---
        # Mean pooling across the temporal (frame) dimension
        video_features = features.mean(dim=1) # Shape: [B, 768]
        
        # Classify
        video_features = self.dropout(video_features)
        logits = self.fc(video_features)
        
        return logits
