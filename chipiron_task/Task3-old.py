# Task 3 - Loss function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms

# Load VGG19 with pretrained weights
vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:16].eval()
for param in vgg.parameters():
    param.requires_grad = False
#
# VGG Normalization (mean and std from ImageNet)
vgg_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

class TVMSELoss(nn.Module):
    def __init__(self, tv_weight=1e-5, perc_weight=1e-1):
        super(TVMSELoss, self).__init__()
        # self.vgg = vgg
        self.tv_weight = tv_weight
        self.perc_weight = perc_weight
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        # Ensure float32 for compatibility with VGG
        output = output.float()
        target = target.float()

        # If grayscale (1 channel), repeat to 3 channels
        if output.shape[1] == 1:
            output = output.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)


        # --- Loss components ---

        # 1. MSE Loss
        mse_loss = self.mse(output, target)

        # 2. Total Variation Loss
        h_diff = torch.pow(output[:, :, 1:, :] - output[:, :, :-1, :], 2).mean()
        w_diff = torch.pow(output[:, :, :, 1:] - output[:, :, :, :-1], 2).mean()
        tv_loss = self.tv_weight * (h_diff + w_diff)

        # 3. Perceptual (VGG) Loss
        output_norm = torch.stack([vgg_normalize(img) for img in output])
        target_norm = torch.stack([vgg_normalize(img) for img in target])
        vgg_device = next(self.vgg.parameters()).device
        output_norm = output_norm.to(dtype=torch.float32, device=vgg_device)
        target_norm = target_norm.to(dtype=torch.float32, device=vgg_device)
        # Pass through VGG
        output_features = self.vgg(output_norm)
        target_features = self.vgg(target_norm)
        perc_loss = self.perc_weight * self.mse(output_features, target_features)

        # 4. Total Loss
        total_loss = mse_loss + tv_loss + perc_loss
        return total_loss

# Instantiate criterion
criterion = TVMSELoss(tv_weight=1e-1, perc_weight=0)
