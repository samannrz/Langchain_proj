# Task 3 - Loss function

import torch
import torch.nn as nn
import torch.nn.functional as F
# from hrinversion import VGG16ConvLoss

class TVMSELoss(nn.Module):
    def __init__(self, tv_weight=1e-5, perc_weight=1e-1):
        super(TVMSELoss, self).__init__()
        # self.vgg = vgg
        self.tv_weight = tv_weight
        self.perc_weight = perc_weight
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        # Ensure float32 for compatibility with VGG

        # --- Loss components ---
        # 1. MSE Loss

        mse_loss = self.mse(output, target)

        # 2. Total Variation Loss

        h_diff = torch.pow(output[:, :, 1:, :] - output[:, :, :-1, :], 2).mean()
        w_diff = torch.pow(output[:, :, :, 1:] - output[:, :, :, :-1], 2).mean()
        tv_loss = self.tv_weight * (h_diff + w_diff)

        # 3. Perceptual (VGG) Loss
        # VGG conv-based perceptual loss
        percep_loss = VGG16ConvLoss().cuda().requires_grad_(False)

        # high-level perceptual loss: d_h
        # percep_loss = VGG16ConvLoss(fea_dict={'features_2': 0., 'features_7': 0., 'features_14': 0.,
        #                                       'features_21': 0.0002, 'features_28': 0.0005,
        #                                       }).cuda().requires_grad_(False)

        fea_target = percep_loss(target)
        fea_pred = percep_loss(output)

        percep_loss = self.perc_weight* F.mse_loss(fea_pred, fea_target, reduction='sum') / 4  # normalized by batch size
        # 4. Total Loss
        total_loss = mse_loss + tv_loss + percep_loss
        return total_loss

# Instantiate criterion
criterion = TVMSELoss(tv_weight=0, perc_weight=1e-1)
