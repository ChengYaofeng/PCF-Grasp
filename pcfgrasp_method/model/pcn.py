import torch
import torch.nn as nn
from extensions.chamfer_distance.chamfer_distance import ChamferDistance


class PCN(nn.Module):
    """
    "PCN: Point Cloud Completion Network"
    (https://arxiv.org/pdf/1808.00671.pdf)

    Attributes:
        num_dense:  16384
        latent_dim: 1024
        grid_size:  4
        num_coarse: 1024
    """

    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4, split='pretrain'):
        super().__init__()

        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.split = split

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )

    def forward(self, xyz):
        B, N, _ = xyz.shape
        
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                                       # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]                          # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)              # (B,  512, N)
        feature = self.second_conv(feature)                                                  # (B, 1024, N)
        feature_global = torch.max(feature,dim=2,keepdim=False)[0]                           # (B, 1024)
        
        # decoder
        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)                    # (B, num_coarse, 3), default[B 1024 3]

        return coarse #BNC


CD = ChamferDistance()

def cd_loss(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0

def l1_cd_metric(pcs1, pcs2):
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.mean(torch.sqrt(dist1), 1)
    dist2 = torch.mean(torch.sqrt(dist2), 1)
    return torch.sum(dist1 + dist2) / 2