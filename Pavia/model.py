import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim1']
        self.backbone_dim2 = backbone['dim2']
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim2),
                nn.ReLU(), nn.Linear(self.backbone_dim2, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x, i):
        features = self.backbone(x, i)
        # features = F.normalize(features, dim=-1)
        out = self.contrastive_head(features)
        out = F.normalize(out, dim=-1)
        return features, out


class GaussianModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128, n_feature=200):
        super(GaussianModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim1']
        self.backbone_dim2 = backbone['dim2']
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim2),
                nn.ReLU(), nn.Linear(self.backbone_dim2, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

        self.fc1 = nn.Linear(n_feature, features_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, spe_x, i):
        spa_features = self.backbone(x, i)
        spe = self.fc1(spe_x)
        spe = self.relu(spe)

        features = torch.cat([spa_features, spe], dim=1)
        out = self.contrastive_head(features)
        out = F.normalize(out, dim=-1)
        return features, out


class SpectralModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128, n_feature=200):
        super(SpectralModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = features_dim
        self.backbone_dim2 = features_dim
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim2),
                nn.ReLU(), nn.Linear(self.backbone_dim2, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

        self.fc1 = nn.Linear(n_feature, features_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, spe_x, i):
        spe = self.fc1(spe_x)
        spe = self.relu(spe)

        out = self.contrastive_head(spe)
        out = F.normalize(out, dim=-1)
        return spe, out


class SpatialModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128, n_feature=200):
        super(SpatialModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = 128
        self.backbone_dim2 = 128
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim2),
                nn.ReLU(), nn.Linear(self.backbone_dim2, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

        self.fc1 = nn.Linear(n_feature, features_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, i):
        spa_features = self.backbone(x, i)
        out = self.contrastive_head(spa_features)
        out = F.normalize(out, dim=-1)
        return spa_features, out


class SpatialModel_vis(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128, n_feature=200):
        super(SpatialModel_vis, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = 128
        self.backbone_dim2 = 128
        self.head = head

        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                nn.Linear(self.backbone_dim, self.backbone_dim2),
                nn.ReLU(), nn.Linear(self.backbone_dim2, features_dim))

        else:
            raise ValueError('Invalid head {}'.format(head))

        self.fc1 = nn.Linear(n_feature, features_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, i):
        conv_features, pool_feature = self.backbone(x, i)
        out = self.contrastive_head(pool_feature)
        out = F.normalize(out, dim=-1)
        return conv_features, pool_feature, out

