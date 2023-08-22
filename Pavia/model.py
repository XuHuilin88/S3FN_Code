import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50


# class Model(nn.Module):
#     def __init__(self, feature_dim=128):
#         super(Model, self).__init__()
#
#         self.f = []
#         for name, module in resnet50().named_children():
#             if name == 'conv1':
#                 module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#             if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
#                 self.f.append(module)
#         # encoder
#         self.f = nn.Sequential(*self.f)
#         # projection head
#         self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
#                                nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
#
#     def forward(self, x):
#         x = self.f(x)
#         feature = torch.flatten(x, start_dim=1)
#         out = self.g(feature)
#         return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


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

        # features = self.contrastive_head(self.backbone(x))
        # features = F.normalize(features, dim=1)
        # return features, features


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


# class ClusterModel18(nn.Module):
#     def __init__(self, backbone, nclusters):
#         super(ClusterModel18, self).__init__()
#         self.backbone = backbone['backbone']
#         self.backbone_dim = backbone['dim']
#         self.cluster_head = nn.Linear(self.backbone_dim, nclusters)
#
#     def forward(self, x, forward_pass='default'):
#         if forward_pass == 'default':
#             features = self.backbone(x)
#             out = self.cluster_head(features)
#
#         elif forward_pass == 'backbone':
#             out = self.backbone(x)
#
#         elif forward_pass == 'head':
#             out = self.cluster_head(x)
#
#         elif forward_pass == 'return_all':
#             features = self.backbone(x)
#             out = {'features': features, 'output': self.cluster_head(features)}
#
#         else:
#             raise ValueError('Invalid forward pass {}'.format(forward_pass))
#
#         return out
#
#
# class ClusterModel(nn.Module):
#     def __init__(self, nclusters=9):
#         super(ClusterModel, self).__init__()
#
#         self.f = []
#         for name, module in resnet50().named_children():
#             if name == 'conv1':
#                 module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#             if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
#                 self.f.append(module)
#         # encoder
#         self.f = nn.Sequential(*self.f)
#         # cluster head
#         self.cluster_head = nn.Linear(2048, nclusters)
#         # self.cluster_head = nn.ModuleList([nn.Linear(2048, nclusters) for _ in range(1)])
#
#     def forward(self, x, forward_pass='default'):
#         if forward_pass == 'default':
#             features = self.f(x)
#             out = self.cluster_head(features)
#
#         elif forward_pass == 'return_all':
#             features = self.f(x)
#             output = self.cluster_head(features)
#             out = {'features': features, 'output': output}
#
#         return out


# class BaseNet(nn.Module):
#     def __init__(self, num_features=103, feature_dim=128):
#         super(BaseNet, self).__init__()
#
#         self.conv0 = nn.Conv2d(5, 64, kernel_size=1, stride=1,
#                                bias=True)
#         self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                                bias=True)
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
#                                bias=True)
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
#         self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#
#         self.num_features = num_features
#
#         self.dropout = dropout
#         self.drop = nn.Dropout(self.dropout)
#
#         n_fc1 = 1024
#         n_fc2 = 512
#
#         self.feat_spe = nn.Linear(self.num_features, n_fc1)
#         self.feat_ss = nn.Linear(n_fc1 + n_fc1, n_fc2)
#
#         # self.classifier = nn.Linear(n_fc2, self.num_classes)
#
#         self.g = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
#                                nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))
#
#     def forward(self, x, y):
#         x = self.conv0(x)
#         x_res = x
#         x = self.conv1(x)
#         x = self.relu(x + x_res)
#         x = self.avgpool(x)
#         x_res = x
#         x = self.conv2(x)
#         x = self.relu(x + x_res)
#         x = self.avgpool(x)
#
#         x = x.view(x.size(0), -1)
#
#         y = self.feat_spe(y)
#         y = self.relu(y)
#
#         feature = torch.cat([x, y], 1)
#
#         out = self.g(feature)
#         return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)