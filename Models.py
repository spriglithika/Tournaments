import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from Tournament import Tournament

class MobileNetBackbone(nn.Module):
    def __init__(self, device='cpu', output_dim=128, freeze=False, unfreeze_last_n=1):
        super(MobileNetBackbone, self).__init__()
        self.device = device

        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.flatten = nn.Flatten()

        # Freeze all layers first
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.avgpool.parameters():
                param.requires_grad = False

            # Unfreeze last `n` blocks
            if unfreeze_last_n > 0:
                blocks = list(self.features.children())
                for block in blocks[-unfreeze_last_n:]:
                    for param in block.parameters():
                        param.requires_grad = True

        # Final projection to output_dim
        # Final projection to output_dim (no activation here -- produce raw features/logits)
        self.fc = nn.Linear(mobilenet.classifier[0].in_features, output_dim)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ResNet18Backbone(nn.Module):
    def __init__(self, device='cpu', output_dim=128, freeze=False, unfreeze_last_n=1):
        super(ResNet18Backbone, self).__init__()
        self.device = device

        # resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet18 = models.resnet18()
        self.features = nn.Sequential(*list(resnet18.children())[:-1])  # Exclude the final FC layer
        self.flatten = nn.Flatten()

        # Freeze all layers first
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

            # Unfreeze last `n` blocks
            if unfreeze_last_n > 0:
                blocks = list(self.features.children())
                for block in blocks[-unfreeze_last_n:]:
                    for param in block.parameters():
                        param.requires_grad = True

        # Final projection to output_dim
        # Final projection to output_dim (no activation here -- produce raw features/logits)
        self.fc = nn.Linear(resnet18.fc.in_features, output_dim)

    def forward(self, x):
        # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class BaseModel(torch.nn.Module):
    def __init__(self, class_count, backbone= 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=1):
        super(BaseModel, self).__init__()
        model = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.device = device
        self.model = model(device=device, output_dim=class_count, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        # Do not apply BatchNorm/ReLU to final logits; keep logits raw.

    def forward(self, x, train = False):
        x = self.model(x)
        # x = self.fc2(x)
        # x is raw logits here
        if train:
            x = F.softmax(x, dim=1)
        return x

class MidModel(torch.nn.Module):
    def __init__(self, class_count, backbone= 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=1):
        super(MidModel, self).__init__()
        model = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.device = device
        self.model = model(device=device, output_dim=50*99, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        # self.fc2 = nn.Linear(128, 50*99)
        self.fc3 = nn.Linear(50*99, class_count)
        self.batchnorm2 = nn.BatchNorm1d(50*99)
        self.batch_norm3 = nn.BatchNorm1d(class_count)
        # keep logits raw at the end

    def forward(self, x, train = False):
        x = self.model(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.batch_norm3(x)
        if train:
            x = F.softmax(x, dim=1)
        return x


class AffineSigmoid(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return torch.sigmoid(self.weight * x + self.bias)


class TournamentModel(torch.nn.Module):
    def __init__(self, class_count, backbone= 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=2):
        super(TournamentModel, self).__init__()
        self.device = device
        self.tournament = Tournament(num_classes=class_count)
        model = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = model(device=device, output_dim=self.tournament.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        self.batchnorm = nn.BatchNorm1d(self.tournament.num_edges)
        # self.layers = [self.model, self.batchnorm, self.sigmoid]
        # self.asigmoid = AffineSigmoid(self.tournament.num_edges)
        self.asigmoid = nn.Sigmoid()
        self.layers = [self.model, self.batchnorm, self.asigmoid]
        # self.layers = [self.model, self.asigmoid]
        self.middle = nn.Sequential(*self.layers)
    def forward(self, x, train = False):
        x = self.middle(x)
        x = self.tournament(x)
        x = (x-.5) * 2
        # if train:
        #     x = F.softmax(x, dim=1)
        return x