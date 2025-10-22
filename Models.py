import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from Tournament import Tournament

class MobileNetBackbone(nn.Module):
    def __init__(self, device='cpu', output_dim=128, pretrained = False, freeze=False, unfreeze_last_n=1):
        super(MobileNetBackbone, self).__init__()
        self.device = device
        output_dim = int(output_dim)
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None) 
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
        self.class_head = nn.Sequential(mobilenet.classifier[0], mobilenet.classifier[1], mobilenet.classifier[2])
        self.fc = nn.Linear(mobilenet.classifier[3].in_features, output_dim)

    def forward(self, x):
        # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.class_head(x)
        x = self.fc(x)
        return x

class ResNet18Backbone(nn.Module):
    def __init__(self, device='cpu', output_dim=128, freeze=False, unfreeze_last_n=1):
        super(ResNet18Backbone, self).__init__()
        self.device = device

        # resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet18 = models.resnet18()
        resnet18.maxpool = nn.Identity()
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
        # if train:
            # x = F.softmax(x, dim=1)
        return x

class MidModel(torch.nn.Module):
    def __init__(self, class_count, backbone= 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=1):
        super(MidModel, self).__init__()
        model = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        edge_count = int(class_count * (class_count - 1) * 0.5)
        self.device = device
        self.model = model(device=device, output_dim=edge_count, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        # self.fc2 = nn.Linear(128, lass_count * (class_count - 1) * 0.5)
        self.fc3 = nn.Linear(edge_count, class_count)
        self.batchnorm2 = nn.BatchNorm1d(edge_count)
        self.batch_norm3 = nn.BatchNorm1d(class_count)
        # keep logits raw at the end

    def forward(self, x, train = False):
        x = self.model(x)
        x = self.batchnorm2(x)
        x = F.mish(x)
        x = self.fc3(x)
        # x = self.batch_norm3(x)
        # if train:
            # x = F.softmax(x, dim=1)
        return x


class AffineSigmoid(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        return torch.sigmoid(self.weight * x + self.bias)

class MinMaxScaler(torch.nn.Module):
    def __init__(self, alpha = .01):
        # alpa chino = alpha, but with a gun
        super().__init__()
        self.register_buffer('min_logit', torch.tensor(1.0))
        self.register_buffer('max_logit', torch.tensor(0.0))
        self.register_buffer('ema_min', torch.tensor(.5))
        self._min_val = 1  # Python float
        self._max_val = 0.0  # Python float
        self._ema_min = .5
        self.register_buffer('alpha', torch.tensor(alpha))

    def forward(self, x):
        temp_min = x.min().item()
        temp_max = x.max().item()

        # if temp_min < self._min_val:
            # self._min_val = temp_min
        if temp_max > self._max_val:
            self._max_val = temp_max
        self._ema_min = temp_min * self.alpha  + self.ema_min * (1 - self.alpha)

        # Update buffers (not used in computation graph)
        self.min_logit.fill_(self._min_val)
        self.max_logit.fill_(self._max_val)
        self.ema_min.fill_(self._ema_min)
        # scaled = torch.min(x, dim=1, keepdim=True)[0] * (self.alpha)
        # scaled += (1 - self.alpha) * (self.ema_min)
        out = (x - self.ema_min) / (self.max_logit - self.ema_min + 1e-6)
        # out = (x - scaled) *  1/(self.max_logit - self.ema_min + 1e-6)
        return out

class TournamentModel(torch.nn.Module):
    def __init__(self, class_count, backbone= 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=2):
        super(TournamentModel, self).__init__()
        self.device = device
        self.tournament = Tournament(num_classes=class_count)
        model = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = model(device=device, output_dim=self.tournament.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        self.batchnorm = nn.BatchNorm1d(self.tournament.num_edges)
        # self.layers = [self.model, self.batchnorm, self.sigmoid]
        self.asigmoid = AffineSigmoid(self.tournament.num_edges)
        # self.asigmoid = nn.Sigmoid()
        self.layers = [self.model, self.batchnorm, self.asigmoid]
        self.mms = MinMaxScaler()
        # self.layers = [self.model, self.asigmoid]
        self.middle = nn.Sequential(*self.layers)
    def forward(self, x, train = False):
        mid = self.middle(x)
        x = self.tournament(mid)
        # print(x.min(), x.max())
        # x = (x-self.tournament.min_logit) / (1-self.tournament.min_logit)
        # x = F.mish(x)
        # x = self.mms(x)
        # x = (x -.5 ) * 2
        # x = x * .5
        # if train:
            # x = F.softmax(x, dim=1)
        return x, mid


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def copy_matching_parameters(source_model, target_model):
    """
    Copy parameters from source_model to target_model for all layers
    where the parameter names and shapes match.
    """
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    matched_params = {}
    for name, param in source_state.items():
        if name in target_state and param.shape == target_state[name].shape:
            matched_params[name] = param

    # Load matched parameters into target model
    target_state.update(matched_params)
    target_model.load_state_dict(target_state)

    print(f"Copied {len(matched_params)} matching parameters from source to target.")

if __name__ == "__main__":
    m = MobileNetBackbone()
    print(m)