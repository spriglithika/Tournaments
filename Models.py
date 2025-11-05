import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from itertools import combinations

from Tournament import Tournament, TournamentGaussianModel

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
        self.dropout = nn.Dropout(.2)
        self.fc = nn.Linear(mobilenet.classifier[3].in_features, output_dim)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.class_head(x)
        x = self.dropout(x)
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

class ResNet18BackboneExtended(nn.Module):
    def __init__(self, device='cpu', output_dim=128, freeze=False, unfreeze_last_n=1):
        super(ResNet18BackboneExtended, self).__init__()
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
        self.fc1 = nn.Linear(resnet18.fc.in_features, 2000)
        self.hswish = nn.Hardswish()
        self.dropout = nn.Dropout(.2, inplace=True)
        self.fc2 = nn.Linear(2000, output_dim)

    def forward(self, x):
        # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.hswish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class BaseModel(torch.nn.Module):
    def __init__(self, class_count, backbone= 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=1):
        super(BaseModel, self).__init__()
        model = ResNet18Backbone if backbone == 'resnet18' else ResNet18BackboneExtended if backbone == 'resnet18ext' else MobileNetBackbone
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
        model = ResNet18Backbone if backbone == 'resnet18' else ResNet18BackboneExtended if backbone == 'resnet18ext' else MobileNetBackbone
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
        

class NeuralIsingTournament(nn.Module):
    def __init__(self, num_classes, backbone = 'resnet18', device = 'cpu', learn_J=False, freeze_backbone=False, unfreeze_last_n=2):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        num_edges = num_classes * (num_classes - 1)// 2
        self.num_edges = num_edges
        self.backbone = backbone
        backbone = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.tournament = Tournament(num_classes=num_classes)
        self.model = backbone(device=device, output_dim=self.tournament.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        self.batchnorm = nn.BatchNorm1d(self.num_edges)
        # self.layers = [self.model, self.batchnorm, self.sigmoid]
        # self.asigmoid = AffineSigmoid(self.num_edges)
        # self.asigmoid = nn.Sigmoid()
        self.layers = [self.model, self.batchnorm]
        # self.mms = MinMaxScaler()
        # self.layers = [self.model, self.asigmoid]
        self.middle = nn.Sequential(*self.layers)
        adjacency_matrix = self.build_signed_adjacency()
        # adjacency_matrix = self.tournament.gt.T
        # self.bias_layer = nn.Linear(self.num_edges, num_edges)  # h_i
        if learn_J:
            self.J = nn.Parameter(adjacency_matrix.clone())
        else:
            self.register_buffer("J", adjacency_matrix)
        # for j in self.J:
        #     print(j)

    def build_signed_adjacency(self, gamma=1.0):
        edge_list = list(combinations(range(self.num_classes), 2))
        num_edges = len(edge_list)
        J = torch.zeros((num_edges, num_edges))

        for a, (i1, j1) in enumerate(edge_list):
            for b, (i2, j2) in enumerate(edge_list):
                if a == b:
                    J[a, b] = gamma  # self-interaction (optional)
                    continue
                shared = {i1, j1}.intersection({i2, j2})
                if shared:
                    shared_player = list(shared)[0]
                    # Determine role of shared player in both edges
                    same_role = ((shared_player == i1 and shared_player == i2) or
                                (shared_player == j1 and shared_player == j2))
                    J[a, b] = gamma if same_role else -gamma
        J = J / J.norm(p=2)
        # print(J)
        return J


    def forward(self, x, max_iter=10, alpha = 1.0, train=False):
        # Extract features
        # features = self.backbone(x)
        h = self.middle(x)
        # print("h mean:", h.mean().item(), "h min:", h.min().item(), "h max:", h.max().item())
        # h = self.bias_layer(features)  # shape: [batch, num_edges]

        # Mean-field inference for marginals
        # Initialize m_i = tanh(h_i)
        m = F.tanh(h)
        for _ in range(max_iter):
            m = torch.tanh(h + alpha * torch.matmul(m, self.J))  # update rule

        # Convert to probabilities in [0,1]
        probs = (m + 1) / 2
        # print('means', probs.mean().item(), 'stds:', probs.std().item())
        t_probs = self.tournament(probs)

        return t_probs, probs#, h


class TournamentModelVariational(torch.nn.Module):
    def __init__(self, class_count, backbone= 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=2):
        super(TournamentModelVariational, self).__init__()
        self.device = device
        self.tournament = Tournament(num_classes=class_count)
        self.variational = TournamentGaussianModel(self.tournament.num_edges, class_count, 10)
        model = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = model(device=device, output_dim=self.tournament.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        self.batchnorm = nn.BatchNorm1d(self.tournament.num_edges)
        # self.layers = [self.model, self.batchnorm, self.sigmoid]
        self.asigmoid = AffineSigmoid(self.tournament.num_edges)
        # self.asigmoid = nn.Sigmoid()
        self.layers = [self.model, self.batchnorm, self.asigmoid]
        # self.mms = MinMaxScaler()
        # self.layers = [self.model, self.asigmoid]
        self.middle = nn.Sequential(*self.layers)
    def inference(self, x):
        mid = self.middle(x)
        probs = self.variational.inference(mid, self.tournament.perms.int())
        # probs = F.sigmoid(probs)
        solved = self.tournament(probs)
        return solved, probs
    def forward(self, x, train = False):
        mid = self.middle(x)
        sample, mahalanobis, reg_loss, kl_div = self.variational(mid)
        sample = F.sigmoid(sample)
        x = self.tournament(sample)
        # print(x.min(), x.max())
        # x = (x-self.tournament.min_logit) / (1-self.tournament.min_logit)
        # x = F.mish(x)
        # x = self.mms(x)
        # x = (x -.5 ) * 2
        # x = x * .5
        # if train:
            # x = F.softmax(x, dim=1)
        return x, mid, sample, mahalanobis, reg_loss, kl_div

class TournamentModelDropout(torch.nn.Module):
    def __init__(self, class_count, mode ='mean', backbone= 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=2):
        super(TournamentModelDropout, self).__init__()
        self.device = device
        self.tournament = Tournament(num_classes=class_count, mode=mode)
        model = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = model(device=device, output_dim=self.tournament.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        self.batchnorm = nn.BatchNorm1d(self.tournament.num_edges)
        # self.layers = [self.model, self.batchnorm, self.sigmoid]
        # self.asigmoid = AffineSigmoid(self.tournament.num_edges)
        # self.asigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        # self.dropout = nn.Dropout(.05)
        self.dropout = nn.Dropout(.2)
        # self.layers = [self.model, self.batchnorm, self.asigmoid]
        self.layers = [self.model, self.batchnorm, self.tanh]
        self.mms = MinMaxScaler()
        # self.layers = [self.model, self.asigmoid]
        self.middle = nn.Sequential(*self.layers)

    def forward(self, x, train = False):
        backboned = self.middle(x)
        to_tourn = backboned# self.dropout(backboned)
        mid = (backboned + 1) / 2 
        to_tourn = (to_tourn + 1) / 2 
        x = self.tournament(to_tourn)
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