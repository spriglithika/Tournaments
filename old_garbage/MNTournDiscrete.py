import torch
import Tournament
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from importlib import reload
reload(Tournament)
sce = Tournament.symmetric_cross_entropy
Tournament = Tournament.Tournament
nn = torch.nn
F = nn.functional
class_count = 100
batch_size = 8
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3,1,1)), # convert to 3 channels
    transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)) # normalize for 3 channels
]))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class MobileNetBackbone(nn.Module):
    def __init__(self, device='cpu', output_dim=128, freeze=True, unfreeze_last_n=1):
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
        self.fc = nn.Sequential(
            nn.Linear(mobilenet.classifier[0].in_features, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class BaseModel(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(BaseModel, self).__init__()
        self.device = device
        self.model = MobileNetBackbone(device=device, output_dim=class_count)
        self.batchnorm = nn.BatchNorm1d(class_count)
        self.relu = nn.ReLU()

    def forward(self, x, train = False):
        x = self.model(x)
        # x = self.fc2(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        if train:
            x = F.softmax(x, dim=1)
        return x

class MidModel(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(MidModel, self).__init__()
        self.device = device
        self.model = MobileNetBackbone(device=device, output_dim=50*99)
        # self.fc2 = nn.Linear(128, 50*99)
        self.fc3 = nn.Linear(50*99, class_count)
        self.batchnorm2 = nn.BatchNorm1d(50*99)
        self.batch_norm3 = nn.BatchNorm1d(class_count)
        self.relu = nn.ReLU()

    def forward(self, x, train = False):
        x = self.model(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
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
    def __init__(self, device = 'cpu'):
        super(TournamentModel, self).__init__()
        self.device = device
        self.tournament = Tournament(num_classes=class_count)
        self.model = MobileNetBackbone(device=device, output_dim=self.tournament.num_edges, unfreeze_last_n=2)
        self.batchnorm = nn.BatchNorm1d(self.tournament.num_edges)
        # self.layers = [self.model, self.batchnorm, self.sigmoid]
        self.asigmoid = AffineSigmoid(self.tournament.num_edges)
        self.layers = [self.model, self.batchnorm, self.asigmoid]
        self.middle = nn.Sequential(*self.layers)
    def forward(self, x, train = False):
        x = self.middle(x)
        x = self.tournament(x)
        # if train:
        #     x = F.softmax(x, dim=1)
        return x



# load models
base_model = BaseModel(device=device).to(device)
base_model.load_state_dict(torch.load('/proj/cvl/users/x_hahel/Tournaments/ckpts/base_model_cifar100_MN.pth', map_location=device))
mid_model = MidModel(device=device).to(device)
mid_model.load_state_dict(torch.load('/proj/cvl/users/x_hahel/Tournaments/ckpts/mid_model_cifar100_MN.pth', map_location=device))

tournament_model = TournamentModel(device=device).to(device)
tournament_model.load_state_dict(torch.load('/proj/cvl/users/x_hahel/Tournaments/ckpts/tournament_model_cifar100_MN.pth', map_location=device))
def clip(data, threshold = .6):
    data = data - threshold
    data = torch.clamp(data, 0, (1-threshold))
    data = data * (1/(1-threshold))
    return data
confidences_base = torch.zeros((class_count,class_count))
confidences_mid = torch.zeros((class_count,class_count))
confidences_tournament = torch.zeros((class_count,class_count))
threshold = .8
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output_base = base_model(data,train=False)
        output_mid = mid_model(data,train=False)
        # output_base = clip(output_base, threshold = threshold)
        output_tournament = tournament_model(data,train=False)
        output_tournament = clip(output_tournament, threshold = threshold)
        for i in range(class_count):
            mask = (target == i)
            if mask.sum() > 0:
                confidences_base[i] += output_base[mask].sum(0).cpu()
                confidences_mid[i] += output_mid[mask].sum(0).cpu()
                confidences_tournament[i] += output_tournament[mask].sum(0).cpu()
confidences_base /= confidences_base.sum(1, keepdim=True)
confidences_mid /= confidences_mid.sum(1, keepdim=True)
confidences_tournament /= confidences_tournament.sum(1, keepdim=True)
# vmin = min(confidences_base.log().min(), confidences_tournament.log().min())
vmin, vmin_m, vmin_t = confidences_base.min(), confidences_mid.min(), confidences_tournament.min()
# vmax = max(confidences_base.log().max(), confidences_tournament.log().max())
vmax, vmax_m, vmax_t = confidences_base.max(), confidences_mid.max(), confidences_tournament.max()
plt.figure(figsize=(12,5))
plt.subplot(1,3,1)
plt.xticks(list(range(class_count)))
plt.yticks(list(range(class_count)))
plt.title('Base Model Confusion Matrix')
plt.imshow(confidences_base, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(1,3,2)
plt.xticks(list(range(class_count)))
plt.yticks(list(range(class_count)))
plt.title('Mid Model Confusion Matrix')
plt.imshow(confidences_mid, vmin=vmin_m, vmax=vmax_m)
plt.colorbar()
plt.subplot(1,3,3)
plt.xticks(list(range(class_count)))
plt.yticks(list(range(class_count)))
plt.title('Tournament Model Confusion Matrix')
plt.imshow(confidences_tournament, vmin=vmin_t, vmax=vmax_t)
plt.colorbar()
# plt.show()
plt.savefig('confusion_matrices_MN.png')

def select_by_binary_indices(data_array, K):
    data_array = torch.as_tensor(data_array)
    # pre-split for faster selection
    left = data_array[:, 0].to(torch.long)
    right = data_array[:, 1].to(torch.long)

    def batch_func(binary_batch):
        # binary_batch: (B, E) with values 0 or 1
        binary_batch = binary_batch.to(torch.long)
        B, E = binary_batch.shape

        # Expand left/right to (B, E)
        left_exp = left.unsqueeze(0).expand(B, -1)
        right_exp = right.unsqueeze(0).expand(B, -1)

        # Select per position: if binary == 0 pick left, else pick right
        selected = torch.where(binary_batch == 0, left_exp, right_exp)  # (B, E)

        # counts per class: (B, K)
        counts = F.one_hot(selected, num_classes=K).sum(dim=1).to(torch.long)

        # predicted class per sample and return one-hot (B, K)
        preds = counts.to(torch.float32).argmax(dim=1)
        return F.one_hot(preds, num_classes=K)

    return batch_func
# ground_truth_val, ground_truth_ind = get_gt_stuff(class_count)
def assign(x):
    x = x.clone()
    x[x <= .5] = 0
    x[x > .5] = 1
    out = select_by_binary_indices(tournament_model.tournament.perms, class_count)(x.to(torch.int64))
    return out
confidences_tournament = torch.zeros((class_count,class_count))
threshold = .8
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # output_base = clip(output_base, threshold = threshold)
        output_tournament = tournament_model.middle(data)
        output_tournament = assign(output_tournament)
        for i in range(class_count):
            mask = (target == i)
            if mask.sum() > 0:
                confidences_tournament[i] += output_tournament[mask].to(torch.float32).sum(0).cpu()
# confidences_base /= confidences_base.sum(1, keepdim=True)
# confidences_tournament = confidences_tournament.softmax(1)
confidences_tournament /= confidences_tournament.sum(0, keepdim=True)
# vmin = min(confidences_base.log().min(), confidences_tournament.log().min())
# vmin, vmin_t = confidences_base.min(), confidences_tournament.min()
# vmax = max(confidences_base.log().max(), confidences_tournament.log().max())
# vmax, vmax_t = confidences_base.max(), confidences_tournament.max()
plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.xticks(list(range(class_count)))
# plt.yticks(list(range(class_count)))
# plt.title('Base Model Confusion Matrix')
# plt.imshow(confidences_base, vmin=vmin, vmax=vmax)
# plt.colorbar()
plt.subplot(1,1,1)
plt.xticks(list(range(class_count)))
plt.yticks(list(range(class_count)))
plt.title(f'{confidences_tournament.trace()/confidences_tournament.sum()}')
plt.imshow(confidences_tournament, vmin=confidences_tournament.min(), vmax=confidences_tournament.max())
plt.colorbar()
# plt.show()
plt.savefig('confusion_matrice_discrete_MN.png')