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

print("MobileNetExpCasted: Modules loaded")

train_dataset = datasets.CIFAR100('../data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
test_dataset = datasets.CIFAR100('../data', train=False,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]))

# Use pinned memory to allow async host->device transfers
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory=True)
class_count = torch.max(torch.tensor(train_dataset.targets)) + 1
image_shape = train_dataset[0][0].shape

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

# joint training: keep all models on GPU and compute losses for each per batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = BaseModel(device = device).to(device)
mid_model = MidModel(device = device).to(device)
tournament_model = TournamentModel(device = device).to(device)
optimizer_base = torch.optim.Adam(base_model.parameters(), lr=0.001)
optimizer_mid = torch.optim.Adam(mid_model.parameters(), lr=0.001)
optimizer_tournament = torch.optim.Adam(tournament_model.parameters(), lr=0.001)

# Single AMP scaler for joint backward to avoid per-model scaler interactions
scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

num_epochs = 10


def joint_train_all(device, train_loader, optimizers, models, epoch):
    # models: dict with keys 'base','mid','tournament' mapping to (model, scaler, optimizer)
    pbar = tqdm(train_loader)
    # make sure all models are in training mode (joint_eval_all sets them to eval())
    for name, (m, _s, opt) in models.items():
        m.train()
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device, non_blocking=True)
        target = F.one_hot(target.to(device, non_blocking=True), num_classes=class_count).float()

        # zero grads for all optimizers
        for _, (m, _s, opt) in models.items():
            opt.zero_grad()

        # forward under autocast
        with torch.amp.autocast(enabled=torch.cuda.is_available(), device_type=device.type):
            out_base = models['base'][0](data, train=True)
            out_mid = models['mid'][0](data, train=True)
            out_tourn = models['tournament'][0](data, train=True)

            loss_base = sce(out_base, target)
            loss_mid = sce(out_mid, target)
            loss_tourn = sce(out_tourn, target)

        # scale the summed loss and backward once to keep AMP stable
        total_loss = loss_base + loss_mid + loss_tourn
        scaler.scale(total_loss).backward()

        # optional: unscale and clip gradients per-model to avoid explosion
        # (unscale requires passing the optimizer whose params' grads should be unscaled)
        for name, (m, s, opt) in models.items():
            # `s` entry is ignored now (kept for compatibility in the models dict)
            try:
                scaler.unscale_(opt)
            except Exception:
                pass
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)

        # step each optimizer (scaler.step handles skipped steps due to infs/nans)
        for name, (m, s, opt) in models.items():
            scaler.step(opt)

        # update scaler once per iteration
        scaler.update()

        pbar.set_postfix({'loss_base': loss_base.item(), 'loss_mid': loss_mid.item(), 'loss_tourn': loss_tourn.item()})

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
sbbi = select_by_binary_indices(tournament_model.tournament.perms, class_count)
def assign(x):
    x = x.clone()
    x[x <= .5] = 0
    x[x > .5] = 1
    out = sbbi(x.to(torch.int64))
    return out

def joint_eval_all(device, test_loader, models):
    for name, (m, _s, opt) in models.items():
        m.eval()
    test_loss = {k:0 for k in models.keys()}
    correct = {k:0 for k in models.keys()}
    correct['tournament_disc'] = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target_oh = F.one_hot(target, num_classes=class_count).float()
            out_base = models['base'][0](data, train=True)
            out_mid = models['mid'][0](data, train=True)
            out_tourn = models['tournament'][0](data, train=True)
            out_tourn_disc = models['tournament'][0].middle(data)
            print(out_tourn_disc.min(), out_tourn_disc.max())
            out_tourn_disc = assign(out_tourn_disc)
            test_loss['base'] += sce(out_base, target_oh, reduction='sum').item()
            test_loss['mid'] += sce(out_mid, target_oh, reduction='sum').item()
            test_loss['tournament'] += sce(out_tourn, target_oh, reduction='sum').item()


            pred_base = out_base.argmax(dim=1, keepdim=True)
            pred_mid = out_mid.argmax(dim=1, keepdim=True)
            pred_tourn = out_tourn.argmax(dim=1, keepdim=True)
            pred_tourn_disc = out_tourn_disc.argmax(dim=1, keepdim=True)

            correct['base'] += pred_base.eq(target.view_as(pred_base)).sum().item()
            correct['mid'] += pred_mid.eq(target.view_as(pred_mid)).sum().item()
            correct['tournament'] += pred_tourn.eq(target.view_as(pred_tourn)).sum().item()
            correct['tournament_disc'] += pred_tourn_disc.eq(target.view_as(pred_tourn_disc)).sum().item()

    for k in models.keys():
        test_loss[k] /= len(test_loader.dataset)
        accuracy = 100. * correct[k] / len(test_loader.dataset)
        print(f"{k} Test set: Average loss: {test_loss[k]:.4f}, Accuracy: {correct[k]}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    accuracy_disc = 100. * correct['tournament_disc'] / len(test_loader.dataset)
    print(f"tournament_disc Test set: Average loss: {test_loss['tournament']:.4f}, Accuracy: {correct['tournament_disc']}/{len(test_loader.dataset)} ({accuracy_disc:.2f}%)")


models = {
    'base': (base_model, None, optimizer_base),
    'mid': (mid_model, None, optimizer_mid),
    'tournament': (tournament_model, None, optimizer_tournament)
}

print("Starting joint training...")
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    joint_train_all(device, train_loader, None, models, epoch)
    joint_eval_all(device, test_loader, models)

# Loop to save the models after training
torch.save(base_model.state_dict(), 'ckpts/base_model_cifar100_MN_unfrozen.pth')
torch.save(mid_model.state_dict(), 'ckpts/mid_model_cifar100_MN_unfrozen.pth')
torch.save(tournament_model.state_dict(), 'ckpts/tournament_model_cifar100_MN_unfrozen.pth')

print("Done")
