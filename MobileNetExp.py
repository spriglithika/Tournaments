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

print("Modules loaded: We have liftoff!")

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
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
class_count = torch.max(torch.tensor(train_dataset.targets)) + 1
image_shape = train_dataset[0][0].shape


class MobileNetBackbone(nn.Module):
    def __init__(self, device='cpu', output_dim=128, freeze=True, unfreeze_last_n=1):
        super(MobileNetBackbone, self).__init__()
        self.device = device

        mobilenet = models.mobilenet_v3_large(pretrained=True)
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, train = False):
        x = self.model(x)
        # x = self.fc2(x)
        x = self.batchnorm(x)
        # x = self.sigmoid(x)
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, train = False):
        x = self.model(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.batch_norm3(x)
        # x = self.sigmoid(x)
        if train:
            x = F.softmax(x, dim=1)
        return x

class TournamentModel(torch.nn.Module):
    def __init__(self, device = 'cpu'):
        super(TournamentModel, self).__init__()
        self.device = device
        self.tournament = Tournament(num_classes=class_count)
        self.model = MobileNetBackbone(device=device, output_dim=self.tournament.num_edges)
        # self.fc2 = nn.Linear(128, self.tournament.num_edges)
        self.batchnorm = nn.BatchNorm1d(self.tournament.num_edges)
        # self.batchnorm = nn.Identity(self.tournament.num_edges)
        self.sigmoid = nn.Sigmoid()
        self.layers = [self.model, self.batchnorm, self.sigmoid]
        self.middle = nn.Sequential(*self.layers)
    def forward(self, x, train = False):
        x = self.middle(x)
        x = self.tournament(x)
        if train:
            x = F.softmax(x, dim=1)
        return x


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), F.one_hot(target.to(device), num_classes=class_count).float()
        optimizer.zero_grad()
        output = model(data, train=True)
        # print(output.min(), output.max())
        loss = sce(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({'loss': loss.item()})
def test_basic(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target_oh = F.one_hot(target, num_classes=class_count).float()
            output_sm = model(data, train=True)
            output = model(data)
            test_loss += sce(output_sm, target_oh, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def clip(data, threshold = .6):
    data = data - threshold
    data = torch.clamp(data, 0, (1-threshold))
    data = data * (1/(1-threshold))
    return data
# clip_test = torch.linspace(0,1,25)
# print(clip(clip_test))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# backbone = MLPModel
base_model = BaseModel(device = device).to(device)
mid_model = MidModel(device = device).to(device)
tournament_model = TournamentModel(device = device).to(device)
optimizer_base = torch.optim.Adam(base_model.parameters(), lr=0.001)
optimizer_mid = torch.optim.Adam(mid_model.parameters(), lr=0.001)
optimizer_tournament = torch.optim.Adam(tournament_model.parameters(), lr=0.001)
num_epochs = 10

print("Everything is defined! Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch}: Base Model")
    train(base_model, device, train_loader, optimizer_base, epoch)
    test_basic(base_model, device, test_loader)
    print(f"Epoch {epoch}: Mid Model")
    train(mid_model, device, train_loader, optimizer_mid, epoch)
    test_basic(mid_model, device, test_loader)
    print(f"Epoch {epoch}: Tournament Model")
    train(tournament_model, device, train_loader, optimizer_tournament, epoch)
    test_basic(tournament_model, device, test_loader)

print("Whew! Training complete. Saving models and generating confusion matrices...")
base_model_path = "./ckpts/base_model_MN.pth"
torch.save(base_model.state_dict(), base_model_path)
mid_model_path = "./ckpts/mid_model_MN.pth"
torch.save(mid_model.state_dict(), mid_model_path)
tournament_model_path = "./ckpts/tournament_model_MN.pth"
torch.save(tournament_model.state_dict(), tournament_model_path)

confidences_base = torch.zeros((class_count,class_count))
confidences_tournament = torch.zeros((class_count,class_count))
threshold = .8
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output_base = base_model(data,train=False)
        # output_base = clip(output_base, threshold = threshold)
        output_tournament = tournament_model(data,train=False)
        output_tournament = clip(output_tournament, threshold = threshold)
        for i in range(class_count):
            mask = (target == i)
            if mask.sum() > 0:
                confidences_base[i] += output_base[mask].sum(0).cpu()
                confidences_tournament[i] += output_tournament[mask].sum(0).cpu()
confidences_base /= confidences_base.sum(1, keepdim=True)
confidences_tournament /= confidences_tournament.sum(1, keepdim=True)
# vmin = min(confidences_base.log().min(), confidences_tournament.log().min())
vmin, vmin_t = confidences_base.min(), confidences_tournament.min()
# vmax = max(confidences_base.log().max(), confidences_tournament.log().max())
vmax, vmax_t = confidences_base.max(), confidences_tournament.max()
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.xticks(list(range(class_count)))
plt.yticks(list(range(class_count)))
plt.title('Base Model Confusion Matrix')
plt.imshow(confidences_base, vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(1,2,2)
plt.xticks(list(range(class_count)))
plt.yticks(list(range(class_count)))
plt.title('Tournament Model Confusion Matrix')
plt.imshow(confidences_tournament, vmin=vmin_t, vmax=vmax_t)
plt.colorbar()
# plt.show()
plt.savefig('confusion_matrices_MN.png')

# def get_gt_stuff(num_classes):
#     ground_truth_indicies, ground_truth_values = [],[]
#     for k in range(num_classes):
#         i,v = torch.where(tournament_model.tournament.perms == k)
#         ground_truth_indicies.append(i)
#         ground_truth_values.append(v)
#     return torch.stack(ground_truth_values).T, torch.stack(ground_truth_indicies).T
plt.clf()
def select_by_binary_indices(data_array, K):
    def func(binary_array):
        selected_values = data_array[torch.arange(binary_array.shape[0]), binary_array].to(torch.int64)
        counts = torch.bincount(selected_values, minlength=K)
        # m = torch.max(counts)
        # mask = counts == m
        # c = mask.sum()
        # condition = (c == 1).to(torch.int32)
        # out = condition * (counts.argmax()+1) - 1
        return F.one_hot(counts.to(torch.float32).argmax(0), num_classes=K)
    return torch.func.vmap(func)
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
confidences_base /= confidences_base.sum(1, keepdim=True)
# confidences_tournament = confidences_tournament.softmax(1)
confidences_tournament /= confidences_tournament.sum(0, keepdim=True)
# vmin = min(confidences_base.log().min(), confidences_tournament.log().min())
vmin, vmin_t = confidences_base.min(), confidences_tournament.min()
# vmax = max(confidences_base.log().max(), confidences_tournament.log().max())
vmax, vmax_t = confidences_base.max(), confidences_tournament.max()
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
plt.imshow(confidences_tournament, vmin=vmin_t, vmax=vmax_t)
plt.colorbar()
# plt.show()
plt.savefig('confusion_matrice_discrete_MN.png')

print("All done!")