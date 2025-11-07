from preamble import *
import Tournament
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from Models import BaseModel, MidModel, TournamentModel, copy_matching_parameters, NeuralIsingTournamentFull
from Training_testing import joint_train_all_ising, joint_eval_all, ConvergenceMonitor
from argparse import ArgumentParser


TournamentModel = NeuralIsingTournamentFull
sce = Tournament.symmetric_cross_entropy
Tournament = Tournament.Tournament

print("MobileNetExpCasted: Modules loaded")


class SubsetImageNet(torch.utils.data.Dataset):
    def __init__(self, dataset: datasets.ImageNet, subset_classes: list[int]):
        self.dataset = dataset
        self.subset_classes = set(subset_classes)

        # Filter samples to only those in subset_classes
        self.filtered_samples = [
            (path, label) for path, label in dataset.samples
            if label in self.subset_classes
        ]

        # Remap labels to [0, len(subset_classes)-1]
        self.label_remap = {old: new for new, old in enumerate(subset_classes)}

    def __getitem__(self, idx):
        path, label = self.filtered_samples[idx]
        img = self.dataset.loader(path)
        if self.dataset.transform:
            img = self.dataset.transform(img)
        return img, self.label_remap[label]

    def __len__(self):
        return len(self.filtered_samples)


def main(num_epochs, path_mod):

    class_count = 100
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # crop to 224x224 with scale jittering
        transforms.RandomHorizontalFlip(),  # 50% chance
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),             # resize shorter side to 256
        transforms.CenterCrop(224),         # crop center 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])
    imagenet_train = datasets.ImageNet('/mimer/NOBACKUP/groups/alvis_cvl/datasets/ImageNet_2012', split='train',
                                transform=train_transform)
    # need to make a random split for validation
    # imagenet_val = datasets.ImageNet('/mimer/NOBACKUP/groups/alvis_cvl/datasets/ImageNet_2012', split='val',
    #                             transform=val_transform)
    
    all_classes = list(imagenet_train.classes)
    subset_classes = torch.randperm(len(all_classes))[:class_count].tolist()
    print("Subclasses:")
    for i in range(10):
        print(subset_classes[i:i+10])
    # Wrap both datasets
    train_dataset = SubsetImageNet(imagenet_train, subset_classes)
    # val_dataset = SubsetImageNet(imagenet_val, subset_classes)

    val_size = int(len(train_dataset)*.2)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=2, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, pin_memory=True, num_workers=1, prefetch_factor=2, persistent_workers=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = nn.DataParallel(BaseModel(class_count, device = device, backbone='poop').to(device))
    mid_model = nn.DataParallel(MidModel(class_count, device = device, backbone='poop').to(device))
    tournament_model = nn.DataParallel(TournamentModel(class_count, device = device, backbone='poop').to(device))
    optimizer_base = torch.optim.SGD(base_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer_mid = torch.optim.SGD(mid_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer_tournament = torch.optim.SGD(tournament_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sched_base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, num_epochs)
    sched_mid = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mid, num_epochs)
    sched_tournament = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tournament, num_epochs)
    # num_epochs = 10

    models = {
        'base': (base_model, None, optimizer_base, sched_base),
        'mid': (mid_model, None, optimizer_mid, sched_mid),
        'tournament': (tournament_model, None, optimizer_tournament, sched_tournament)
    }

    # prepare convergence monitor and ckpt directory
    _path_mod = '/default' if path_mod == '' else f'/{path_mod}'

    ckpt_base = f'ckpts/imagenet/mobilenet{_path_mod}'
    monitor = ConvergenceMonitor(patience=3, mode='max', save_dir=ckpt_base)

    print("Starting joint training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        joint_train_all_ising(device, train_loader, models, class_count, temps = [1,1,1], lbda = [0,1,1,0], epoch=epoch/num_epochs)
        joint_eval_all(device, val_loader, models, class_count, monitor=monitor, epoch=epoch)
    print("Done")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--path_mod", type=str, default="")
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()
    fix_random_seeds(args.seed)
    main(args.epochs, args.path_mod)

