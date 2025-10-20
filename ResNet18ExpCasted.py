from preamble import *
import Tournament
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from Models import BaseModel, MidModel, TournamentModel
from Training_testing import joint_train_all, joint_eval_all, ConvergenceMonitor
from argparse import ArgumentParser

sce = Tournament.symmetric_cross_entropy
Tournament = Tournament.Tournament
nn = torch.nn
F = nn.functional

print("ResNet18ExpCasted: Modules loaded")

def main(num_epochs, path_mod):
    train_dataset = datasets.CIFAR100('../data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ]))
    # need to make a random split for validation
    val_size = 5000
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    test_dataset = datasets.CIFAR100('../data', train=False,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                ]))

    # Use pinned memory to allow async host->device transfers
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory=True)
    class_count = torch.max(torch.tensor(test_dataset.targets)) + 1
    # image_shape = train_dataset[0][0].shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = BaseModel(class_count, device = device, backbone='resnet18').to(device)
    mid_model = MidModel(class_count, device = device, backbone='resnet18').to(device)
    tournament_model = TournamentModel(class_count, device = device, backbone='resnet18').to(device)
    optimizer_base = torch.optim.AdamW(base_model.parameters(), lr=0.001)
    optimizer_mid = torch.optim.AdamW(mid_model.parameters(), lr=0.001)
    optimizer_tournament = torch.optim.AdamW(tournament_model.parameters(), lr=0.001)
    # optimizer_base = torch.optim.SGD(base_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # optimizer_mid = torch.optim.SGD(mid_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    # optimizer_tournament = torch.optim.SGD(tournament_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # num_epochs = 10

    models = {
        'base': (base_model, None, optimizer_base),
        'mid': (mid_model, None, optimizer_mid),
        'tournament': (tournament_model, None, optimizer_tournament)
    }

    # prepare convergence monitor and ckpt directory
    _path_mod = '/default' if path_mod == '' else f'/{path_mod}'

    ckpt_base = f'ckpts/cifar100/resnet18{_path_mod}'
    monitor = ConvergenceMonitor(patience=3, mode='max', save_dir=ckpt_base)

    print("Starting joint training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        joint_train_all(device, train_loader, models, class_count)
        joint_eval_all(device, val_loader, models, class_count, monitor=monitor, epoch=epoch)
    # _path_mod = '' if path_mod == '' else f'_{path_mod}'
    # Loop to save the models after training
    # torch.save(base_model.state_dict(), f'ckpts/cifar100/resnet18/base_model_{num_epochs}{_path_mod}.pth')
    # torch.save(mid_model.state_dict(), f'ckpts/cifar100/resnet18/mid_model_{num_epochs}{_path_mod}.pth')
    # torch.save(tournament_model.state_dict(), f'ckpts/cifar100/resnet18/tournament_model_{num_epochs}{_path_mod}.pth')

    print("Done")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--path_mod", type=str, default="")

    args = parser.parse_args()
    main(args.epochs, args.path_mod)
