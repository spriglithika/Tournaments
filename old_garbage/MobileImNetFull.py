from preamble import *
import Tournament
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from old_garbage.Models import BaseModel, MidModel, TournamentModel, copy_matching_parameters, NeuralIsingTournamentSparse
from Training_testing_old import train_tourn_ising, eval_tourn, ConvergenceMonitor
from argparse import ArgumentParser
from PIL import Image
from torch.utils.data import Dataset
from TournamentGroundTruth import get_gt_sparse
TournamentModel = NeuralIsingTournamentSparse
sce = Tournament.symmetric_cross_entropy
Tournament = Tournament.Tournament


import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

def setup_ddp():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


print("MobileNet ImageNet1000: Modules loaded")

def main(num_epochs, path_mod):
    local_rank = setup_ddp()

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
    imagenet_train = datasets.ImageNet('/mimer/NOBACKUP/groups/alvis_cvl/datasets/ImageNet_2012', split='train', transform=train_transform)
    # imagenet_train = datasets.ImageNet('/dev/shm', split='train', transform=train_transform)

    imagenet_val = datasets.ImageNet('/mimer/NOBACKUP/groups/alvis_cvl/datasets/ImageNet_2012', split='val', transform=val_transform)
    # imagenet_val = datasets.ImageNet('/dev/shm', split='val', transform=val_transform)

    class_count = len(imagenet_train.classes)
    train_sampler = DistributedSampler(imagenet_train)
    # train_loader = DataLoader(imagenet_train, batch_size=320, shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=32, persistent_workers=True)
    train_loader = DataLoader(imagenet_train, batch_size=400, sampler=train_sampler, pin_memory=True, num_workers=4, prefetch_factor=16, persistent_workers=True)
    val_loader = DataLoader(imagenet_val, batch_size=1000, shuffle=False, pin_memory=True, num_workers=4, prefetch_factor=16, persistent_workers=True)
    print(class_count, len(imagenet_train), sorted(imagenet_train.class_to_idx.keys())[:10])  # Sample synsets)

    device = torch.device(f"cuda:{local_rank}")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # tournament_model = TournamentModel(class_count, device = device, backbone='poop').to(device)
    tournament_model = nn.parallel.DistributedDataParallel(TournamentModel(class_count, device = device, backbone='poop').to(device), device_ids=[local_rank])
    optimizer_tournament = torch.optim.SGD(tournament_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    sched_tournament = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tournament, 200)

    models = (tournament_model, None, optimizer_tournament, sched_tournament)
    gt_stuff = get_gt_sparse()
    # prepare convergence monitor and ckpt directory
    _path_mod = '/default' if path_mod == '' else f'/{path_mod}'

    ckpt_base = f'ckpts/ImNet1000/mobilenet{_path_mod}'
    monitor = ConvergenceMonitor(patience=3, mode='max', save_dir=ckpt_base)

    print("Starting joint training...")
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        train_tourn_ising(device, train_loader, models, class_count, gt_stuff, temps = [1,1,1], lbda = [0,1,1,0], epoch=epoch/num_epochs, verbose=True)
        eval_tourn(device, val_loader, models, class_count, monitor=monitor, epoch=epoch)
        train_sampler.set_epoch(epoch)

    print("Done")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--path_mod", type=str, default="")
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()
    fix_random_seeds(args.seed)
    main(args.epochs, args.path_mod)

