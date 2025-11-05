from preamble import *
import Tournament
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from Models import BaseModel, MidModel, copy_matching_parameters, NeuralIsingTournament
from Training_testing import joint_train_all, joint_eval_all, ConvergenceMonitor
from argparse import ArgumentParser
TournamentModel = NeuralIsingTournament
sce = Tournament.symmetric_cross_entropy
Tournament = Tournament.Tournament


def main(checkpoint_path):
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
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1000, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, pin_memory=True)
    class_count = (torch.max(torch.tensor(test_dataset.targets)) + 1).item()
    # image_shape = train_dataset[0][0].shape

    checkpoints = [None, None, None]
    for path, dirs, files in os.walk(checkpoint_path):
        for file in files:
            terms = file.split('_')
            if terms[0] == 'base':
                checkpoints[0] = path + '/' + file
            if terms[0] == 'mid':
                checkpoints[1] = path + '/' + file
            if terms[0] == 'tournament':
                checkpoints[2] = path + '/' + file
    if None in checkpoints:
        print('Invalid Path: Missing Checkpoints')
        exit()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = BaseModel(class_count, device = device, backbone='resnet18').to(device)
    base_checkpoint = torch.load(checkpoints[0], map_location=device)
    base_model.load_state_dict(base_checkpoint)
    mid_model = MidModel(class_count, device = device, backbone='resnet18').to(device)
    mid_checkpoint = torch.load(checkpoints[1], map_location=device)
    mid_model.load_state_dict(mid_checkpoint)
    # copy_matching_parameters(base_model, mid_model)
    tournament_model = TournamentModel(class_count, device = device, backbone='resnet18').to(device)
    tourn_checkpoint = torch.load(checkpoints[2], map_location=device)
    tournament_model.load_state_dict(tourn_checkpoint)
    # copy_matching_parameters(base_model, tournament_model)
    # optimizer_base = torch.optim.AdamW(base_model.parameters(), lr=0.01)
    # optimizer_mid = torch.optim.AdamW(mid_model.parameters(), lr=0.01)
    # optimizer_tournament = torch.optim.AdamW(tournament_model.parameters(), lr=0.01)
    # optimizer_base = torch.optim.SGD(base_model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    # optimizer_mid = torch.optim.SGD(mid_model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    # optimizer_tournament = torch.optim.SGD(tournament_model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    # sched_base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_base, 200)
    # sched_mid = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mid, 200)
    # sched_tournament = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tournament, 200)
    # num_epochs = 10

    # models = {
    #     'base': (base_model, None, optimizer_base, sched_base),
    #     'mid': (mid_model, None, optimizer_mid, sched_mid),
    #     'tournament': (tournament_model, None, optimizer_tournament, sched_tournament)
    # }
    models = {
        'base': (base_model, None, None, None),
        'mid': (mid_model, None, None, None),
        'tournament': (tournament_model, None, None, None)
    }

    # prepare convergence monitor and ckpt directory
    # _path_mod = '/default' if path_mod == '' else f'/{path_mod}'

    # ckpt_base = f'ckpts/cifar100/resnet18{_path_mod}'
    # monitor = ConvergenceMonitor(patience=3, mode='max', save_dir=ckpt_base)

    # print("Starting joint training...")
    # for epoch in range(num_epochs):
    #     print(f"Epoch {epoch}")
    #     # joint_train_all(device, train_loader, models, class_count, temps = [1,1, 10], lbda = 1.0)
    joint_eval_all(device, test_loader, models, class_count, monitor=None, epoch=None)
    # _path_mod = '' if path_mod == '' else f'_{path_mod}'
    # Loop to save the models after training
    # torch.save(base_model.state_dict(), f'ckpts/cifar100/resnet18/base_model_{num_epochs}{_path_mod}.pth')
    # torch.save(mid_model.state_dict(), f'ckpts/cifar100/resnet18/mid_model_{num_epochs}{_path_mod}.pth')
    # torch.save(tournament_model.state_dict(), f'ckpts/cifar100/resnet18/tournament_model_{num_epochs}{_path_mod}.pth')

    print("Done")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default='/mimer/NOBACKUP/groups/alvis_cvl/hannahhe/Tournaments/ckpts/cifar100/resnet18/dropout_0.2_lambda_1')
    parser.add_argument("--seed", type=int, default=69)
    args = parser.parse_args()
    fix_random_seeds(args.seed)
    main(args.checkpoint_path)