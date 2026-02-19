from preamble import *
from argparse import ArgumentParser
from Utils.TrainingFuncs import train, eval, ConvergenceMonitor
from Data import get_data_loader
from Utils.config_reader import load_config
from Models.BuildModel import build_model
from Utils.Saving import SaveModule




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--sanity_check', default=False, action='store_true', help='Run sanity checks only')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Set device
    fix_random_seeds(cfg.get('seed', 42))
    # Get data loaders
    train_loader = get_data_loader(train=True,
                                   batch_size=cfg.get('train.batch_size', 32),
                                   dataset=cfg.get('dataset', 'mnist'),
                                   num_classes=cfg.get('num_classes', 10),
                                   class_list=cfg.get('class_list', None),
                                   samples_per_class=cfg.get('samples_per_class', 10),
                                   resize=cfg.get('resize', 28))

    test_loader = get_data_loader(train=False,
                                  batch_size=cfg.get('test.batch_size', None),
                                  dataset=cfg.get('dataset', 'mnist'),
                                  num_classes=cfg.get('num_classes', 10),
                                  class_list=cfg.get('class_list', None),
                                  samples_per_class=cfg.get('samples_per_class', 10),
                                  resize=cfg.get('resize', 28))
    val_split = cfg.get('val.split', 0.1)
    if val_split > 0:
        total_size = len(train_loader.dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_loader.dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.get('train.batch_size', 32), shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.get('train.batch_size', 32), shuffle=False)
    else:
        val_loader = None

    # Initialize model
    model = build_model(cfg)
    monitor = ConvergenceMonitor(patience=cfg.get('val.patience', 5), mode=cfg.get('val.mode', 'max'), save_dir=cfg.get('save_dir', args.config.replace('.json', '').replace('configs', 'outputs')), filename='best.pth')
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('train.lr', 1e-3))

    # Training loop
    num_epochs = cfg.get('train.num_epochs', 10)
    save_module = SaveModule(model, cfg.get('save_dir', args.config.replace('.json', '').replace('configs', 'outputs')))

    if args.sanity_check:
        print("Sanity check mode: Running a single training and evaluation step.")
        train_loss, train_acc, train_conf_mat, train_loss_history = train(model, device, train_loader, optimizer, epoch=1)
        print(f'Sanity Check - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        test_loss, test_acc, test_conf_mat = eval(model, device, test_loader)
        print(f'Sanity Check - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        save_module.save_confusion_matrix(test_conf_mat, mode='test')
        save_module.save_loss_history(train_loss_history)
        if cfg.get('save_J_heatmap', False):
            try:
                if hasattr(model, 'J'):
                    Jmat = model.J() if callable(model.J) else model.J
                    if torch.is_tensor(Jmat):
                        jnp = Jmat.detach().cpu().numpy()
                    else:
                        jnp = np.array(Jmat)
                    save_module.save_J(jnp)
            except Exception:
                pass
        exit()

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_conf_mat, train_loss_history = train(model, device, train_loader, optimizer, epoch)
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

        if val_split > 0:
            val_loss, val_acc, val_conf_mat = eval(model, device, val_loader)
            print(f'Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            if monitor.update(val_acc, epoch, model):
                print(f'New best model saved at epoch {epoch} with Val Acc: {val_acc:.2f}%')
            if monitor.converged:
                print(f'Converged at epoch {epoch}. Stopping training.')
                break
            save_module.save_confusion_matrix(val_conf_mat, mode='val')
            save_module.save_loss_history(train_loss_history)
            if cfg.get('save_J_heatmap', False):
                try:
                    if hasattr(model, 'J'):
                        Jmat = model.J() if callable(model.J) else model.J
                        if torch.is_tensor(Jmat):
                            jnp = Jmat.detach().cpu().numpy()
                        else:
                            jnp = np.array(Jmat)
                        save_module.save_J(jnp, mod=f'val_{epoch}')
                except Exception:
                    pass
    test_loss, test_acc, test_conf_mat = eval(model, device, test_loader)
    print(f'Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    save_module.save_confusion_matrix(test_conf_mat, mode='test')
    save_module.save_loss_history(train_loss_history)
    if cfg.get('save_J_heatmap', False):
        try:
            if hasattr(model, 'J'):
                Jmat = model.J() if callable(model.J) else model.J
                if torch.is_tensor(Jmat):
                    jnp = Jmat.detach().cpu().numpy()
                else:
                    jnp = np.array(Jmat)
                save_module.save_J(jnp, mod='test')
        except Exception:
            pass