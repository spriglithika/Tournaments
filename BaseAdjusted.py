from preamble import *
from argparse import ArgumentParser
from Utils.TrainingFuncs import ddn_extended_base_train, ddn_extended_base_eval, ConvergenceMonitor, ddn_extended_eval_ece
from Data import add_label_noise, get_data_loader
from Utils.config_reader import load_config
from Models.BuildModel import build_model
from Utils.Saving import SaveModule, notify, plot_calibration_curve




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--base_config', type=str, default='experiments/configs/CifarBaseModel.json', help='Path to the configuration file')
    parser.add_argument('--ddn_config', type=str, default='experiments/configs/CIFARDec.json', help='Path to the configuration file')
    parser.add_argument('--sanity_check', default=False, action='store_true', help='Run sanity checks only')
    parser.add_argument('--just_test', default=False, action='store_true', help='Only run testing using the best saved model')
    parser.add_argument('--compute_ece', default=False, action='store_true', help='Compute ECE on test set after training')
    parser.add_argument('--ignore_main_eval', default=False, action='store_true', help='Skip evaluation steps during training')
    args = parser.parse_args()
    if args.sanity_check and args.just_test:
        raise ValueError("Cannot use --sanity_check and --just_test together.")
    # Load configuration
    cfg = load_config(args.base_config)
    ddn = load_config(args.ddn_config)
    save_dir = args.base_config.replace('.json', '').replace('configs', 'outputs_ddn')
    ddn_path = args.ddn_config.replace('.json', '').replace('configs', 'outputs')
    cfg.deep_update({'save_dir': save_dir})
    # Set device
    fix_random_seeds(cfg.get('seed', 42))
    # Get data loaders
    if not args.just_test:
        train_loader = get_data_loader(train=True,
                                    batch_size=cfg.get('train.batch_size', 32),
                                    dataset=cfg.get('dataset', 'mnist'),
                                    num_classes=cfg.get('num_classes', 10),
                                    class_list=cfg.get('class_list', None),
                                    samples_per_class=cfg.get('samples_per_class', 10),
                                    resize=cfg.get('resize', 28),
                                    imbalance=cfg.get('imbalance', None),)
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
        if cfg.get('label_noise', 0.0) > 0.0:
            # Re-apply label noise to entire training set
            train_loader = add_label_noise(train_loader,  num_classes=cfg.get('num_classes', 10), noise_rate=cfg.get('label_noise', 0.0))

    test_loader = get_data_loader(train=False,
                                  batch_size=cfg.get('test.batch_size', None),
                                  dataset=cfg.get('dataset', 'mnist'),
                                  num_classes=cfg.get('num_classes', 10),
                                  class_list=cfg.get('class_list', None),
                                  samples_per_class=cfg.get('samples_per_class', 10),
                                  resize=cfg.get('resize', 28))

    def signed_laplacian(C):
        Wp = C.clamp(min=0)
        Wn = (-C).clamp(min=0)
        Dp = torch.diag(Wp.sum(dim=1))
        Dn = torch.diag(Wn.sum(dim=1))
        return Dp + Dn - (Wp - Wn)

    # Initialize model
    model = build_model(cfg)
    class_matrix = torch.from_numpy(np.load(os.path.join(ddn_path, 'class_matrix.npy'))).float().to(device)
    # class_matrix = F.softmax(class_matrix, dim=0)
    class_matrix = signed_laplacian(class_matrix)

    # optionally artifact the ddn class_matrix to wandb
    if args.wandb:
        wb = get_wandb()
        if wb is None:
            try:
                init_wandb()
                wb = get_wandb()
            except Exception:
                wb = None
        if wb is not None:
            try:
                cm_path = os.path.join(ddn_path, 'class_matrix.npy')
                if os.path.exists(cm_path):
                    art = wb.Artifact('ddn_class_matrix', type='dataset')
                    art.add_file(cm_path)
                    wb.log_artifact(art)
            except Exception:
                print('Warning: wandb artifact upload failed for ddn class_matrix')

    monitor = ConvergenceMonitor(patience=cfg.get('val.patience', 5), mode=cfg.get('val.mode', 'max'), save_dir=cfg.get('save_dir', args.base_config.replace('.json', '').replace('configs', 'outputs_ddn')), filename='best.pth')
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('train.lr', 1e-3))


    # Training loop
    num_epochs = cfg.get('train.num_epochs', 10)
    save_module = SaveModule(model, cfg.get('save_dir', args.base_config.replace('.json', '').replace('configs', 'outputs_ddn')))

    if args.sanity_check:
        print("Sanity check mode: Running a single training and evaluation step.")
        train_loss, train_acc, train_conf_mat, train_loss_history = ddn_extended_base_train(model, class_matrix, device, train_loader, optimizer, epoch=1)
        print(f'Sanity Check - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        test_loss, test_acc, test_conf_mat = ddn_extended_base_eval(model, class_matrix, device, test_loader)
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
    epoch = 0
    train_loss_history = []
    if not args.just_test:
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc, train_conf_mat, loss_history = ddn_extended_base_train(model, class_matrix, device, train_loader, optimizer, epoch, forward_kwargs=cfg.get('forward_kwargs', {}))
            train_loss_history.extend(loss_history)
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

            if val_split > 0:
                val_loss, val_acc, val_conf_mat = ddn_extended_base_eval(model, class_matrix, device, val_loader, forward_kwargs=cfg.get('forward_kwargs', {}))
                ece, accs, confs, counts = ddn_extended_eval_ece(model, class_matrix, val_loader, 20)
                print(f'Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f} Val ECE: {ece:.4f}')

                # save calibration arrays + image locally
                try:
                    calib_name = save_module.save_calibration(confs, accs, ece, epoch=epoch)
                except Exception:
                    calib_name = None

                # optional wandb artifacting
                if args.wandb:
                    wb = get_wandb()
                    if wb is None:
                        try:
                            init_wandb()
                            wb = get_wandb()
                        except Exception:
                            wb = None
                    if wb is not None:
                        try:
                            art = wb.Artifact(f'ddn-val-epoch-{epoch}', type='ddn')
                            # class matrix from ddn outputs
                            cm_path = os.path.join(ddn_path, 'class_matrix.npy')
                            if os.path.exists(cm_path):
                                art.add_file(cm_path)
                            if calib_name is not None:
                                art.add_file(os.path.join(save_module.out_dir, calib_name))
                            cm_name = save_module.save_confusion_matrix(val_conf_mat, mode='val')
                            if cm_name is not None:
                                art.add_file(os.path.join(save_module.out_dir, cm_name))
                            if cfg.get('save_J_heatmap', False):
                                try:
                                        jname = None
                                        if hasattr(model, 'J'):
                                            Jmat = model.J() if callable(model.J) else model.J
                                            if torch.is_tensor(Jmat):
                                                jnp = Jmat.detach().cpu().numpy()
                                            else:
                                                jnp = np.array(Jmat)
                                            jname = save_module.save_J(jnp, mod=f'val_{epoch}')
                                        if jname is not None:
                                            art.add_file(os.path.join(save_module.out_dir, jname))
                                except Exception:
                                        pass
                            wb.log_artifact(art)
                        except Exception:
                            print('Warning: wandb artifact failed for ddn val')
                else:
                    save_module.save_confusion_matrix(val_conf_mat, mode='val')

                save_module.save_loss_history(train_loss_history)
                if cfg.get('save_J_heatmap', False) and not args.wandb:
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

    if not args.ignore_main_eval:
        test_loss, test_acc, test_conf_mat = ddn_extended_base_eval(model, class_matrix, device, test_loader, forward_kwargs=cfg.get('forward_kwargs', {}))
        print(f'Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        model = build_model(cfg) # re-initialize model using best saved weights
        test_loss, test_acc, test_conf_mat = ddn_extended_base_eval(model, class_matrix, device, test_loader, forward_kwargs=cfg.get('forward_kwargs', {}))

        save_module.save_confusion_matrix(test_conf_mat, mode='test')
        save_module.save_loss_history(train_loss_history)
        try:
            ece, accs, confs, counts = ddn_extended_eval_ece(model, class_matrix, test_loader, 20)
            save_module.save_calibration(confs, accs, ece, epoch='test')
        except Exception:
            ece = None

        if cfg.get('save_J_heatmap', False):
            try:
                jname = None
                if hasattr(model, 'J'):
                    Jmat = model.J() if callable(model.J) else model.J
                    if torch.is_tensor(Jmat):
                        jnp = Jmat.detach().cpu().numpy()
                    else:
                        jnp = np.array(Jmat)
                    jname = save_module.save_J(jnp, mod='test')
            except Exception:
                jname = None

        # optional wandb artifacts
        if args.wandb:
            wb = get_wandb()
            if wb is None:
                try:
                    init_wandb()
                    wb = get_wandb()
                except Exception:
                    wb = None
            if wb is not None:
                try:
                    art = wb.Artifact('ddn-final-results', type='experiment')
                    cm_name = os.path.join(save_module.out_dir, f'confusion_matrix_test.npy')
                    if os.path.exists(cm_name):
                        art.add_file(cm_name)
                    calib_name = os.path.join(save_module.out_dir, 'calibration_epoch_test.npz')
                    if os.path.exists(calib_name):
                        art.add_file(calib_name)
                    cm_path = os.path.join(ddn_path, 'class_matrix.npy')
                    if os.path.exists(cm_path):
                        art.add_file(cm_path)
                    if cfg.get('save_J_heatmap', False) and jname is not None:
                        jpath = os.path.join(save_module.out_dir, jname)
                        if os.path.exists(jpath):
                            art.add_file(jpath)
                    wb.log_artifact(art)
                except Exception:
                    print('Warning: wandb artifact upload failed for ddn final results')

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
        notify("Experiment Complete", f"Training and testing for {args.base_config.split('/')[-1]} DDN extension complete. Final Test Acc: {test_acc:.2f}%")

    if args.compute_ece:
        ece, accs, confs,counts = ddn_extended_eval_ece(model, class_matrix, test_loader, 20)
        print(f'Test ECE: {ece:.4f}%')
        plot_calibration_curve(confs, accs, ece, savepath=save_dir)
