from preamble import *
from argparse import ArgumentParser
from Utils.TrainingFuncs import train, eval, ConvergenceMonitor, eval_ece
from Data import add_label_noise, get_data_loader
from Utils.config_reader import load_config
from Models.BuildModel import build_model
from Utils.Saving import SaveModule, notify, plot_calibration_curve




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json', help='Path to the configuration file')
    parser.add_argument('--sanity_check', default=False, action='store_true', help='Run sanity checks only')
    parser.add_argument('--just_test', default=False, action='store_true', help='Only run testing using the best saved model')
    parser.add_argument('--compute_ece', default=False, action='store_true', help='Compute ECE on test set after training')
    parser.add_argument('--ignore_main_eval', default=False, action='store_true', help='Skip evaluation steps during training')
    parser.add_argument('--lightning', default=False, action='store_true', help='Use PyTorch Lightning training loop')
    parser.add_argument('--wandb', default=False, action='store_true', help='Enable Weights & Biases logging (flag)')
    parser.add_argument('--inspect_models', default=False, action='store_true', help='Instantiate all available models and print parameter counts')
    parser.add_argument('--model_summary', default=False, action='store_true', help='Print parameter summary for the model in the config and exit')
    args = parser.parse_args()
    if args.sanity_check and args.just_test:
        raise ValueError("Cannot use --sanity_check and --just_test together.")
    # Load configuration
    cfg = load_config(args.config)
    save_dir = args.config.replace('.json', '').replace('configs', 'outputs')
    cfg.deep_update({'save_dir': save_dir})
    # Set device
    fix_random_seeds(cfg.get('seed', 42))

    # Optional: initialize wandb early if requested or if API key present
    if args.wandb:
        try:
            init_wandb()
        except Exception:
            print('Warning: wandb initialization failed or wandb not installed; continuing without W&B')
    # Get data loaders
    if not args.just_test:
        train_loader = get_data_loader(train=True,
                                    batch_size=cfg.get('train.batch_size', 32),
                                    dataset=cfg.get('dataset', 'mnist'),
                                    num_classes=cfg.get('num_classes', 10),
                                    class_list=cfg.get('class_list', None),
                                    samples_per_class=cfg.get('samples_per_class', 10),
                                    resize=cfg.get('resize', 28),
                                    imbalance=cfg.get('imbalance', None),
                                    imbalance_factor=cfg.get('imbalance_factor', 1.0))
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

    # Initialize model
    model = build_model(cfg)

    # quick model-inspection CLI hooks
    if args.model_summary:
        from Utils.QuickTests import model_summary
        model_summary(model)
        exit()

    if args.inspect_models:
        from Utils.QuickTests import model_summary
        from Models.BuildModel import MODEL_DICT
        print('Inspecting all available model classes:')
        for name in sorted(MODEL_DICT.keys()):
            try:
                cfg.deep_update({'model': {'name': name}})
                m = build_model(cfg)
                print(f"\n{name}:")
                model_summary(m)
            except Exception as e:
                print(f"Failed to instantiate {name}: {e}")
        exit()

    monitor = ConvergenceMonitor(patience=cfg.get('val.patience', 5), mode=cfg.get('val.mode', 'max'), save_dir=save_dir, filename='best.pth')

    # number of epochs (used by both legacy and Lightning branches)
    num_epochs = cfg.get('train.num_epochs', 10)
    # Ensure the SaveModule exists for downstream evaluation/saving
    save_module = SaveModule(model, save_dir)

    # --- PyTorch Lightning branch (opt-in) ---
    if args.lightning:
        try:
            import pytorch_lightning as pl
        except Exception as e:
            raise RuntimeError("PyTorch Lightning is not installed. Install it (pip install pytorch-lightning) or run without --lightning") from e
        from Utils.lightning_adapter import PLWrapper, TournamentsDataModule, SaveBestPthCallback

        # DataModule + LightningModule
        dm = TournamentsDataModule(cfg)
        pl_model = PLWrapper(cfg)

        # Logger (WandB optional)
        logger = None
        if args.wandb:
            try:
                from pytorch_lightning.loggers import WandbLogger
                # ensure wandb is imported / logged-in (preamble helper)
                try:
                    init_wandb()
                except Exception:
                    pass
                logger = WandbLogger(project=cfg.get('wandb.project', 'tournaments'), name=cfg.get('wandb.run_name', args.config.replace('.json','')), save_dir=cfg.get('save_dir'))
            except Exception:
                print('Warning: wandb/WandbLogger not available; continuing without WandbLogger')
                logger = None
        else:
            from pytorch_lightning.loggers import CSVLogger
            logger = CSVLogger(save_dir=cfg.get('save_dir'))

        callbacks = [SaveBestPthCallback(save_dir=cfg.get('save_dir', save_dir), monitor='val/acc', mode='max')]
        if cfg.get('val.patience', 0) > 0:
            callbacks.append(pl.callbacks.EarlyStopping(monitor='val/acc', patience=cfg.get('val.patience', 5), mode='max'))

        trainer_kwargs = dict(max_epochs=num_epochs, logger=logger, callbacks=callbacks, default_root_dir=cfg.get('save_dir'))
        # optional precision (set in config as `train.precision`, e.g. 16)
        if cfg.get('train.precision', None) is not None:
            trainer_kwargs['precision'] = cfg.get('train.precision')
        # device selection: prefer CUDA, else MPS (if available), else CPU
        if torch.cuda.is_available():
            trainer_kwargs.update(accelerator='gpu', devices=cfg.get('gpus', 1))
        elif torch.backends.mps.is_available():
            trainer_kwargs.update(accelerator='mps', devices=1)
        else:
            trainer_kwargs.update(accelerator='cpu', devices=1)

        trainer = pl.Trainer(**trainer_kwargs)

        # If user asked to *only* run test with Lightning, run test on the best checkpoint and exit
        if args.just_test:
            dm.setup(stage='test')
            try:
                res = trainer.test(pl_model, datamodule=dm, ckpt_path='best')
                print('Lightning test results:', res)
            except Exception as e:
                print(f'Lightning test failed: {e}')
            exit()

        if args.sanity_check:
            trainer.fit(pl_model, datamodule=dm, limit_train_batches=1, limit_val_batches=1, max_epochs=1)
            print('Sanity check completed (Lightning).')
            exit()

        trainer.fit(pl_model, datamodule=dm)

        # run Lightning test with the best checkpoint (convenience; legacy eval still runs later)
        if not args.ignore_main_eval:
            try:
                trainer.test(pl_model, datamodule=dm, ckpt_path='best')
            except Exception as e:
                print(f'Warning: Lightning test failed or no best checkpoint found: {e}')

        # compatible variables used later in the script
        train_loss_history = []
        epoch = num_epochs

    # --- Legacy training (original loop) ---
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('train.lr', 1e-3), weight_decay=cfg.get('train.weight_decay', 0.0))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=cfg.get('train.lr_min', 1e-5)) if cfg.get('train.use_scheduler', False) else None

    if args.sanity_check:
        if args.lightning:
            print("Sanity check already performed in Lightning branch; exiting.")
            exit()
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
    epoch = 0
    train_loss_history = []
    val_ece_history = []
    if not args.just_test and not args.lightning:
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc, train_conf_mat, loss_history = train(model, device, train_loader, optimizer, epoch, forward_kwargs=cfg.get('forward_kwargs', {}))
            train_loss_history.extend(loss_history)
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
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
                        if wb.run is None:
                            wb.init(project=cfg.get('wandb.project','tournaments'), name=cfg.get('wandb.run_name', args.config.replace('.json','')), config=dict(cfg))
                        wb.log({'epoch': epoch, 'train/loss': train_loss, 'train/acc': train_acc})
                    except Exception:
                        print('Warning: wandb logging failed for train step')

            if val_split > 0:
                val_loss, val_acc, val_conf_mat = eval(model, device, val_loader, forward_kwargs=cfg.get('forward_kwargs', {}))
                ece, accs, confs, counts = eval_ece(model, val_loader, 20)
                val_ece_history.append(ece)
                print(f'Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f} Val ECE: {ece:.4f}')

                # save calibration arrays + image locally
                try:
                    calib_name = save_module.save_calibration(confs, accs, ece, epoch=epoch)
                except Exception:
                    calib_name = None

                # wandb logging + artifact (if enabled)
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
                            if wb.run is None:
                                wb.init(project=cfg.get('wandb.project','tournaments'), name=cfg.get('wandb.run_name', args.config.replace('.json','')), config=dict(cfg))
                            wb.log({'epoch': epoch, 'val/loss': val_loss, 'val/acc': val_acc, 'val/ece': ece})

                            art = wb.Artifact(f'calibration-epoch-{epoch}', type='calibration')
                            if calib_name is not None:
                                art.add_file(os.path.join(save_module.out_dir, calib_name))

                            # attach confusion matrix
                            cm_name = save_module.save_confusion_matrix(val_conf_mat, mode='val')
                            if cm_name is not None:
                                art.add_file(os.path.join(save_module.out_dir, cm_name))

                            # attach J file if present
                            if cfg.get('save_J_heatmap', False):
                                try:
                                    jname = None
                                    try:
                                        if hasattr(model, 'J'):
                                            Jmat = model.J() if callable(model.J) else model.J
                                            if torch.is_tensor(Jmat):
                                                jnp = Jmat.detach().cpu().numpy()
                                            else:
                                                jnp = np.array(Jmat)
                                            jname = save_module.save_J(jnp, mod=f'val_{epoch}')
                                    except Exception:
                                        jname = None
                                    if jname is not None:
                                        art.add_file(os.path.join(save_module.out_dir, jname))
                                except Exception:
                                    pass

                            wb.log_artifact(art)
                        except Exception:
                            print('Warning: wandb logging/artifact failed for val step')
                else:
                    # ensure local copies exist when not using wandb
                    save_module.save_confusion_matrix(val_conf_mat, mode='val')
                if monitor.update(val_acc, epoch, model):
                    print(f'New best model saved at epoch {epoch} with Val Acc: {val_acc:.2f}%')
                if monitor.converged:
                    print(f'Converged at epoch {epoch}. Stopping training.')
                    break
                save_module.save_confusion_matrix(val_conf_mat, mode='val')
                save_module.save_loss_history(train_loss_history)
                if cfg.get('save_J_heatmap', False):
                    # save_module.save_J(model.J.detach().cpu().numpy(), mod =f'val_{epoch}')
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
            if scheduler is not None:
                scheduler.step()

    if not args.ignore_main_eval:
        test_loss, test_acc, test_conf_mat = eval(model, device, test_loader, forward_kwargs=cfg.get('forward_kwargs', {}))
        print(f'Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        model = build_model(cfg) # re-initialize model using best saved weights
        test_loss, test_acc, test_conf_mat = eval(model, device, test_loader, forward_kwargs=cfg.get('forward_kwargs', {}))

        # save confusion, loss history, calibration for test set
        save_module.save_confusion_matrix(test_conf_mat, mode='test')
        save_module.save_loss_history(train_loss_history)
        try:
            ece, accs, confs, counts = eval_ece(model, test_loader, 20)
            save_module.save_calibration(confs, accs, ece, epoch='test')
        except Exception:
            ece = None

        if cfg.get('save_J_heatmap', False):
            try:
                jname = None
                try:
                    if hasattr(model, 'J'):
                        Jmat = model.J() if callable(model.J) else model.J
                        if torch.is_tensor(Jmat):
                            jnp = Jmat.detach().cpu().numpy()
                        else:
                            jnp = np.array(Jmat)
                        jname = save_module.save_J(jnp, mod='test')
                except Exception:
                    jname = None
            except Exception:
                jname = None

        # optionally log artifacts to wandb
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
                    art = wb.Artifact('final-results', type='experiment')
                    cm_name = os.path.join(save_module.out_dir, f'confusion_matrix_test.npy')
                    if os.path.exists(cm_name):
                        art.add_file(cm_name)
                    loss_name = os.path.join(save_module.out_dir, 'loss_history.npy')
                    if os.path.exists(loss_name):
                        art.add_file(loss_name)
                    calib_name = os.path.join(save_module.out_dir, 'calibration_epoch_test.npz')
                    if os.path.exists(calib_name):
                        art.add_file(calib_name)
                    if cfg.get('save_J_heatmap', False) and jname is not None:
                        jpath = os.path.join(save_module.out_dir, jname)
                        if os.path.exists(jpath):
                            art.add_file(jpath)
                    wb.log_artifact(art)
                except Exception:
                    print('Warning: wandb artifact upload failed for final results')

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
        notify("Experiment Complete", f"Training and testing for {args.config.split('/')[-1]} complete. Final Test Acc: {test_acc:.2f}%")

    if args.compute_ece:
        ece, accs, confs,counts = eval_ece(model, test_loader, 20)
        print(f'Test ECE: {ece:.4f}%')
        plot_calibration_curve(confs, accs, ece, savepath=save_dir)
