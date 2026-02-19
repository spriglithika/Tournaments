"""PyTorch Lightning adapter for the existing training code.

Provides:
- PLWrapper: a LightningModule that wraps the repo's `build_model(cfg)` model
  and re-uses its forward(loss, logits) signature.
- TournamentsDataModule: small LightningDataModule using existing
  `get_data_loader` helpers so the repo `configs` remain usable.
- SaveBestPthCallback: saves `best.pth` (same filename used by legacy flow)
  when validation metric improves so downstream code stays compatible.

This file intentionally keeps the Lightning wrappers lightweight so the
rest of the repo (evaluation, saving, config format) can be reused
without large changes.
"""
from __future__ import annotations

import os
from typing import Optional

import torch
from torch.utils.data import DataLoader

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - runtime import guard
    pl = None

from Models.BuildModel import build_model
from Data import get_data_loader
from preamble import get_wandb, init_wandb
from Utils.TrainingFuncs import eval_ece
from Utils.Saving import plot_calibration_curve
import numpy as np


class PLWrapper(pl.LightningModule if pl is not None else object):
    """LightningModule wrapper around the repo's model.

    Important: the wrapped model uses the repo's forward signature
    (loss, logits = model(x, y, train=...)), so training/validation steps
    call that forward and log the returned loss/logits.
    """
    def __init__(self, cfg):
        assert pl is not None, "pytorch_lightning is required for PLWrapper"
        super().__init__()
        self.cfg = cfg
        self.forward_kwargs = cfg.get('forward_kwargs', {})
        # build_model may place the model on device; Lightning will move it as needed
        self.model = build_model(cfg)
        # buffer for per-validation-step outputs (Lightning v2 removed validation_epoch_end(outputs))
        self._val_step_outputs = []

    def forward(self, x):
        # return logits for inference
        loss, logits = self.model(x, None, train=False, **self.forward_kwargs)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, logits = self.model(x, y, train=True, **self.forward_kwargs)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean() * 100.0
        # epoch-level logging (Lightning aggregates automatically)
        self.log('train/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, logits = self.model(x, y, train=False, **self.forward_kwargs)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean() * 100.0
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        # return logits/labels so we can compute calibration/ECE at epoch end if needed
        probs = torch.softmax(logits, dim=1)
        confs, pred_labels = probs.max(dim=1)

        # buffer outputs for v2-style on_validation_epoch_end processing
        try:
            self._val_step_outputs.append({'confs': confs.detach().cpu(), 'preds': pred_labels.detach().cpu(), 'labels': y.detach().cpu()})
        except Exception:
            # ensure attribute exists
            self._val_step_outputs = [{'confs': confs.detach().cpu(), 'preds': pred_labels.detach().cpu(), 'labels': y.detach().cpu()}]

        return {'val_loss': loss, 'val_acc': acc}
    def on_validation_start(self):
        # clear buffer at start of validation epoch
        self._val_step_outputs = []

    def on_validation_epoch_end(self):
        # prefer using datamodule/val_dataloader for ECE computation when available
        dm = getattr(self.trainer, 'datamodule', None)
        val_loader = None
        if dm is not None:
            try:
                val_loader = dm.val_dataloader()
            except Exception:
                val_loader = None

        per_bin_acc = per_bin_conf = counts = None
        ece = None

        if val_loader is None:
            # aggregate from buffered outputs
            if not hasattr(self, '_val_step_outputs') or len(self._val_step_outputs) == 0:
                return
            all_confs = torch.cat([o['confs'] for o in self._val_step_outputs])
            all_preds = torch.cat([o['preds'] for o in self._val_step_outputs])
            all_labels = torch.cat([o['labels'] for o in self._val_step_outputs])
            from Utils.TrainingFuncs import compute_ece_quantile
            try:
                ece, per_bin_acc, per_bin_conf, counts = compute_ece_quantile(all_confs, all_preds, all_labels, n_bins=self.cfg.get('val.n_bins', 20))
                self.log('val/ece', ece, on_epoch=True, prog_bar=True)
            except Exception:
                return
        else:
            try:
                ece, per_bin_acc, per_bin_conf, counts = eval_ece(self.model, val_loader, self.cfg.get('val.n_bins', 20))
                self.log('val/ece', ece, on_epoch=True, prog_bar=True)
            except Exception:
                # fallback to buffered aggregation
                if hasattr(self, '_val_step_outputs') and len(self._val_step_outputs) > 0:
                    all_confs = torch.cat([o['confs'] for o in self._val_step_outputs])
                    all_preds = torch.cat([o['preds'] for o in self._val_step_outputs])
                    all_labels = torch.cat([o['labels'] for o in self._val_step_outputs])
                    from Utils.TrainingFuncs import compute_ece_quantile
                    try:
                        ece, per_bin_acc, per_bin_conf, counts = compute_ece_quantile(all_confs, all_preds, all_labels, n_bins=self.cfg.get('val.n_bins', 20))
                        self.log('val/ece', ece, on_epoch=True, prog_bar=True)
                    except Exception:
                        return
                else:
                    return

        # save calibration arrays + image to save_dir
        save_dir = self.cfg.get('save_dir', None) or getattr(self.trainer, 'default_root_dir', None)
        epoch = int(self.current_epoch)
        try:
            np.savez(os.path.join(save_dir, f'calibration_epoch_{epoch}.npz'), confs=np.array(per_bin_conf), accs=np.array(per_bin_acc), ece=float(ece))
            plot_calibration_curve(per_bin_conf, per_bin_acc, ece, savepath=save_dir, fname=f'calibration_curve_epoch_{epoch}.png')
        except Exception:
            pass

        # artifact to wandb if available (via logger)
        try:
            logger = getattr(self.trainer, 'logger', None)
            wb = None
            if logger is not None and hasattr(logger, 'experiment'):
                wb = logger.experiment
            if wb is not None:
                art = wb.Artifact(f'calibration-epoch-{epoch}', type='calibration')
                npz = os.path.join(save_dir, f'calibration_epoch_{epoch}.npz')
                png = os.path.join(save_dir, f'calibration_curve_epoch_{epoch}.png')
                if os.path.exists(npz):
                    art.add_file(npz)
                if os.path.exists(png):
                    art.add_file(png)

                # If model exposes J(), save J and class-matrix and attach to artifact
                try:
                    if hasattr(self.model, 'J'):
                        Jmat = self.model.J() if callable(self.model.J) else self.model.J
                        if torch.is_tensor(Jmat):
                            jnp = Jmat.detach().cpu().numpy()
                        else:
                            jnp = np.array(Jmat)
                        jpath = os.path.join(save_dir, 'J.npy')
                        try:
                            np.save(jpath, jnp)
                            art.add_file(jpath)
                        except Exception:
                            pass

                        # compute class matrix using default label mapping and attach
                        try:
                            from Utils.JVisualization import compute_class_matrix, save_class_matrix
                            labels = [i % self.cfg.get('num_classes', 10) for i in range(jnp.shape[0])]
                            class_mat = compute_class_matrix(torch.from_numpy(jnp), labels, self.cfg.get('num_classes', 10))
                            save_class_matrix(class_mat, f"{self.cfg.get('save_dir','')}_epoch_{epoch}", out_dir=save_dir)
                            # attach any saved class matrix files (png/npy)
                            for fname in os.listdir(save_dir):
                                if fname.startswith('class_interaction_') and fname.endswith(('.npy', '.png')):
                                    art.add_file(os.path.join(save_dir, fname))
                        except Exception:
                            pass
                except Exception:
                    pass

                wb.log_artifact(art)
        except Exception:
            pass

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, logits = self.model(x, y, train=False, **self.forward_kwargs)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean() * 100.0
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log('test/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.get('train.lr', 1e-3), weight_decay=self.cfg.get('train.weight_decay', 0.0))
        if self.cfg.get('train.use_scheduler', False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.get('train.num_epochs', 10), eta_min=self.cfg.get('train.lr_min', 1e-5))
            return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}
        return optimizer

    def on_fit_start(self):
        # Log/print model parameter counts for sanity checks
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        try:
            # Lightning logging (use grouped keys for W&B friendly display)
            self.log('model/total_params', float(total), prog_bar=False)
            self.log('model/trainable_params', float(trainable), prog_bar=False)
        except Exception:
            pass
        # Friendly console output
        print(f"Model param counts -> total: {total:,}, trainable: {trainable:,}")



class TournamentsDataModule(pl.LightningDataModule if pl is not None else object):
    """Wrap existing get_data_loader into a LightningDataModule; keeps config API.

    The implementation mirrors the behavior in `TrainandTest.py` so configs
    remain directly usable.
    """
    def __init__(self, cfg):
        assert pl is not None, "pytorch_lightning is required for TournamentsDataModule"
        super().__init__()
        self.cfg = cfg
        self._train_loader = None
        self._val_loader = None
        self._test_loader = None

    def _compute_recommended_num_workers(self, cfg: dict = None) -> int:
        """Compute a sensible default for DataLoader.num_workers.

        Priority order:
        1. explicit config value: cfg['train.num_workers'] (if present)
        2. if Trainer is attached and exposes device/process counts, use that to scale
        3. fallback to a heuristic based on CPU count
        """
        cfg = cfg or self.cfg
        # explicit override in config
        cfg_n = cfg.get('train.num_workers', None)
        if isinstance(cfg_n, int):
            return max(0, int(cfg_n))

        # try to infer from trainer/device count
        devices = None
        try:
            trainer = getattr(self, 'trainer', None)
            if trainer is not None:
                devices = getattr(trainer, 'num_devices', None) or getattr(trainer, 'num_processes', None) or getattr(trainer, 'world_size', None)
                if isinstance(devices, (list, tuple)):
                    devices = len(devices)
        except Exception:
            devices = None

        if devices is None:
            # fallback to CUDA device count if available, else 1
            try:
                devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
            except Exception:
                devices = 1

        cpus = os.cpu_count() or 1
        # heuristic: divide CPUs by (2 * devices) and cap to a small maximum
        workers = max(0, min(16, cpus // max(1, devices * 2)))
        return int(workers)

    def setup(self, stage: Optional[str] = None):
        # build train/val loaders only when needed
        if stage in (None, 'fit'):
            cfg = self.cfg
            # first obtain a dataset-backed loader from existing helper (so label_noise etc. remain applied)
            base_loader = get_data_loader(train=True,
                                          batch_size=cfg.get('train.batch_size', 32),
                                          dataset=cfg.get('dataset', 'mnist'),
                                          num_classes=cfg.get('num_classes', 10),
                                          class_list=cfg.get('class_list', None),
                                          samples_per_class=cfg.get('samples_per_class', 10),
                                          resize=cfg.get('resize', 28),
                                          imbalance=cfg.get('imbalance', None),
                                          imbalance_factor=cfg.get('imbalance_factor', 1.0))

            # determine recommended workers (honour explicit cfg if present)
            recommended_workers = self._compute_recommended_num_workers(cfg)

            val_split = cfg.get('val.split', 0.1)
            if val_split > 0:
                total_size = len(base_loader.dataset)
                val_size = int(total_size * val_split)
                train_size = total_size - val_size
                train_dataset, val_dataset = torch.utils.data.random_split(base_loader.dataset, [train_size, val_size])
                self._train_loader = DataLoader(train_dataset, batch_size=cfg.get('train.batch_size', 32), shuffle=True, num_workers=recommended_workers, pin_memory=torch.cuda.is_available(), persistent_workers=True)
                self._val_loader = DataLoader(val_dataset, batch_size=cfg.get('train.batch_size', 32), shuffle=False, num_workers=max(0, recommended_workers // 2), pin_memory=torch.cuda.is_available(), persistent_workers=True)
            else:
                # single loader case: reuse base_loader.dataset but control workers
                self._train_loader = DataLoader(base_loader.dataset, batch_size=cfg.get('train.batch_size', 32), shuffle=True, num_workers=recommended_workers, pin_memory=torch.cuda.is_available(), persistent_workers=True)
                self._val_loader = None

            # label noise is already applied inside the base dataset if requested; no extra handling needed

        if stage in (None, 'test'):
            cfg = self.cfg
            base_test_loader = get_data_loader(train=False,
                                              batch_size=cfg.get('test.batch_size', None),
                                              dataset=cfg.get('dataset', 'mnist'),
                                              num_classes=cfg.get('num_classes', 10),
                                              class_list=cfg.get('class_list', None),
                                              samples_per_class=cfg.get('samples_per_class', 10),
                                              resize=cfg.get('resize', 28))
            recommended_workers = self._compute_recommended_num_workers(cfg)
            # prefer dataset-backed DataLoader so we can control num_workers
            self._test_loader = DataLoader(base_test_loader.dataset, batch_size=cfg.get('test.batch_size', None), shuffle=False, num_workers=recommended_workers, pin_memory=torch.cuda.is_available())

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader


class SaveBestPthCallback(pl.callbacks.Callback if pl is not None else object):
    """Save `best.pth` (state_dict of the wrapped `model`) when monitored
    metric improves. Keeps compatibility with the repo's existing
    `build_model`/`SaveModule` behaviour which expects `best.pth`.

    NOTE: monitor default changed to `val/acc` to match Lightning/W&B keying.
    """
    def __init__(self, save_dir: str, monitor: str = 'val/acc', mode: str = 'max'):
        assert pl is not None, "pytorch_lightning is required for SaveBestPthCallback"
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best = None

    def _better(self, cur, best):
        if best is None:
            return True
        if self.mode == 'max':
            return cur > best
        return cur < best

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if self.monitor not in metrics:
            return
        cur = metrics[self.monitor]
        try:
            cur_val = float(cur)
        except Exception:
            return
        if self._better(cur_val, self.best):
            self.best = cur_val
            path = os.path.join(self.save_dir, 'best.pth')
            # save the wrapped model's state_dict (keeps exact same file used elsewhere)
            torch.save(pl_module.model.state_dict(), path)
            # also write a tiny text file for quick inspection
            with open(os.path.join(self.save_dir, 'best_info.txt'), 'w') as f:
                f.write(f'{self.monitor}={cur_val}\n')


__all__ = ['PLWrapper', 'TournamentsDataModule', 'SaveBestPthCallback']
