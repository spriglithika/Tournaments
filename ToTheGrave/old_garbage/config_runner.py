"""Config runner to instantiate callables/classes from declarative JSON

Config schema (recommended):
{
  "model": {"path": "Models.NeuralIsingTournamentFull", "kwargs": {"num_classes": 3, ...}},
  "dataset": "mnist",
  "num_classes": 3,
  "samples_per_class": 8,
  "resize": 64,
  "out_dir": "./outputs",
  "device": "cuda",  # optional
  "outputs": [
    {"path": "neural_ising_grid.run_single_experiment", "kwargs": {"alpha": 1.0, "train_epochs": 0}},
    {"path": "neural_ising_grid.plot_trace", "kwargs": {"title": "my trace", "savepath": "trace.png"}}
  ]
}

The runner will resolve dotted import paths and also look for names in the provided
context module (typically the calling script) to allow local helpers to be invoked.
"""
from __future__ import annotations

import importlib
import json
import os
from typing import Any, Dict

import torch


def _resolve(path: str):
    """Resolve a dotted path to a Python object by importing its module.

    Example: 'mypkg.mymod.MyClass' -> import mypkg.mymod; return MyClass
    """
    if '.' not in path:
        raise ImportError(f"Cannot resolve bare name '{path}'; use a full dotted path")
    module_name, attr = path.rsplit('.', 1)
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def instantiate(cfg: Dict[str, Any]):
    """Instantiate a callable/class from config.

    cfg must be a dict with keys: 'path' (dotted import path) and optional 'kwargs'.
    """
    if not isinstance(cfg, dict) or 'path' not in cfg:
        raise ValueError("Invalid instantiation config; expected dict with 'path'")
    obj = _resolve(cfg['path'])
    kwargs = cfg.get('kwargs', {}) or {}
    return obj(**kwargs) if callable(obj) else obj


def run_config(cfg: Dict[str, Any]):
    """Run a declarative config.

    - Instantiates `model` if present
    - Builds dataset loaders using `make_mnist_subset` from context if available
    - Executes entries from `outputs` by resolving callables and calling them with
      kwargs merged with contextual defaults (model, device, loaders, out_dir)
    """
    # No external context required; resolution is done via import paths.

    # pick device
    device = cfg.get('device')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    else:
        device = torch.device(device)

    num_classes = int(cfg.get('num_classes', 10))
    samples_per_class = int(cfg.get('samples_per_class', 100))
    resize = int(cfg.get('resize', 32))
    out_dir = cfg.get('out_dir', './experiments/outputs/unlabeled')
    dataset_name = cfg.get('dataset', 'mnist')
    root = cfg.get('root', '.')

    os.makedirs(out_dir, exist_ok=True)

    # Pre-instantiate any top-level config entries that declare a `path`.
    # This allows configs like `"graph": {"path": "mymod.build_graph", "kwargs": {...}}`
    # to be replaced with the built object so callers can use `cfg.get('graph')`.
    raw_store = {}
    for k, v in list(cfg.items()):
        # skip reserved keys
        if k in ('model', 'outputs'):
            continue
        # Dict with path -> instantiate
        if isinstance(v, dict) and 'path' in v:
            try:
                raw_store[k] = v
                cfg[k] = instantiate(v)
            except Exception:
                # leave original on failure
                cfg[k] = v
        # List of such dicts -> instantiate each
        elif isinstance(v, list) and len(v) > 0 and all(isinstance(el, dict) and 'path' in el for el in v):
            try:
                raw_store[k] = v
                cfg[k] = [instantiate(el) for el in v]
            except Exception:
                cfg[k] = v

    if raw_store:
        cfg.setdefault('_raw_configs', {}).update(raw_store)

    # instantiate model if requested
    model = None
    if 'model' in cfg:
        model = instantiate(cfg['model'])
        # try to move model to device if it is a torch module
        try:
            model = model.to(device)
        except Exception:
            pass

    # build loaders using helper in context if available
    train_loader = None
    test_loader = None
    # Build loaders: prefer explicit `loader` path in config, otherwise try
    # the conventional `neural_ising_grid.make_mnist_subset` helper if available.
    loader_path = cfg.get('loader') or cfg.get('loader_path')
    if isinstance(loader_path, str):
        try:
            make_loader = _resolve(loader_path)
            train_loader = make_loader(num_classes=num_classes,
                                       samples_per_class=samples_per_class * 6,
                                       train=True,
                                       resize=resize,
                                       device=device,
                                       dataset=dataset_name,
                                       root=root)
            test_loader = make_loader(num_classes=num_classes,
                                      samples_per_class=samples_per_class,
                                      train=False,
                                      resize=resize,
                                      device=device,
                                      dataset=dataset_name,
                                      root=root)
        except Exception:
            train_loader = None
            test_loader = None
    else:
        # try default helper in this repo
        try:
            make_loader = _resolve('neural_ising_grid.make_mnist_subset')
            train_loader = make_loader(num_classes=num_classes,
                                       samples_per_class=samples_per_class * 6,
                                       train=True,
                                       resize=resize,
                                       device=device,
                                       dataset=dataset_name,
                                       root=root)
            test_loader = make_loader(num_classes=num_classes,
                                      samples_per_class=samples_per_class,
                                      train=False,
                                      resize=resize,
                                      device=device,
                                      dataset=dataset_name,
                                      root=root)
        except Exception:
            train_loader = None
            test_loader = None

    # execution context defaults passed to outputs
    defaults = dict(device=device, model=model, train_loader=train_loader, test_loader=test_loader,
                    num_classes=num_classes, samples_per_class=samples_per_class, resize=resize,
                    out_dir=out_dir, cfg=cfg)

    outputs = cfg.get('outputs', []) or []
    results = []
    for out_cfg in outputs:
        if isinstance(out_cfg, str):
            path = out_cfg
            kwargs = {}
        elif isinstance(out_cfg, dict):
            path = out_cfg.get('path')
            kwargs = out_cfg.get('kwargs', {}) or {}
        else:
            raise ValueError('Invalid output entry, must be string or dict')

        func = _resolve(path)

        # merge defaults but allow explicit kwargs to override
        call_kwargs = dict(defaults)
        call_kwargs.update(kwargs)

        # call the function; it's the function's responsibility to accept these
        try:
            res = func(**call_kwargs)
            results.append(res)
        except TypeError:
            # try calling with positional (model) then kwargs
            try:
                res = func(model, **call_kwargs)
                results.append(res)
            except Exception as e:
                raise

    return results


def load_and_run(config_path: str):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    return run_config(cfg)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <config.json>")
        sys.exit(1)
    config_path = sys.argv[1]
    load_and_run(config_path)