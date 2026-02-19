"""Minimal JSON config reader with a dict-like `ConfigDict` class.

Provides a thin wrapper around nested dicts with:
- dotted-key `get_dot()` method
- `deep_update()` to merge another dict in place
- `from_file()` / `to_file()` helpers

Top-level convenience functions `load_config`, `save_config`, `get`, and
`deep_merge` are provided for backwards compatibility.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Mapping


class ConfigDict(dict):
    """A dict subclass with helpers for nested/dotted access and merging.

    Example:
        cfg = ConfigDict.from_file('cfg.json')
        val = cfg.get_dot('train.batch_size', default=32)
        cfg.deep_update({'train': {'lr': 1e-3}})
    """

    @classmethod
    def from_file(cls, path: str) -> 'ConfigDict':
        path = os.path.expanduser(os.path.expandvars(path))
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(data)

    def to_file(self, path: str) -> None:
        path = os.path.expanduser(os.path.expandvars(path))
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self, f, indent=2, sort_keys=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by dotted key, e.g. 'a.b.c'. Returns default if missing."""
        if key is None or key == '':
            return self
        cur: Any = self
        for part in key.split('.'):
            if not isinstance(cur, Mapping):
                return default
            if part not in cur:
                return default
            cur = cur[part]
        return cur

    def deep_update(self, other: Mapping) -> None:
        """Merge `other` into `self` recursively (in-place)."""
        for k, v in other.items():
            if k in self and isinstance(self[k], dict) and isinstance(v, Mapping):
                # recursively merge
                ConfigDict(self[k]).deep_update(v)
                self[k] = self[k]
            else:
                self[k] = v

    @staticmethod
    def deep_merge(a: Mapping, b: Mapping) -> 'ConfigDict':
        """Return a new ConfigDict with `b` merged into `a` (non-destructive)."""
        out = ConfigDict(dict(a))
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
                out[k] = ConfigDict.deep_merge(out[k], v)
            else:
                out[k] = v
        return out


# Backwards-compatible convenience functions
def load_config(path: str) -> ConfigDict:
    return ConfigDict.from_file(path)


def save_config(cfg: Mapping, path: str) -> None:
    cfg_out = cfg if isinstance(cfg, dict) else dict(cfg)
    path = os.path.expanduser(os.path.expandvars(path))
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(cfg_out, f, indent=2, sort_keys=True)


def get(cfg: Mapping, key: str, default: Any = None) -> Any:
    if isinstance(cfg, ConfigDict):
        return cfg.get(key, default=default)
    cd = ConfigDict(dict(cfg))
    return cd.get(key, default=default)


def deep_merge(a: Mapping, b: Mapping) -> ConfigDict:
    return ConfigDict.deep_merge(a, b)


__all__ = ['ConfigDict', 'load_config', 'save_config', 'get', 'deep_merge']
