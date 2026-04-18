"""Small utilities: seeding, device selection, logging."""

from __future__ import annotations

import logging
import os
import random
import subprocess
from contextlib import contextmanager
from typing import Iterator


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy, torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(preferred: str = "auto") -> str:
    """Return 'cuda', 'mps', or 'cpu'. 'auto' picks the best available."""
    if preferred != "auto":
        return preferred
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def git_sha(repo_dir: str | None = None) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def get_logger(name: str = "mech_spoof", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
        logger.addHandler(h)
        logger.setLevel(level)
    return logger


@contextmanager
def timer(label: str, logger: logging.Logger | None = None) -> Iterator[None]:
    import time
    logger = logger or get_logger()
    t0 = time.time()
    try:
        yield
    finally:
        logger.info(f"[{label}] {time.time() - t0:.2f}s")
