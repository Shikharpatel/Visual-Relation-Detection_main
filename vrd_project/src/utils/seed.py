"""
Seed management for reproducibility.
=====================================

Provides deterministic seed control for PyTorch, NumPy, and Python random.
"""

import os
import random
import numpy as np
import torch

_GLOBAL_SEED = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value
        deterministic: If True, use deterministic algorithms (may impact performance)
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

    # Python random
    random.seed(seed)

    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        # Enable deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set CUBLAS workspace config for deterministic CUDA operations
        # Required for operations like LSTM on CUDA with deterministic algorithms
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

        # PyTorch 1.8+ deterministic flag
        if hasattr(torch, 'use_deterministic_algorithms'):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                # Older PyTorch versions don't have warn_only parameter
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    # Some operations don't have deterministic implementations
                    pass
    else:
        # Allow non-deterministic for better performance
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_seed() -> int:
    """
    Get the currently set global seed.

    Returns:
        The global seed value, or None if not set
    """
    return _GLOBAL_SEED


def worker_init_fn(worker_id: int) -> None:
    """
    Initialize worker with unique seed based on global seed.

    Use this as worker_init_fn in DataLoader for reproducibility.

    Args:
        worker_id: Worker ID from DataLoader
    """
    global _GLOBAL_SEED
    if _GLOBAL_SEED is not None:
        worker_seed = _GLOBAL_SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


def get_generator(seed: int = None) -> torch.Generator:
    """
    Get a PyTorch Generator with optional seed.

    Args:
        seed: Optional seed, uses global seed if None

    Returns:
        Seeded torch.Generator
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    elif _GLOBAL_SEED is not None:
        g.manual_seed(_GLOBAL_SEED)
    return g
