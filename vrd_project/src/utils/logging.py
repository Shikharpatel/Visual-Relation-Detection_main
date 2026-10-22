"""
Logging utilities for VRD training and evaluation.
===================================================

Provides both standard Python logging and CSV-based metric logging.
"""

import os
import sys
import logging
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Global logger registry
_LOGGERS: Dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "vrd",
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    console: bool = True,
    file_logging: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and/or file handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Logging level
        console: Whether to log to console
        file_logging: Whether to log to file

    Returns:
        Configured logger
    """
    global _LOGGERS

    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # Clear existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        fmt='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file_logging and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'{name}_{timestamp}.log'

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def get_logger(name: str = "vrd") -> logging.Logger:
    """
    Get an existing logger or create a basic one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name in _LOGGERS:
        return _LOGGERS[name]
    return setup_logger(name, console=True, file_logging=False)


class CSVLogger:
    """
    CSV-based metric logger for training curves.

    Logs metrics to a CSV file that can be loaded for plotting or analysis.
    """

    def __init__(
        self,
        log_dir: str,
        filename: str = "metrics.csv",
        fieldnames: Optional[List[str]] = None
    ):
        """
        Initialize CSV logger.

        Args:
            log_dir: Directory for the CSV file
            filename: Name of the CSV file
            fieldnames: Column names (will be inferred if not provided)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self.fieldnames = fieldnames
        self._file = None
        self._writer = None
        self._initialized = False

    def _initialize(self, fieldnames: List[str]) -> None:
        """Initialize the CSV file with headers."""
        if self._initialized:
            return

        self.fieldnames = fieldnames
        self._file = open(self.filepath, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()
        self._initialized = True

    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log a row of metrics.

        Args:
            metrics: Dictionary of metric name -> value
        """
        if not self._initialized:
            # Initialize with keys from first log call
            fieldnames = list(metrics.keys())
            self._initialize(fieldnames)

        # Ensure all fields are present
        row = {k: metrics.get(k, '') for k in self.fieldnames}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        """Close the CSV file."""
        if self._file:
            self._file.close()
            self._file = None
            self._initialized = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TensorBoardLogger:
    """
    TensorBoard logger wrapper (optional dependency).
    """

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard logger.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.enabled = True
        except ImportError:
            self.writer = None
            self.enabled = False
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value."""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars."""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int) -> None:
        """Log a histogram."""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, img_tensor, step: int) -> None:
        """Log an image."""
        if self.enabled:
            self.writer.add_image(tag, img_tensor, step)

    def close(self) -> None:
        """Close the writer."""
        if self.enabled and self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MetricTracker:
    """
    Track and aggregate metrics over batches/epochs.
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}

    def update(self, name: str, value: float, count: int = 1) -> None:
        """
        Update a metric with a new value.

        Args:
            name: Metric name
            value: Metric value (usually batch average)
            count: Number of samples in this update
        """
        if name not in self.metrics:
            self.metrics[name] = []
            self.counts[name] = 0

        self.metrics[name].append(value * count)
        self.counts[name] += count

    def get_average(self, name: str) -> float:
        """Get the average of a metric."""
        if name not in self.metrics or self.counts[name] == 0:
            return 0.0
        return sum(self.metrics[name]) / self.counts[name]

    def get_all_averages(self) -> Dict[str, float]:
        """Get averages of all tracked metrics."""
        return {name: self.get_average(name) for name in self.metrics}

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
