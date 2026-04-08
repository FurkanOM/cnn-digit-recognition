"""Common constants and type aliases shared across project utilities."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


DatasetSplit = Dict[str, np.ndarray]
DatasetCollection = Dict[str, DatasetSplit]
EvaluationResults = Dict[str, Dict[str, Any]]
ImageShape = Tuple[int, int, int]

MNIST_HEIGHT = 28
MNIST_WIDTH = 28
NUM_CLASSES = 10
