"""Provides auxiliary type hints."""

from typing import Sequence, Callable
from torch import Tensor


type tensor = Tensor
type tensors = Sequence[tensor]
type vector = Sequence[float] | tensor
type tensor_function = Callable[[vector, tensor], tensor]
