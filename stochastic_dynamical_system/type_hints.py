from typing import Sequence
from torch import Tensor


type tensor = Tensor
type tensors = Sequence[tensor]
type vector = Sequence[float] | tensor
