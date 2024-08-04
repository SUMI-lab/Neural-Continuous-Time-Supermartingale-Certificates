"""
Provides the class and some methods for sublevel sets based on some function.
"""
from typing import Callable, Self
import torch
from .membership_set import MembershipSet


class SublevelSet(MembershipSet):
    """
    Sublevel set, that is, a set of all elements where the value of
    some function does not exceed a given threshold.
    """

    def __init__(self,
                 function: Callable[[torch.Tensor], torch.Tensor],
                 threshold: torch.float
                 ):
        self.threshold = threshold
        self.function = function

        def membership_function(x: torch.Tensor) -> torch.bool:
            return function(x).flatten() <= self.threshold

        super().__init__(membership_function)

    @property
    def boundary(self) -> MembershipSet:
        """Sublevel set's boundary.

        Returns:
            MembershipSet: the boundary
        """
        def membership_function(x: torch.Tensor) -> torch.bool:
            return self.function(x) == self.threshold
        return MembershipSet(membership_function)

    @property
    def interior(self) -> MembershipSet:
        """Sublevel set's interior.

        Returns:
            MembershipSet: the interior
        """
        def membership_function(x: torch.Tensor) -> torch.bool:
            return self.function(x) < self.threshold
        return MembershipSet(membership_function)

    @property
    def interior_complement(self) -> MembershipSet:
        """Complement of the sublevel set's interior.

        Returns:
            MembershipSet: the complement of the set's interior
        """
        def membership_function(x: torch.Tensor) -> torch.bool:
            return self.function(x) >= self.threshold
        return MembershipSet(membership_function)
