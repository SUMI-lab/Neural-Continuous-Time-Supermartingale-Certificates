"""
Provides the base class and some methods for sets based on membership functions.
"""
from typing import Callable, Self
import torch
from .singleton import Singleton


class MembershipSet:
    """Base class for sets based on membership functions."""

    def __init__(self,
                 membership_function: Callable[[torch.Tensor], torch.Tensor]
                 ):
        super().__init__()
        self._membership_function = membership_function

    def contains(self, elements: torch.Tensor) -> torch.Tensor:
        """Inclusion relationship.

        Args:
            elements (torch.Tensor): elements to check

        Returns:
            torch.bool: `True` for the elements in the set, otherwise `False`
        """
        return self._membership_function(elements)

    def filter(self, elements: torch.Tensor) -> torch.Tensor:
        """Filter a tensor removing elements not in the set.

        Args:
            elements (torch.Tensor): an iterable of elements to check

        Returns:
            torch.Tensor:
                a list of elements from the iterable that are in the set
        """
        return elements[self.contains(elements), :]

    @property
    def complement(self) -> Self:
        """
        Complement of a set, that is the set containing elements not in the
        original set.

        Returns:
            MembershipSet: the complement
        """
        return MembershipSet(lambda x: torch.logical_not(self.contains(x)))


class EmptySet(MembershipSet, metaclass=Singleton):
    """The empty set."""

    def __init__(self,):
        super().__init__(lambda _: False)


def union(a: MembershipSet, b: MembershipSet) -> MembershipSet:
    """Union of two sets.

    Args:
        a (MembershipSet[T]): first set
        b (MembershipSet[T]): second set

    Returns:
        MembershipSet[T]: the union of the given sets
    """
    return MembershipSet(lambda x:
                         torch.logical_or(a.contains(x), b.contains(x))
                         )


def intersection(a: MembershipSet, b: MembershipSet) -> MembershipSet:
    """Intersection of two sets.

    Args:
        a (MembershipSet[T]): first set
        b (MembershipSet[T]): second set

    Returns:
        MembershipSet[T]: the intersection of the given sets
    """
    return MembershipSet(lambda x:
                         torch.logical_and(a.contains(x), b.contains(x))
                         )


def difference(a: MembershipSet, b: MembershipSet) -> MembershipSet:
    """Difference of two sets.

    Args:
        a (MembershipSet[T]): first set
        b (MembershipSet[T]): second set

    Returns:
        MembershipSet[T]: the difference, that is, all element of a not in b
    """
    return intersection(a, b.complement)
