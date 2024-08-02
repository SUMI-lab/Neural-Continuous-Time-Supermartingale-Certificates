"""
Provides the base class and some methods for sets based on membership functions.
"""
from typing import TypeVar, Generic, Callable

T = TypeVar("T")


class MembershipSet(Generic["T"]):
    """Base class for sets based on membership functions."""

    def __init__(self, membership_function: Callable[[T], bool]):
        super().__init__()
        self._membership_function = membership_function

    def contains(self, element: T) -> bool:
        """Inclusion relationship.

        Args:
            element (T): element to check

        Returns:
            bool: `True` if the set contains the element, otherwise `False`
        """
        return self._membership_function(element)


def union(a: MembershipSet[T], b: MembershipSet[T]) -> MembershipSet[T]:
    """Union of two sets.

    Args:
        a (MembershipSet[T]): first set
        b (MembershipSet[T]): second set

    Returns:
        MembershipSet[T]: the union of the given sets
    """
    return MembershipSet[T](lambda x: a.contains(x) or b.contains(x))


def intersection(a: MembershipSet[T], b: MembershipSet[T]) -> MembershipSet[T]:
    """Intersection of two sets.

    Args:
        a (MembershipSet[T]): first set
        b (MembershipSet[T]): second set

    Returns:
        MembershipSet[T]: the intersection of the given sets
    """
    return MembershipSet[T](lambda x: a.contains(x) and b.contains(x))


def complement(a: MembershipSet[T]) -> MembershipSet[T]:
    """Complement of a set.

    Args:
        a (MembershipSet[T]): the set

    Returns:
        MembershipSet[T]: its complement
    """
    return MembershipSet[T](lambda x: not a.contains(x))
