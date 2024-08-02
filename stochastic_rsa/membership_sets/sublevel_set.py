"""
Provides the class and some methods for sublevel sets based on some function.
"""
from typing import TypeVar, Callable
from .membership_set import MembershipSet

T = TypeVar("T")


class SublevelMembershipSet(MembershipSet[T]):
    """
    Sublevel set, that is, a set of all elements where the value of
    some function does not exceed a given threshold.
    """

    def __init__(self, function: Callable[[T], float], threshold: float):
        self.threshold = threshold
        self.function = function

        def membership_function(x: T) -> bool:
            return function(x) <= self.threshold

        super().__init__(membership_function)


def boundary(a: SublevelMembershipSet[T]) -> MembershipSet[T]:
    """Sublevel set's boundary.

    Args:
        a (SublevelMembershipSet[T]): the set

    Returns:
        MembershipSet[T]: its boundary
    """
    def membership_function(x: T) -> bool:
        return a.function(x) == a.threshold
    return MembershipSet(membership_function)


def interior(a: SublevelMembershipSet[T]) -> MembershipSet[T]:
    """Sublevel set's interior.

    Args:
        a (SublevelMembershipSet[T]): the set

    Returns:
        MembershipSet[T]: its interior
    """
    def membership_function(x: T) -> bool:
        return a.function(x) < a.threshold
    return MembershipSet(membership_function)
