"""Provides classes and method for sets based on membership functions."""
from .membership_set import MembershipSet, EmptySet
from .sublevel_set import SublevelSet
from .membership_set import union, intersection, difference

__all__ = [
    "MembershipSet",
    "EmptySet",
    "SublevelSet",
    "union",
    "intersection",
    "difference"
]
