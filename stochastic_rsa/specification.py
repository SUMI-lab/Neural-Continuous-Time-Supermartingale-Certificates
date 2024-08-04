"""Provides a data class for reach-state-avoid specification"""
from typing import Callable
import dataclasses
from .membership_sets import MembershipSet, SublevelSet


@dataclasses.dataclass
class Specification:
    """
    Reach-state-avoid specification as defined in the paper
    """
    time_homogenous: bool
    interest_set: MembershipSet
    initial_set: MembershipSet
    unsafe_set: MembershipSet
    target_set: SublevelSet
    reach_avoid_probability: float
    stay_probability: float
