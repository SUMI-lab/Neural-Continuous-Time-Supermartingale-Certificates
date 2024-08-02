"""Provides a data class for reach-state-avoid specification"""
from typing import Callable
import dataclasses
from .membership_sets import MembershipSet


@dataclasses.dataclass
class RSASpecification:
    """
    Reach-state-avoid specification as defined in the paper
    """
    interest_set: MembershipSet
    initial_set: MembershipSet
    unsafe_set_family: Callable[[float], MembershipSet]
    target_set_family: Callable[[float], MembershipSet]
    reach_avoid_probability: float
    stay_probability: float
