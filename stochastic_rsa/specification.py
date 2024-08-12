"""Provides a data class for reach-state-avoid specification"""
import dataclasses
from .aabb import AABBSet


@dataclasses.dataclass
class Specification:
    """
    Reach-state-avoid specification as defined in the paper
    """
    interest_set: AABBSet
    initial_set: AABBSet
    unsafe_set: AABBSet
    target_set: AABBSet
    reach_avoid_probability: float
    stay_probability: float
