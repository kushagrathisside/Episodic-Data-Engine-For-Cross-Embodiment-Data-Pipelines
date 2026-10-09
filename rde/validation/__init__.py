"""Validation checks, organised by the question they answer.

``structural``     does the data exist and hold together?
``temporal``       is time well-behaved?
``numerical``      are the numbers physically plausible?
``dataset_level``  is the collection as a whole fit for use?

:func:`default_registry` assembles the standard suite.  Callers can build their
own registry to run a subset, or register custom checks for a new modality.
"""

from .base import (
    Check,
    CheckContext,
    CheckRegistry,
    DatasetCheck,
    EpisodeCheck,
)
from .dataset_level import DATASET_CHECKS
from .numerical import NUMERICAL_CHECKS
from .structural import STRUCTURAL_CHECKS
from .temporal import TEMPORAL_CHECKS


def default_registry(include_sync: bool = True) -> CheckRegistry:
    """The standard suite: structural, temporal, numerical, dataset-level, sync.

    Sync checks are imported lazily: :mod:`rde.synchronization` builds on
    :mod:`rde.validation.base`, so importing it at module scope would be
    circular.  The dependency direction is deliberate -- alignment is a
    *consumer* of the check framework, not part of its foundation.
    """
    checks = [*STRUCTURAL_CHECKS, *TEMPORAL_CHECKS, *NUMERICAL_CHECKS, *DATASET_CHECKS]
    if include_sync:
        from ..synchronization.align import SYNC_CHECKS

        checks = [*checks, *SYNC_CHECKS]
    return CheckRegistry().add_all(checks)


__all__ = [
    "DATASET_CHECKS",
    "NUMERICAL_CHECKS",
    "STRUCTURAL_CHECKS",
    "TEMPORAL_CHECKS",
    "Check",
    "CheckContext",
    "CheckRegistry",
    "DatasetCheck",
    "EpisodeCheck",
    "default_registry",
]
