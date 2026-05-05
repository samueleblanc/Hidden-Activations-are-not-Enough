"""Registry + dispatch for the 9-measure panel.

Used by the per-chunk workers (S1, S2, S3) to iterate over all measures
without each worker needing to know each measure's class name.
"""
from typing import Dict, List, Type
from .base import MeasureBase
from .cka import DebiasedLinearCKA
from .angular_cka import AngularCKA
from .procrustes import ProcrustesShapeDistance
from .bures import BuresSimilarity
from .soft_matching import SoftMatching
from .rsa import RSASpearman
from .output_jsd import OutputJSD
from .gromov_wasserstein import GromovWasserstein
from .dcor import DistanceCorrelation


# All 9 measures. Order matters for table generation.
PANEL: List[Type[MeasureBase]] = [
    DebiasedLinearCKA, AngularCKA, ProcrustesShapeDistance, BuresSimilarity,
    SoftMatching, RSASpearman, OutputJSD, GromovWasserstein, DistanceCorrelation,
]


def panel_names() -> List[str]:
    return [cls.name for cls in PANEL]


def cross_dim_native_names() -> List[str]:
    return [cls.name for cls in PANEL if cls.cross_dim_native]


def make_panel() -> Dict[str, MeasureBase]:
    return {cls.name: cls() for cls in PANEL}
