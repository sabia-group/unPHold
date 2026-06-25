"""unPHold — Phonon band unfolding for moiré and supercell systems.

Public API
----------
Core unfolding::

    from unPHold import Unfold, UnfoldTwistBilayer, concatenate_bands

Mode character metrics::

    from unPHold import compute_APR, compute_APR_from_phonopy
    from unPHold import compute_L, compute_L_from_phonopy
    from unPHold import compute_V, compute_V_p2
    from unPHold import band_expansion, gaussian_function
"""

from .unfold import Unfold, UnfoldTwistBilayer, concatenate_bands
from .metrics import (
    compute_APR,
    compute_APR_from_phonopy,
    compute_L,
    compute_L_from_phonopy,
    compute_V,
    compute_V_p2,
    band_expansion,
    gaussian_function,
    rotmat_xOy,
)

__all__ = [
    # unfolding
    "Unfold",
    "UnfoldTwistBilayer",
    "concatenate_bands",
    # metrics
    "compute_APR",
    "compute_APR_from_phonopy",
    "compute_L",
    "compute_L_from_phonopy",
    "compute_V",
    "compute_V_p2",
    "band_expansion",
    "gaussian_function",
    "rotmat_xOy",
]