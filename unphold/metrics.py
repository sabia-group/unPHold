"""Phonon mode character metrics.

Functions for quantifying the physical character of phonon modes:

- [`compute_APR`][unphold.metrics.compute_APR]: acoustic participation ratio (APR)
- [`compute_L`][unphold.metrics.compute_L]: longitudinality (L)
- [`compute_V`][unphold.metrics.compute_V]: out-of-plane polarisation (V)

Each has a `*_from_phonopy` wrapper that takes a `Phonopy` object instead of
explicit eigenvectors.
"""

import numpy
from ase.atoms import Atoms as aseAtoms
from phonopy import Phonopy

from .utils import atoms_ph2ase


def compute_APR_from_phonopy(ph: Phonopy) -> list:
    """Compute APR for all k-path segments from a Phonopy object.

    Phonopy must have its band structure already computed (``ph.run_band_structure``
    called with ``with_eigenvectors=True``).

    Args:
        ph (Phonopy): Phonopy object with computed band structure.

    Returns:
        list[numpy.ndarray]: APR per segment, each of shape ``(nqpoints, nbands)``.
    """
    assert ph._band_structure is not None, "Band structure not computed."
    apr_list = []
    for kseg_idx in range(len(ph._band_structure.qpoints)):
        apr_list.append(
            compute_APR(
                atoms=atoms_ph2ase(ph.primitive),
                ph_eigvecs=ph._band_structure.eigenvectors[kseg_idx],
            )
        )
    return apr_list


def compute_APR(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,
) -> numpy.ndarray:
    r"""Acoustic participation ratio (APR) for phonon modes.

    Quantifies how acoustic-like a mode is. APR = 1 for a perfect acoustic mode,
    APR → 0 for an optic mode.

    $$\mathrm{APR}_{q,n} = \frac{2}{N(N+1)}
    \frac{
        \left| \sum_{\alpha \leq \beta} A_{\alpha\beta} \right|^2
    }{
        \sum_{\alpha \leq \beta} \left| A_{\alpha\beta} \right|^2
    },
    \qquad
    A_{\alpha\beta} =
    \frac{(e_{q,n}^\alpha)^\dagger e_{q,n}^\beta}{\sqrt{m_\alpha m_\beta}}$$

    The sums run over unique atom pairs $\alpha \leq \beta$, following the
    reference definition. Both sums are evaluated in O(N) memory without
    forming the pair matrix $A$.

    Reference:
        N. Strasser et al., *Int. J. Mol. Sci.* **25**, 5 (2024).

    Args:
        atoms (aseAtoms): Structure with atomic masses (must match ``ph_eigvecs``).
        ph_eigvecs (numpy.ndarray): Eigenvectors, shape ``(nqpoints, natoms*3, nbands)``.

    Returns:
        numpy.ndarray: APR values, shape ``(nqpoints, nbands)``.

    Raises:
        ValueError: If the eigenvector dimension is not a multiple of 3, or the
            atom count disagrees between ``atoms`` and ``ph_eigvecs``.
    """
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    if natoms3 % 3 != 0:
        raise ValueError(f"eigenvector dimension {natoms3} is not a multiple of 3")
    natoms = natoms3 // 3
    masses = numpy.asarray(atoms.get_masses())
    if masses.shape[0] != natoms:
        raise ValueError(
            f"mass/eigenvector mismatch: {masses.shape[0]} atoms in `atoms` but {natoms} in `ph_eigvecs`. "
            "If these came from a Phonopy object, the eigenvectors are over `ph.primitive`, not `ph.unitcell`."
        )

    # G[q, a, x, n] = e / sqrt(m), so A_ab = sum_x conj(G_ax) G_bx
    G = ph_eigvecs.reshape(nqpoints, natoms, 3, nbands) / numpy.sqrt(masses)[None, :, None, None]

    # sum over a <= b of A_ab, via prefix sums: sum_b conj(sum_{a <= b} G_a) . G_b
    G_prefix = numpy.cumsum(G, axis=1)
    triu_sum = numpy.einsum("qaxn,qaxn->qn", G_prefix.conj(), G)
    numerator = numpy.abs(triu_sum) ** 2

    # sum over a <= b of |A_ab|^2 = (sum over all (a, b) + diagonal) / 2, since |A_ab| = |A_ba|
    diag = numpy.einsum("qaxn,qaxn->qan", G.conj(), G).real  # A_aa
    gram = numpy.einsum("qaxn,qayn->qxyn", G.conj(), G)  # 3x3 Cartesian Gram matrix
    sum_absA2_all = numpy.einsum("qxyn,qxyn->qn", gram, gram.conj()).real
    denominator = 0.5 * (sum_absA2_all + numpy.einsum("qan,qan->qn", diag, diag))

    return (2 / (natoms * (natoms + 1))) * (numerator / denominator)


def compute_L_from_phonopy(ph: Phonopy) -> list:
    """Compute longitudinality for all k-path segments from a Phonopy object.

    Args:
        ph (Phonopy): Phonopy object with computed band structure.

    Returns:
        list[numpy.ndarray]: L per segment, each of shape ``(nqpoints, nbands)``.
    """
    assert ph._band_structure is not None, "Band structure not computed."
    L_list = []
    cell_reciprocal = atoms_ph2ase(ph.primitive).cell.reciprocal()
    for kseg_idx in range(len(ph._band_structure.qpoints)):
        L_list.append(
            compute_L(
                atoms=atoms_ph2ase(ph.primitive),
                ph_eigvecs=ph._band_structure.eigenvectors[kseg_idx],
                q=2 * numpy.pi * ph._band_structure.qpoints[kseg_idx] @ cell_reciprocal,
            )
        )
    return L_list


def compute_L(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,
    q: numpy.ndarray,
) -> numpy.ndarray:
    r"""Longitudinality of phonon modes.

    Measures the degree to which atomic displacement directions are parallel to
    the wavevector **q**. L = 1 for a purely longitudinal in-phase mode, L = 0
    for a transverse one.

    $$L_{q,n} = \left|
        \frac{1}{N} \sum_{\alpha=1}^{N}
        \frac{\hat{q} \cdot e_{q,n}^{\alpha}}{|e_{q,n}^{\alpha}|}
    \right|$$

    Note:
        The projections are averaged with their sign, so an antiphase longitudinal
        mode (e.g. LO-like or bilayer LA antiphase) gives L ~= 0, the same as a transverse mode.
        Atoms with vanishing displacement contribute ~= 0: the per-atom norm in the denominator
        is regularised by a small constant.

    Reference:
        L. Legenstein et al., *ACS Mater. Au* **3**, 371 (2023).

    Args:
        atoms (aseAtoms): Structure (used for natoms consistency check).
        ph_eigvecs (numpy.ndarray): Eigenvectors, shape ``(nqpoints, natoms*3, nbands)``.
        q (numpy.ndarray): Cartesian q-vectors (without 2π), shape ``(nqpoints, 3)``.

    Returns:
        numpy.ndarray: L values, shape ``(nqpoints, nbands)``.
    """
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    natoms = len(atoms)
    assert natoms3 == natoms * 3
    # per-atom direction e_a / |e_a| + regularised by 1e-5 of the average per-atom amplitude |e| / sqrt(N)
    eigvec_norms = numpy.linalg.norm(ph_eigvecs, axis=1)[:, None, None, :]
    ph_eigvec = ph_eigvecs.reshape(nqpoints, natoms, 3, nbands)
    atom_norms = numpy.linalg.norm(ph_eigvec, axis=2)[:, :, None, :]
    ph_eigvec_normed = ph_eigvec / (atom_norms + 1e-5 * eigvec_norms / natoms**0.5)

    q_normed = q / (numpy.linalg.norm(q, axis=1)[:, None] + 1e-5)

    lgt = numpy.einsum("qaxn,qx->qan", ph_eigvec_normed, q_normed)
    return numpy.abs(lgt.mean(axis=1))


def compute_V_p1(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,
) -> numpy.ndarray:
    r"""Verticality of phonon modes (linear average).

    Measures out-of-plane character. V = 1 for purely out-of-plane,
    V = 0 for purely in-plane.

    Note:
        This variant over-emphasises atoms with small displacements.
        Prefer [`compute_V`][unphold.metrics.compute_V] in most cases.

    Args:
        atoms (aseAtoms): Structure.
        ph_eigvecs (numpy.ndarray): Eigenvectors, shape ``(nqpoints, natoms*3, nbands)``.

    Returns:
        numpy.ndarray: V values, shape ``(nqpoints, nbands)``.
    """
    ph_eigvec_normed = ph_eigvecs / numpy.linalg.norm(ph_eigvecs, axis=1)[:, None, :]
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    natoms = len(atoms)
    assert natoms3 == natoms * 3
    ph_eigvec_normed = ph_eigvec_normed.reshape(nqpoints, natoms, 3, nbands)

    vtcl = numpy.abs(ph_eigvec_normed[:, :, 2, :])
    vtcl = natoms**0.5 * vtcl.mean(axis=1)
    return vtcl


def compute_V_from_phonopy(ph: Phonopy) -> list:
    """Compute verticality (p=2 form) for all k-path segments from a Phonopy object.

    Args:
        ph (Phonopy): Phonopy object with computed band structure.

    Returns:
        list[numpy.ndarray]: V per segment, each of shape ``(nqpoints, nbands)``.
    """
    assert ph._band_structure is not None, "Band structure not computed."
    V_list = []
    for kseg_idx in range(len(ph._band_structure.qpoints)):
        V_list.append(
            compute_V(
                atoms=atoms_ph2ase(ph.primitive),
                ph_eigvecs=ph._band_structure.eigenvectors[kseg_idx],
            )
        )
    return V_list


def compute_V(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,
) -> numpy.ndarray:
    r"""Verticality of phonon modes (collective / p=2 norm).

    Preferred over [`compute_V_p1`][unphold.metrics.compute_V_p1] because it does not over-weight atoms
    with small displacements.

    $$V_{q,n}^{p=2} =
    \frac{\sum_\alpha |\hat{z} \cdot e_{q,n}^\alpha|^2}
         {\sum_\alpha |e_{q,n}^\alpha|^2}$$

    Args:
        atoms (aseAtoms): Structure.
        ph_eigvecs (numpy.ndarray): Eigenvectors, shape ``(nqpoints, natoms*3, nbands)``.

    Returns:
        numpy.ndarray: V values, shape ``(nqpoints, nbands)``.
    """
    ph_eigvec_normed = ph_eigvecs / numpy.linalg.norm(ph_eigvecs, axis=1)[:, None, :]
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    natoms = len(atoms)
    assert natoms3 == natoms * 3
    ph_eigvec_normed = ph_eigvec_normed.reshape(nqpoints, natoms, 3, nbands)

    vtcl2 = numpy.abs(ph_eigvec_normed[:, :, 2, :])
    vtcl2 = numpy.sum(vtcl2**2, axis=1)
    return vtcl2


def rotmat_xy(angle: float) -> numpy.ndarray:
    """3×3 rotation matrix for rotation about the z-axis.

    Args:
        angle (float): Rotation angle in radians.

    Returns:
        numpy.ndarray: Rotation matrix, shape ``(3, 3)``.
    """
    return numpy.array(
        [
            [numpy.cos(angle), -numpy.sin(angle), 0],
            [numpy.sin(angle), numpy.cos(angle), 0],
            [0, 0, 1],
        ]
    )
