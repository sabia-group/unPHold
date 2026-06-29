"""Phonon mode character metrics.

Functions for quantifying the physical character of phonon modes:
acoustic participation ratio (APR), longitudinality (L), verticality (V),
and Gaussian band expansion for plotting.
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
    for kseg_idx in range(len(ph._band_structure.get_qpoints())):
        apr_list.append(
            compute_APR(
                atoms=atoms_ph2ase(ph.unitcell),
                ph_eigvecs=ph._band_structure.get_eigenvectors()[kseg_idx],
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
        \left| \sum_{\alpha,\beta}
        \frac{(e_{q,n}^\alpha)^\dagger e_{q,n}^\beta}{\sqrt{m_\alpha m_\beta}}
        \right|^2
    }{
        \sum_{\alpha,\beta}
        \left| \frac{(e_{q,n}^\alpha)^\dagger e_{q,n}^\beta}{\sqrt{m_\alpha m_\beta}} \right|^2
    }$$

    Reference:
        N. Strasser et al., *Int. J. Mol. Sci.* **25**, 5 (2024).

    Args:
        atoms (aseAtoms): Structure with atomic masses.
        ph_eigvecs (numpy.ndarray): Eigenvectors, shape ``(nqpoints, natoms*3, nbands)``.

    Returns:
        numpy.ndarray: APR values, shape ``(nqpoints, nbands)``.
    """
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    assert natoms3 % 3 == 0
    natoms = natoms3 // 3

    masses_sqrt = numpy.sqrt(atoms.get_masses())
    eigvec_div_mass_sqrt = ph_eigvecs.reshape(nqpoints, natoms, 3, nbands) / masses_sqrt[None, :, None, None]

    inner_prod = numpy.einsum(
        "qaxn,qbxn->qabn",
        eigvec_div_mass_sqrt.conj(),
        eigvec_div_mass_sqrt,
    )

    triu_indices = numpy.triu_indices(natoms)
    inner_prod_triu = inner_prod[:, triu_indices[0], triu_indices[1], :]

    numerator = numpy.abs(inner_prod_triu.sum(axis=1)) ** 2
    denominator = numpy.sum(numpy.abs(inner_prod_triu) ** 2, axis=1)

    N = natoms
    apr = (2 / (N * (N + 1))) * (numerator / denominator)
    del inner_prod
    return apr


def compute_L_from_phonopy(ph: Phonopy) -> list:
    """Compute longitudinality for all k-path segments from a Phonopy object.

    Args:
        ph (Phonopy): Phonopy object with computed band structure.

    Returns:
        list[numpy.ndarray]: L per segment, each of shape ``(nqpoints, nbands)``.
    """
    assert ph._band_structure is not None, "Band structure not computed."
    L_list = []
    cell_reciprocal = atoms_ph2ase(ph.unitcell).cell.reciprocal()
    for kseg_idx in range(len(ph._band_structure.get_qpoints())):
        L_list.append(
            compute_L(
                atoms=atoms_ph2ase(ph.unitcell),
                ph_eigvecs=ph._band_structure.get_eigenvectors()[kseg_idx],
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

    Measures the degree to which atomic displacements are parallel to the
    wavevector **q**. L = 1 for a purely longitudinal mode, L = 0 for transverse.

    $$L_{q,n} = \left|
        \frac{1}{N} \sum_{\alpha=1}^{N}
        \frac{\hat{q} \cdot e_{q,n}^{\alpha}}{|e_{q,n}^{\alpha}|}
    \right|$$

    Args:
        atoms (aseAtoms): Structure (used for natoms consistency check).
        ph_eigvecs (numpy.ndarray): Eigenvectors, shape ``(nqpoints, natoms*3, nbands)``.
        q (numpy.ndarray): Cartesian q-vectors (without 2π), shape ``(nqpoints, 3)``.

    Returns:
        numpy.ndarray: L values, shape ``(nqpoints, nbands)``.
    """
    ph_eigvec_normed = ph_eigvecs / numpy.linalg.norm(ph_eigvecs, axis=1)[:, None, :]
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    natoms = len(atoms)
    assert natoms3 == natoms * 3
    ph_eigvec_normed = ph_eigvec_normed.reshape(nqpoints, natoms, 3, nbands)

    q_normed = q / (numpy.linalg.norm(q, axis=1)[:, None] + 1e-5)

    lgt = numpy.einsum("qaxn,qx->qan", ph_eigvec_normed, q_normed)
    lgt = lgt.mean(axis=1)
    lgt = natoms**0.5 * numpy.abs(lgt)
    return lgt


def compute_V(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,
) -> numpy.ndarray:
    r"""Verticality of phonon modes (linear average).

    Measures out-of-plane character. V = 1 for purely out-of-plane,
    V = 0 for purely in-plane.

    Note:
        This variant over-emphasises atoms with small displacements.
        Prefer [`compute_V_p2`][unphold.metrics.compute_V_p2] in most cases.

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


def compute_V_p2(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,
) -> numpy.ndarray:
    r"""Verticality of phonon modes (collective / p=2 norm).

    Preferred over :func:`compute_V` because it does not over-weight atoms
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


def rotmat_xOy(angle: float) -> numpy.ndarray:
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
