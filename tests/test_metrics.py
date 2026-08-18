"""Unit tests for the mode character metrics (APR, L, V) on synthetic eigenvectors."""

import numpy
import pytest
from ase.atoms import Atoms as aseAtoms
from phonopy.cui.load import load as load_phonopy
from phonopy.file_IO import read_force_constants_hdf5
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from unphold.metrics import (
    compute_APR,
    compute_APR_from_phonopy,
    compute_L,
    compute_L_from_phonopy,
    compute_V,
    compute_V_from_phonopy,
    compute_V_p1,
)
from unphold.utils import atoms_ph2ase


def _two_atoms(masses=(1.0, 1.0)) -> aseAtoms:
    atoms = aseAtoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.set_masses(masses)
    return atoms


def _pack_modes(*modes) -> numpy.ndarray:
    """Stack per-mode displacement lists into an eigenvector array of shape (1, natoms*3, nbands).

    Each mode is a list of per-atom 3-vectors; every mode is flattened and normalized to unit norm.
    """
    columns = []
    for mode in modes:
        v = numpy.asarray(mode, dtype=complex).reshape(-1)
        columns.append(v / numpy.linalg.norm(v))
    return numpy.stack(columns, axis=1)[None, :, :]


def test_apr_in_phase_is_one():
    """Two atoms moving identically (equal masses) form a perfect acoustic mode: APR = 1."""
    eigvecs = _pack_modes([[1, 0, 0], [1, 0, 0]])
    apr = compute_APR(atoms=_two_atoms(), ph_eigvecs=eigvecs)
    assert numpy.allclose(apr, 1.0)


def test_apr_acoustic_and_optic():
    """APR is computed independently per q-point and per band."""
    per_q = _pack_modes([[1, 0, 0], [1, 0, 0]], [[1, 0, 0], [-1, 0, 0]])  # acoustic, optic
    eigvecs = numpy.concatenate([per_q, per_q], axis=0)  # two identical q-points
    apr = compute_APR(atoms=_two_atoms(), ph_eigvecs=eigvecs)
    assert apr.shape == (2, 2)
    assert numpy.allclose(apr, [[1.0, 1.0 / 9.0], [1.0, 1.0 / 9.0]])


def test_apr_matches_dense_pair_sum():
    """The O(N) evaluation reproduces the explicit dense unique-pair sum (complex eigenvectors)."""
    rng = numpy.random.default_rng(7)
    natoms, nbands = 4, 5
    eigvecs = rng.normal(size=(2, natoms * 3, nbands)) + 1j * rng.normal(size=(2, natoms * 3, nbands))
    eigvecs /= numpy.linalg.norm(eigvecs, axis=1, keepdims=True)
    atoms = aseAtoms("H2ON", positions=numpy.zeros((natoms, 3)))  # unequal masses

    G = eigvecs.reshape(2, natoms, 3, nbands) / numpy.sqrt(atoms.get_masses())[None, :, None, None]
    pair = numpy.einsum("qaxn,qbxn->qabn", G.conj(), G)
    triu = numpy.triu_indices(natoms)
    pair_triu = pair[:, triu[0], triu[1], :]
    reference = (
        (2 / (natoms * (natoms + 1)))
        * numpy.abs(pair_triu.sum(axis=1)) ** 2
        / numpy.sum(numpy.abs(pair_triu) ** 2, axis=1)
    )
    assert numpy.allclose(compute_APR(atoms=atoms, ph_eigvecs=eigvecs), reference)


def test_apr_mismatched_atom_count_raises():
    """A structure/eigenvector atom-count mismatch raises with a hint about ph.primitive."""
    eigvecs = _pack_modes([[1, 0, 0], [1, 0, 0]])  # 2 atoms
    atoms = aseAtoms("H3", positions=numpy.zeros((3, 3)))
    with pytest.raises(ValueError, match="ph.primitive"):
        compute_APR(atoms=atoms, ph_eigvecs=eigvecs)


def test_longitudinality_parallel_and_perpendicular():
    """Displacements along q give L = 1, perpendicular to q give L = 0."""
    eigvecs = _pack_modes([[1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]])
    q = numpy.array([[1.0, 0.0, 0.0]])
    lgt = compute_L(atoms=_two_atoms(), ph_eigvecs=eigvecs, q=q)
    assert numpy.allclose(lgt, [[1.0, 0.0]], atol=1e-4)


def test_longitudinality_zero_at_gamma():
    """At q = 0 the propagation direction is undefined and L is evaluated as 0."""
    eigvecs = _pack_modes([[1, 0, 0], [1, 0, 0]])
    lgt = compute_L(atoms=_two_atoms(), ph_eigvecs=eigvecs, q=numpy.zeros((1, 3)))
    assert numpy.allclose(lgt, 0.0)


def test_longitudinality_amplitude_independent():
    """A longitudinal mode with unequal per-atom amplitudes still gives L = 1.

    Per-atom normalisation means only displacement directions enter, so a 2:1
    amplitude ratio along q does not reduce L.
    """
    eigvecs = _pack_modes([[2, 0, 0], [1, 0, 0]])
    q = numpy.array([[1.0, 0.0, 0.0]])
    lgt = compute_L(atoms=_two_atoms(), ph_eigvecs=eigvecs, q=q)
    assert numpy.allclose(lgt, 1.0, atol=1e-4)


def test_longitudinality_antiphase_is_zero():
    """An antiphase longitudinal mode averages to L = 0, like a transverse one."""
    eigvecs = _pack_modes([[1, 0, 0], [-1, 0, 0]])
    q = numpy.array([[1.0, 0.0, 0.0]])
    lgt = compute_L(atoms=_two_atoms(), ph_eigvecs=eigvecs, q=q)
    assert numpy.allclose(lgt, 0.0, atol=1e-4)


def test_verticality_out_of_plane_and_in_plane():
    """Displacements along z give V = 1, in-plane give V = 0, an equal mix gives V = 0.5."""
    eigvecs = _pack_modes(
        [[0, 0, 1], [0, 0, 1]],  # purely out-of-plane
        [[1, 0, 0], [0, 1, 0]],  # purely in-plane
        [[1, 0, 1], [1, 0, 1]],  # half in-plane, half out-of-plane
    )
    vp2 = compute_V(atoms=_two_atoms(), ph_eigvecs=eigvecs)
    assert numpy.allclose(vp2, [[1.0, 0.0, 1 / 2]])


def test_verticality_p1_out_of_plane_and_in_plane():
    """V_p1 reaches 1 along z and drops to 0 in plane, as the p=2 form does.

    The two forms part ways on the third mode, where every atom is half in-plane and half out-of-plane: the linear average gives 1/sqrt{2}, the p=2 form gives 1/2.
    """
    eigvecs = _pack_modes(
        [[0, 0, 1], [0, 0, 1]],  # purely out-of-plane
        [[1, 0, 0], [0, 1, 0]],  # purely in-plane
        [[1, 0, 1], [1, 0, 1]],  # half in-plane, half out-of-plane
    )
    vp1 = compute_V_p1(atoms=_two_atoms(), ph_eigvecs=eigvecs)
    assert numpy.allclose(vp1, [[1.0, 0.0, 1 / numpy.sqrt(2)]])


def test_verticality_p1_over_weights_small_displacements():
    """For a large in-plane + small out-of-plane pair, V_p1 exceeds the p=2 form.

    With amplitudes (a, b) = (sqrt(0.99), sqrt(0.01)): V = b^2 = 0.01 while
    V_p1 = sqrt(2) * (0 + b) / 2 = 0.1 / sqrt(2), about seven times larger.
    """
    a, b = numpy.sqrt(0.99), numpy.sqrt(0.01)
    eigvecs = _pack_modes([[a, 0, 0], [0, 0, b]])
    vp2 = compute_V(atoms=_two_atoms(), ph_eigvecs=eigvecs)
    vp1 = compute_V_p1(atoms=_two_atoms(), ph_eigvecs=eigvecs)
    assert numpy.allclose(vp2, b**2)
    assert numpy.allclose(vp1, numpy.sqrt(2) * b / 2)
    assert vp1[0, 0] > vp2[0, 0]


def test_from_phonopy_wrappers_match_direct(data_dir):
    """The *_from_phonopy wrappers reproduce the direct per-segment computation on graphene."""
    run_dir = data_dir / "graphene" / "uc_1_sc_9_mace"
    ph = load_phonopy(run_dir / "phonopy.yaml")
    ph.force_constants = read_force_constants_hdf5(run_dir / "force_constants.h5")
    kpts, connections = get_band_qpoints_and_path_connections(
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2 / 3, 1 / 3, 0.0]]], npoints=3
    )
    ph.run_band_structure(kpts, path_connections=connections, with_eigenvectors=True)
    bs = ph.band_structure
    atoms = atoms_ph2ase(ph.primitive)
    cell_reciprocal = atoms.cell.reciprocal()

    apr_wrapped = compute_APR_from_phonopy(ph)
    lgt_wrapped = compute_L_from_phonopy(ph)
    v_wrapped = compute_V_from_phonopy(ph)
    assert len(apr_wrapped) == len(lgt_wrapped) == len(v_wrapped) == len(bs.qpoints)
    for seg_idx in range(len(bs.qpoints)):
        apr_direct = compute_APR(atoms=atoms, ph_eigvecs=bs.eigenvectors[seg_idx])
        lgt_direct = compute_L(
            atoms=atoms,
            ph_eigvecs=bs.eigenvectors[seg_idx],
            q=2 * numpy.pi * bs.qpoints[seg_idx] @ cell_reciprocal,
        )
        v_direct = compute_V(atoms=atoms, ph_eigvecs=bs.eigenvectors[seg_idx])
        assert numpy.allclose(apr_wrapped[seg_idx], apr_direct)
        assert numpy.allclose(lgt_wrapped[seg_idx], lgt_direct)
        assert numpy.allclose(v_wrapped[seg_idx], v_direct)


def test_from_phonopy_wrappers_with_nonidentity_primitive_matrix(data_dir):
    """The wrappers follow ``ph.primitive`` when it differs from ``ph.unitcell``.

    With primitive_matrix = identity/2 (what phonopy v4 resolves on its own through
    its default ``primitive_matrix="auto"``), the 16-atom Si cell reduces to a
    2-atom primitive cell and the eigenvectors carry 6 bands, not 48.
    """
    run_dir = data_dir / "si" / "uc_2_sc_1_aims"
    # plain list, not numpy array: phonopy compares primitive_matrix == "auto" internally
    ph = load_phonopy(
        run_dir / "phonopy.yaml",
        primitive_matrix=[[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]],
    )
    ph.force_constants = read_force_constants_hdf5(run_dir / "force_constants.h5")
    kpts, connections = get_band_qpoints_and_path_connections([[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]], npoints=3)
    ph.run_band_structure(kpts, path_connections=connections, with_eigenvectors=True)
    assert len(ph.unitcell) == 16 and len(ph.primitive) == 2

    apr_wrapped = compute_APR_from_phonopy(ph)
    for seg in (*apr_wrapped, *compute_L_from_phonopy(ph), *compute_V_from_phonopy(ph)):
        assert seg.shape == (3, 3 * len(ph.primitive))
        assert numpy.all(numpy.isfinite(seg))
    # at Gamma (first point of the first segment) the acoustic modes are in-phase: APR = 1
    assert numpy.allclose(apr_wrapped[0][0, :3], 1.0)
