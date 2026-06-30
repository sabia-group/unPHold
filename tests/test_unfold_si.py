"""Integration tests for Unfold using Si FHI-aims data (uc_1_sc_2_aims / uc_2_sc_1_aims)."""

import numpy
from phonopy.cui.load import load as load_phonopy
from phonopy.file_IO import read_force_constants_hdf5
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from unphold import Unfold
from unphold.utils import atoms_ph2ase, concatenate_bands, gaussian_function


SI_TMAT = numpy.diag([2, 2, 2])

SI_KPATH = [
    [  # fractional coordinates in unit cell BZ
        [0.0, 0.0, 0.0],  # G
        [0.5, 0.0, 0.5],  # X
        [0.625, 0.25, 0.625],  # U
        [1.0, 1.0, 1.0],  # G1
        [0.5, 0.5, 0.5],  # L
    ]
]

SI_BZ_PATH = [["G", "X", "U", "G1", "L"]]

ENERGY_GRID = numpy.arange(-1.0, 18.0001, 0.01)
SIGMA = 0.1


def _load_si(data_dir, name, primitive_matrix=None):
    """Load phonopy object + force constants for one Si run directory.

    Args:
        primitive_matrix: Passed to phonopy. Use ``'P'`` (identity) when loading
            a SC-as-unitcell run (e.g. ``uc_2_sc_1_aims``) so that phonopy 4+ does
            not auto-reduce to the 2-atom FCC primitive (default changed to ``'auto'``).
    """
    run_dir = data_dir / "si" / name
    ph = load_phonopy(run_dir / "phonopy.yaml", primitive_matrix=primitive_matrix)
    ph.force_constants = read_force_constants_hdf5(run_dir / "force_constants.h5")
    return ph


def _build_si_kpts():
    kpts_uc, connections = get_band_qpoints_and_path_connections(SI_KPATH, npoints=21)
    kpts_flat, bz_idx = concatenate_bands(kpts_uc, connections)
    return kpts_uc, kpts_flat, connections, bz_idx


def test_si_weight_conservation(data_dir):
    """Unfolding weight sum per k-point must equal 3 * n_uc_atoms = 6 (Si 2-atom PC)."""
    ph_sc = _load_si(data_dir, "uc_2_sc_1_aims", primitive_matrix="P")

    kpts_uc, kpts_flat, connections, _ = _build_si_kpts()
    kpts_sc = [kpt @ SI_TMAT for kpt in kpts_uc]  # transfrom from fractional UC to fractional SC

    ph_sc.run_band_structure(kpts_sc, path_connections=connections, labels=SI_BZ_PATH)

    atoms_uc = atoms_ph2ase(_load_si(data_dir, "uc_1_sc_2_aims").unitcell)
    atoms_sc = atoms_ph2ase(ph_sc.unitcell)

    unfold = Unfold(unitcell=atoms_uc, supercell=atoms_sc, transformation_matrix=SI_TMAT)
    unfold.set_kpts_in_unitcell(kpts_flat, format="fractional")
    unfold.calculate_sc_phonon(dyn_sc=ph_sc.dynamical_matrix, factor="thz")
    unfold.calculate_weights()

    expected = 3 * len(atoms_uc)  # = 6
    weight_sum_mean = unfold.weights.sum(axis=1).mean()
    assert abs(weight_sum_mean - expected) < 1e-6, f"Weight sum mean {weight_sum_mean:.8f} != {expected}"


def test_si_spectral_matches_uc(data_dir):
    """For a perfect SC, unfolded spectral function must reproduce the UC phonon spectrum.

    Pearson r > 0.9999 and max normalised |diff| < 0.001 (manually validated: r = 1.0, diff = 0, machine precision).
    """
    ph_uc = _load_si(data_dir, "uc_1_sc_2_aims")
    ph_sc = _load_si(data_dir, "uc_2_sc_1_aims", primitive_matrix="P")

    kpts_uc, kpts_flat, connections, _ = _build_si_kpts()
    kpts_sc = [kpt @ SI_TMAT.T for kpt in kpts_uc]

    ph_uc.run_band_structure(kpts_uc, path_connections=connections, labels=SI_BZ_PATH)
    ph_sc.run_band_structure(kpts_sc, path_connections=connections, labels=SI_BZ_PATH)

    atoms_uc = atoms_ph2ase(ph_uc.unitcell)
    atoms_sc = atoms_ph2ase(ph_sc.unitcell)

    unfold = Unfold(unitcell=atoms_uc, supercell=atoms_sc, transformation_matrix=SI_TMAT)
    unfold.set_kpts_in_unitcell(kpts_flat, format="fractional")
    unfold.calculate_sc_phonon(dyn_sc=ph_sc.dynamical_matrix, factor="thz")
    unfold.calculate_weights()
    grid, _ = unfold.calculate_band_expansion(grid=ENERGY_GRID, sigma=SIGMA)
    spectral_unfolded = unfold.energies_on_grid  # (nkpts, ngrid)

    # Build reference UC spectral function (weight=1 per mode)
    uc_freqs_flat = numpy.concatenate(
        [seg[:-1] if c else seg for seg, c in zip(ph_uc._band_structure.frequencies, connections)]
    )  # (nkpts_flat, n_uc_modes)
    assert uc_freqs_flat.shape[0] == len(kpts_flat)

    spectral_uc = numpy.stack(
        [gaussian_function(grid, mu=uc_freqs_flat[ik], sigma=SIGMA).sum(axis=0) for ik in range(len(kpts_flat))]
    )  # (nkpts, ngrid)

    a = spectral_unfolded.ravel()
    b = spectral_uc.ravel()
    pearson_r = numpy.corrcoef(a, b)[0, 1]

    a_norm = a / a.max()
    b_norm = b / b.max()
    max_rel_diff = numpy.abs(a_norm - b_norm).max()

    assert pearson_r > 0.9999, f"Pearson r = {pearson_r:.6f}, expected > 0.9999"
    assert max_rel_diff < 0.001, f"Max normalised |diff| = {max_rel_diff:.4f}, expected < 0.001"
