"""Integration tests for Unfold on a graphene monolayer vacancy (uc_1_sc_9_mace / vacancy_uc_9_sc_1_mace).

The vacancy cell has one atom fewer than the ideal 9×9×1 primitive-cell tiling, so ``Unfold``'s
automatic (equal-count) matching cannot be used; ``match_atoms_with_vacancies`` builds the
``perm_sc2gen`` index array instead, marking the vacancy site with ``-1``.
"""

import numpy
from ase.build import make_supercell
from ase.io import read
from phonopy import Phonopy
from phonopy.cui.load import load as load_phonopy
from phonopy.file_IO import read_force_constants_hdf5

from unphold import Unfold
from unphold.utils import atoms_ph2ase, match_atoms_with_vacancies

GRAPHENE_TMAT = numpy.diag([9, 9, 1])  # 2-atom primitive cell -> ideal 9×9×1 tiling (162 sites)


def _load_graphene(data_dir, name) -> Phonopy:
    """Load phonopy object + force constants for one graphene run directory."""
    run_dir = data_dir / "graphene" / name
    ph = load_phonopy(run_dir / "phonopy.yaml")
    ph.force_constants = read_force_constants_hdf5(run_dir / "force_constants.h5")
    return ph


def _match_vacancy(data_dir):
    """Shared setup: load the primitive cell, the real vacancy cell, and the vacancy matching."""
    atoms_pc = read(data_dir / "graphene" / "vacancy_uc_9_sc_1_mace" / "gp_pc.xyz")  # relaxed 2-atom primitive cell
    ph_vac = _load_graphene(data_dir, "vacancy_uc_9_sc_1_mace")
    sc_real = atoms_ph2ase(ph_vac.unitcell)  # 161-atom cell with one vacancy
    sc_by_tmat = make_supercell(atoms_pc, GRAPHENE_TMAT, wrap=False)  # ideal 162-site tiling
    match = match_atoms_with_vacancies(ideal=sc_by_tmat, real=sc_real, spatial_tolerance=0.5)
    return ph_vac, atoms_pc, sc_real, match


def test_graphene_vacancy_matching(data_dir):
    """match_atoms_with_vacancies must find exactly one vacancy and match all 161 real atoms."""
    _, atoms_pc, sc_real, match = _match_vacancy(data_dir)

    assert match["fail_reason"] is None, match["fail_reason"]
    perm = match["perm_real2ideal"]
    n_ideal = len(atoms_pc) * int(round(numpy.linalg.det(GRAPHENE_TMAT)))  # 2 * 81 = 162
    assert perm.shape == (n_ideal,), f"perm shape {perm.shape} != ({n_ideal},)"
    assert len(match["vacancy_indices"]) == 1, f"found {len(match['vacancy_indices'])} vacancies, expected 1"
    n_matched = int((perm >= 0).sum())
    assert n_matched == len(sc_real), f"matched {n_matched} atoms != {len(sc_real)} real atoms"


def test_graphene_vacancy_weight_conservation(data_dir):
    """Unfolding weight sum per k-point equals 3*n_uc_atoms - 3*n_v/N_uc = 6 - 3/81 (one vacancy)."""
    ph_vac, atoms_pc, sc_real, match = _match_vacancy(data_dir)
    assert match["fail_reason"] is None, match["fail_reason"]

    unfold = Unfold(
        unitcell=atoms_pc,
        supercell=sc_real,
        transformation_matrix=GRAPHENE_TMAT,
        perm_sc2gen=match["perm_real2ideal"],
    )

    # Three k-points along Γ–K keep the test fast (~0.4 s of diagonalization).
    kpts = numpy.array([[0.0, 0.0, 0.0], [1.0 / 3.0, 1.0 / 6.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.0]])
    unfold.set_kpts_in_unitcell(kpts, format="fractional")
    unfold.calculate_sc_phonon(dyn_sc=ph_vac.dynamical_matrix, factor="thz")
    unfold.calculate_weights()

    n_vac = len(match["vacancy_indices"])
    expected = 3 * len(atoms_pc) - 3 * n_vac / unfold.nucs_in_sc  # 6 - 3/81 = 5.96296...
    weight_sum_mean = unfold.weights.sum(axis=1).mean()
    assert abs(weight_sum_mean - expected) < 1e-4, f"Weight sum mean {weight_sum_mean:.8f} != {expected:.8f}"
