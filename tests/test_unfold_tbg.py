"""Integration test for Unfold using TBG MACE MLIP data (m_2_r_1_sc_4_mace, unfolded onto layer0 PC)."""

import numpy
from ase.build import make_supercell
from ase.io import read
from phonopy.cui.load import load as load_phonopy
from phonopy.file_IO import read_force_constants_hdf5

from unphold import Unfold
from unphold.utils import atoms_ph2ase, calculate_pc_rotation_angle, match_two_2d_atoms_pbc_with_2d_frac_shift

TBG_NAME = "m_2_r_1_sc_4_mace"


def _load_tbg(data_dir, name):
    run_dir = data_dir / "tbg" / name
    ph = load_phonopy(run_dir / "phonopy.yaml")
    ph.force_constants = read_force_constants_hdf5(run_dir / "force_constants.h5")
    return ph


def test_tbg_layer0_weight_conservation(data_dir):
    """Unfolding weight sum per k-point must equal 3 * n_uc_atoms (layer0 PC).

    This is an algebraic identity of the projector (holds as long as perm_sc2gen is a
    complete, injective mapping and no vacancies), independent of how well the real,
    relaxed moire structure matches the ideal rigid supercell used to build the projector.
    """
    run_dir = data_dir / "tbg" / TBG_NAME
    ph_tbg = _load_tbg(data_dir, TBG_NAME)
    tmat_l0 = dict(**numpy.load(run_dir / "tmat.npz", allow_pickle=True))["tmat_l0"]
    atoms_gp_pc = read(run_dir / "gp_pc.xyz")
    atoms_tbg_uc = atoms_ph2ase(ph_tbg.unitcell)

    ret_pc_rot = calculate_pc_rotation_angle(atoms_gp_pc, tmat_l0)
    atoms_pc_rot = ret_pc_rot["atoms_pc_rot"]
    sc_from_pc_rot = make_supercell(atoms_pc_rot, tmat_l0)

    layer0_indices = numpy.where(atoms_tbg_uc.positions[:, 2] < atoms_tbg_uc.positions[:, 2].mean())[0]
    atoms_layer0 = atoms_tbg_uc[layer0_indices]

    match_result = match_two_2d_atoms_pbc_with_2d_frac_shift(
        atoms_layer0,
        sc_from_pc_rot,
        shift_0_frac=0.005,
        shift_0_seg=6,
        shift_1_frac=0.005,
        shift_1_seg=6,
        spatial_tolerance=1.0,  # moire relaxation causes real out-of-plane bending
        tolerance_xyz_scaler=numpy.array([0.5, 0.5, 1.5]),
        ignore_z=False,
    )
    assert "atoms_indices_a2b" in match_result, "layer0 <-> rigid-PC-supercell matching failed"
    perm_sc2gen_l0 = layer0_indices[match_result["atoms_indices_a2b"]]

    unfold = Unfold(
        unitcell=atoms_pc_rot,
        supercell=atoms_tbg_uc,
        transformation_matrix=tmat_l0,
        perm_sc2gen=perm_sc2gen_l0,
    )
    unfold.set_kpts_in_unitcell(numpy.array([[0.0, 0.0, 0.0]]), format="fractional")  # Gamma only, cheap
    unfold.calculate_sc_phonon(dyn_sc=ph_tbg.dynamical_matrix, factor="thz")
    unfold.calculate_weights()

    expected = 3 * len(atoms_pc_rot)
    weight_sum = unfold.weights.sum(axis=1)[0]
    assert abs(weight_sum - expected) / expected < 0.02, f"weight sum {weight_sum:.3f} != {expected} (>2% off)"
