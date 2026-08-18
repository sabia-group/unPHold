"""Smoke tests - verify the package is importable and core classes instantiate."""

import numpy
import pytest


def test_import():
    import unphold  # noqa: F401


def test_public_api():
    from unphold import Unfold
    from unphold.metrics import compute_APR, compute_L, compute_V, compute_V_p1
    from unphold.utils import band_expansion, concatenate_bands, gaussian_function

    assert callable(Unfold)
    assert callable(concatenate_bands)
    assert callable(compute_APR)


def test_gaussian_function_scalar_mu():
    from unphold.utils import gaussian_function

    x = numpy.linspace(-3, 3, 100)
    g = gaussian_function(x, mu=0, sigma=1.0)
    assert g.shape == x.shape
    # integral ≈ 1
    assert abs(numpy.trapezoid(g, x) - 1.0) < 0.01


def test_gaussian_function_array_mu():
    from unphold.utils import gaussian_function

    x = numpy.linspace(-5, 5, 200)
    mu = numpy.array([0.0, 1.0, 2.0])
    g = gaussian_function(x, mu=mu, sigma=0.5)
    assert g.shape == (3, 200)


def test_band_expansion_shape():
    from unphold.utils import band_expansion

    energies = numpy.array([10.0, 20.0, 30.0])
    grid = numpy.linspace(0, 40, 200)
    result = band_expansion(energies, grid, sigma=1.0)
    assert result.shape == (3, 200)


def test_concatenate_bands_simple():
    from unphold.utils import concatenate_bands

    seg0 = numpy.linspace([0, 0, 0], [0.5, 0, 0], 5)  # shape (5, 3)
    seg1 = numpy.linspace([0.5, 0, 0], [1.0, 0, 0], 5)
    kpts, indices = concatenate_bands([seg0, seg1], connections=[True, False])
    # connected: seg0 loses last point → 4 + 5 = 9
    assert kpts.shape[0] == 9
    assert indices[0] == 0


def test_fractional_part_around_zero():
    from unphold.utils import fractional_part_around_zero

    arr = numpy.array([-1.5, -0.5, 0.0, 0.25, 0.5, 1.5, 2.5])
    result = fractional_part_around_zero(arr)
    assert result.shape == arr.shape
    assert numpy.all(result >= -0.5) and numpy.all(result < 0.5)
    # non-halfway values are unaffected by the rounding convention
    assert numpy.isclose(fractional_part_around_zero(numpy.array([0.25]))[0], 0.25)


def test_match_two_2d_atoms_pbc_with_2d_frac_shift_identity():
    from ase.atoms import Atoms as aseAtoms

    from unphold.utils import match_two_2d_atoms_pbc_with_2d_frac_shift

    cell = numpy.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 20.0]])
    positions = numpy.array([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]])
    a = aseAtoms(symbols=["C", "C"], cell=cell, positions=positions, pbc=True)
    b = a.copy()

    result = match_two_2d_atoms_pbc_with_2d_frac_shift(a, b, shift_0_seg=3, shift_1_seg=3)
    assert "atoms_indices_a2b" in result
    numpy.testing.assert_array_equal(result["atoms_indices_a2b"], numpy.array([0, 1]))
    numpy.testing.assert_array_equal(result["atoms_indices_b2a"], numpy.array([0, 1]))


def test_match_two_2d_atoms_pbc_with_2d_frac_shift_no_match():
    from ase.atoms import Atoms as aseAtoms

    from unphold.utils import match_two_2d_atoms_pbc_with_2d_frac_shift

    cell = numpy.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 20.0]])
    a = aseAtoms(symbols=["C", "C"], cell=cell, positions=[[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]], pbc=True)
    b = aseAtoms(symbols=["C", "C"], cell=cell, positions=[[0.3, 0.7, 5.0], [1.6, 0.2, 5.0]], pbc=True)

    result = match_two_2d_atoms_pbc_with_2d_frac_shift(
        a, b, shift_0_frac=0.01, shift_0_seg=3, shift_1_frac=0.01, shift_1_seg=3
    )
    assert "atoms_indices_a2b" not in result
    assert "atoms_dist_list" in result


def test_match_atoms_with_vacancies_single_vacancy():
    from ase.atoms import Atoms as aseAtoms

    from unphold.utils import match_atoms_with_vacancies

    cell = numpy.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]])
    ideal = aseAtoms(
        symbols=["C", "C", "C"],
        cell=cell,
        positions=[[0.0, 0.0, 5.0], [1.0, 1.0, 5.0], [2.0, 2.0, 5.0]],
        pbc=True,
    )
    real = ideal.copy()
    del real[1]  # remove the middle atom -> single vacancy

    result = match_atoms_with_vacancies(ideal, real, spatial_tolerance=1e-2)
    assert result["fail_reason"] is None
    numpy.testing.assert_array_equal(result["perm_real2ideal"], numpy.array([0, -1, 1]))
    numpy.testing.assert_array_equal(result["vacancy_indices"], numpy.array([1]))


def test_match_atoms_with_vacancies_no_vacancy_is_full_permutation():
    from ase.atoms import Atoms as aseAtoms

    from unphold.utils import match_atoms_with_vacancies

    cell = numpy.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]])
    ideal = aseAtoms(
        symbols=["C", "C"],
        cell=cell,
        positions=[[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]],
        pbc=True,
    )
    real = ideal.copy()

    result = match_atoms_with_vacancies(ideal, real, spatial_tolerance=1e-2)
    assert result["fail_reason"] is None
    numpy.testing.assert_array_equal(result["perm_real2ideal"], numpy.array([0, 1]))
    assert len(result["vacancy_indices"]) == 0


def test_match_atoms_with_vacancies_interstitial_fails():
    from ase.atoms import Atoms as aseAtoms

    from unphold.utils import match_atoms_with_vacancies

    cell = numpy.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]])
    ideal = aseAtoms(symbols=["C"], cell=cell, positions=[[0.0, 0.0, 5.0]], pbc=True)
    real = aseAtoms(symbols=["C", "C"], cell=cell, positions=[[0.0, 0.0, 5.0], [1.5, 1.5, 5.0]], pbc=True)

    result = match_atoms_with_vacancies(ideal, real, spatial_tolerance=1e-2)
    assert result["perm_real2ideal"] is None
    assert "interstitial" in result["fail_reason"] or "more atoms" in result["fail_reason"]


def test_match_atoms_with_vacancies_species_mismatch_fails():
    from ase.atoms import Atoms as aseAtoms

    from unphold.utils import match_atoms_with_vacancies

    cell = numpy.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]])
    ideal = aseAtoms(symbols=["C", "N"], cell=cell, positions=[[0.0, 0.0, 5.0], [1.0, 1.0, 5.0]], pbc=True)
    real = aseAtoms(symbols=["C"], cell=cell, positions=[[1.0, 1.0, 5.0]], pbc=True)  # wrong species at this site

    result = match_atoms_with_vacancies(ideal, real, spatial_tolerance=1e-2)
    assert result["perm_real2ideal"] is None
    assert result["fail_reason"] is not None


def test_calculate_pc_rotation_angle_removes_shear():
    from ase.atoms import Atoms as aseAtoms
    from ase.build import make_supercell

    from unphold.utils import calculate_pc_rotation_angle

    # hexagonal graphene-like PC, deliberately tilted off the x-axis
    cell = numpy.array([[2.46, 0.0, 0.0], [1.23, 2.13042249, 0.0], [0.0, 0.0, 20.0]])
    positions = numpy.array([[0.0, 0.0, 10.0], [1.23, 0.71014083, 10.0]])
    atoms_pc = aseAtoms(symbols=["C", "C"], cell=cell, positions=positions, pbc=True)
    tmat = numpy.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])

    result = calculate_pc_rotation_angle(atoms_pc, tmat)
    assert "atoms_pc_rot" in result and "rot_angle_deg" in result
    assert len(result["atoms_pc_rot"]) == len(atoms_pc)

    # tilts past 90 deg would fool an arctan(y/x)-based angle (180 deg ambiguity)
    for tilt_deg in (7.0, 100.0, 170.0):
        atoms_pc_tilt = atoms_pc.copy()
        atoms_pc_tilt.rotate(tilt_deg, "z", rotate_cell=True)
        result = calculate_pc_rotation_angle(atoms_pc_tilt, tmat)
        sc_rot = make_supercell(result["atoms_pc_rot"], tmat)
        assert abs(sc_rot.cell[0, 1]) < 1e-8, f"tilt {tilt_deg} deg: cell[0, 1] not zero"
        assert sc_rot.cell[0, 0] > 0, f"tilt {tilt_deg} deg: first lattice vector not along +x"


def test_unfold_save_load_roundtrip(tmp_path):
    import pickle

    from ase.atoms import Atoms as aseAtoms
    from ase.build import make_supercell

    from unphold import Unfold

    uc = aseAtoms("C", cell=numpy.eye(3) * 2.0, positions=[[0.0, 0.0, 0.0]], pbc=True)
    tmat = numpy.diag([2, 2, 2])
    sc = make_supercell(uc, tmat, wrap=False)
    unfold = Unfold(unitcell=uc, supercell=sc, transformation_matrix=tmat)
    unfold.set_kpts_in_unitcell(numpy.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), format="fractional")
    nbands = 3 * len(sc)
    unfold.bs_sc_energies = numpy.zeros((2, nbands))
    unfold.bs_sc_eigenvecs = numpy.eye(nbands)[None, ...].repeat(2, axis=0)
    unfold.weights = numpy.ones((2, nbands))

    fpath = tmp_path / "unfold.pkl"
    unfold.save(fpath)
    loaded = Unfold.load(fpath)

    numpy.testing.assert_array_equal(loaded.perm_sc2gen, unfold.perm_sc2gen)
    numpy.testing.assert_array_equal(loaded.kpts_uc_frac, unfold.kpts_uc_frac)
    numpy.testing.assert_allclose(loaded.kpts_sc_frac, unfold.kpts_sc_frac)
    numpy.testing.assert_allclose(loaded.bs_sc_energies, unfold.bs_sc_energies)
    numpy.testing.assert_allclose(loaded.bs_sc_eigenvecs, unfold.bs_sc_eigenvecs)
    numpy.testing.assert_allclose(loaded.weights, unfold.weights)

    # a valid pickle that is not an unPHold save payload must be rejected
    bad_fpath = tmp_path / "not_a_save_file.pkl"
    with open(bad_fpath, "wb") as f:
        pickle.dump({"foo": 1}, f)
    with pytest.raises(ValueError):
        Unfold.load(bad_fpath)


def test_relax_by_spring_origin_spring_pulls_atom_back():
    from ase.atoms import Atoms as aseAtoms

    from unphold.utils import RelaxBySpring

    cell = numpy.eye(3) * 10.0
    positions = numpy.array([[5.0, 5.0, 5.0]])
    atoms = aseAtoms(symbols=["C"], cell=cell, positions=positions, pbc=True)

    relax = RelaxBySpring(atoms)
    relax.install_origin_spring(k=1.0)
    displaced = relax.positions.copy()
    displaced[0] += 1.0
    relax._pos = displaced  # simulate a displacement away from the origin spring target

    relax.relax(steps=200, delta=0.05)
    assert numpy.allclose(relax.positions, positions, atol=1e-2)
    info = relax.get_spring_info()
    assert info == {"origin_springs": 1, "layer_springs": 0, "neighbor_springs": 0}

    relax.clear_springs()
    assert relax.get_spring_info() == {"origin_springs": 0, "layer_springs": 0, "neighbor_springs": 0}
