"""Internal utilities for atom matching and structure conversion.

These are implementation details used by the Unfold class.
They are not part of the public API and should not be imported directly by users.
"""

import numpy
from ase.atoms import Atoms as aseAtoms
from phonopy.structure.atoms import PhonopyAtoms


def match_two_atoms(
    a: aseAtoms,
    b: aseAtoms,
    spatial_tolerance: float = 1e-2,
):
    """Match atoms between two ASE Atoms objects by position.

    Finds the permutation mapping a → b and b → a. Does not check species.

    Args:
        a (aseAtoms): First atoms object.
        b (aseAtoms): Second atoms object.
        spatial_tolerance (float): Position matching tolerance in Angstrom.

    Returns:
        dict with keys:
            - ``atoms_indices_a2b``: index array such that ``b = a[atoms_indices_a2b]``
            - ``atoms_indices_b2a``: index array such that ``a = b[atoms_indices_b2a]``
            - ``fail_reason``: string describing the failure, or None if successful
    """
    ret_dict = {
        "atoms_indices_a2b": None,
        "atoms_indices_b2a": None,
        "fail_reason": None,
    }
    st = spatial_tolerance
    if len(a) != len(b):
        ret_dict["fail_reason"] = f"len(a)={len(a)} != len(b)={len(b)}"
        return ret_dict
    if sorted(a.get_chemical_symbols()) != sorted(b.get_chemical_symbols()):
        ret_dict["fail_reason"] = "chemical_symbols mismatch"
        return ret_dict
    if not numpy.allclose(a.cell, b.cell, atol=st):
        ret_dict["fail_reason"] = "cell mismatch"
        return ret_dict
    a = a.copy()
    a.wrap()
    b = b.copy()
    b.wrap()
    atoms_dist = numpy.linalg.norm(a.positions[:, None, :] - b.positions[None, :, :], axis=2)  # shape (natoms, natoms)
    if numpy.sum(atoms_dist < st) != len(a):
        ret_dict["fail_reason"] = f"atoms positions mismatch, too few/many atoms' pairs with distance < {st}"
        return ret_dict
    ret_dict["atoms_indices_a2b"] = numpy.argmin(atoms_dist, axis=0)  # b = a[atoms_indices_a2b]
    ret_dict["atoms_indices_b2a"] = numpy.argmin(atoms_dist, axis=1)  # a = b[atoms_indices_b2a]
    return ret_dict


def atoms_ase2ph(atoms: aseAtoms) -> PhonopyAtoms:
    """Convert an ASE Atoms object to a PhonopyAtoms object.

    Args:
        atoms (aseAtoms): ASE Atoms object. Must have 3D PBC; a warning is printed if not.

    Returns:
        PhonopyAtoms: Equivalent Phonopy structure.
    """
    if not numpy.all(atoms.get_pbc()):
        print("WARNING: for PhonopyAtoms the pbc must be T T T. Set to T T T.")
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell().array,
        positions=atoms.get_positions(),
    )


def atoms_ph2ase(atoms: PhonopyAtoms) -> aseAtoms:
    """Convert a PhonopyAtoms object to an ASE Atoms object.

    Args:
        atoms (PhonopyAtoms): Phonopy structure.

    Returns:
        aseAtoms: Equivalent ASE structure with pbc=True.
    """
    return aseAtoms(
        symbols=atoms.symbols,
        cell=atoms.cell,
        positions=atoms.positions,
        pbc=True,
    )


def gaussian_function(
    x: numpy.ndarray,
    mu: int | numpy.ndarray = 0,
    sigma: float = 1e-2,
) -> numpy.ndarray:
    """Gaussian (normal) distribution function.

    Args:
        x (numpy.ndarray): Input array.
        mu (Union[int, numpy.ndarray]): Mean(s). If ndarray, output is a tensor product
            of shape ``(*mu.shape, *x.shape)``.
        sigma (float): Standard deviation.

    Returns:
        numpy.ndarray: Gaussian values.

    Raises:
        ValueError: If ``mu`` is neither int nor numpy.ndarray.
    """
    if isinstance(mu, int):
        pass
    elif isinstance(mu, numpy.ndarray):
        assert x.ndim == 1
        mu = mu[..., numpy.newaxis]
    else:
        raise ValueError("mu should be int or numpy.ndarray, but got " + str(type(mu)))
    return numpy.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * numpy.sqrt(2 * numpy.pi))


def band_expansion(
    energies: numpy.ndarray,
    grid: numpy.ndarray,
    sigma: float = 1e-2,
) -> numpy.ndarray:
    """Expand discrete band energies onto a grid via Gaussian broadening.

    Args:
        energies (numpy.ndarray): Band energies, shape ``(nbands,)``.
        grid (numpy.ndarray): Energy grid, shape ``(ngrid,)``.
        sigma (float): Gaussian broadening width (same units as energies).

    Returns:
        numpy.ndarray: Expanded values, shape ``(nbands, ngrid)``.
    """
    grid_delta_min = numpy.min(numpy.diff(grid))
    if grid_delta_min > sigma:
        print(f"Warning: grid delta is larger than sigma: {grid_delta_min:.3e} > {sigma:.3e}")
    return gaussian_function(grid, energies, sigma)


def concatenate_bands(
    kpts: list,
    connections: list,
) -> tuple[numpy.ndarray, list[int]]:
    """Merge Phonopy k-path segments and compute high-symmetry point indices.

    Takes the output of ``phonopy.phonon.band_structure.get_band_qpoints_and_path_connections``
    and removes duplicate boundary k-points where consecutive segments share an endpoint.

    Args:
        kpts (list[numpy.ndarray]): K-point arrays per segment, each shape ``(npts, 3)``.
        connections (list[bool]): ``connections[i]`` is True if segment ``i`` and ``i+1``
            share an endpoint.

    Returns:
        tuple:
            - **kpts_concat** (numpy.ndarray): Concatenated k-points, shape ``(N, 3)``.
            - **bz_label_indices** (list[int]): Indices into ``kpts_concat`` corresponding
              to the high-symmetry points (for tick marks in plots).
    """
    assert len(kpts) == len(connections)
    kpts_new = []
    for i in range(len(kpts)):
        if connections[i]:
            kpts_new.append(kpts[i][:-1])
        else:
            kpts_new.append(kpts[i])

    bz_label_indices = [0]
    next_seg_begin = 0
    for i in range(len(kpts)):
        if connections[i]:
            next_seg_begin += kpts[i].shape[0] - 1
            bz_label_indices.append(next_seg_begin)
        else:
            next_seg_begin += kpts[i].shape[0]
            if i != len(kpts) - 1:
                bz_label_indices.append(next_seg_begin - 1)
                bz_label_indices.append(next_seg_begin)
            else:
                bz_label_indices.append(next_seg_begin - 1)

    return numpy.concatenate(kpts_new, axis=0), bz_label_indices
