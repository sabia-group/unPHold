"""Internal atom-matching utilities.

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
    atoms_dist = numpy.linalg.norm(
        a.positions[:, None, :] - b.positions[None, :, :], axis=2
    )  # shape (natoms, natoms)
    if numpy.sum(atoms_dist < st) != len(a):
        ret_dict["fail_reason"] = (
            f"atoms positions mismatch, too few/many atoms' pairs with distance < {st}"
        )
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