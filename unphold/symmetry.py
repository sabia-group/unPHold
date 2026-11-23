"""Threefold rotations of periodic structures and real-space force constants."""

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.harmonic.force_constants import compact_fc_to_full_fc, full_fc_to_compact_fc

from .utils import atoms_ph2ase


def rotmat_xy(angle: float) -> np.ndarray:
    """3×3 rotation matrix for rotation about the z-axis.

    Args:
        angle (float): Rotation angle in radians.

    Returns:
        np.ndarray: Rotation matrix, shape ``(3, 3)``.
    """
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def map_atoms_c3(
    atoms: Atoms,
    center: np.ndarray | None = None,
    center_frac: np.ndarray | None = None,
    lattice_atol=1e-5,
    position_atol=1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map each atom to its periodic image partner under a +120 degree rotation.

    Args:
        atoms (Atoms): Fully periodic structure with N atoms. Array indices refer to
            the input atom order. Rows of atoms.cell are lattice vectors in Å.
        center (np.ndarray, optional): Cartesian point on the rotation axis,
            shape (3,), in Å. The axis is parallel to +z.
        center_frac (np.ndarray, optional): Point on the same axis in fractional
            coordinates, shape (3,). Supply at most one center argument. If both
            are omitted, the axis passes through the origin.
        lattice_atol (float): Absolute dimensionless tolerance for rounding the
            fractional rotation matrix to integer entries. Defaults to 1e-5.
        position_atol (float): Upper bound on the Cartesian mismatch of each mapped atom,
            in Å. Defaults to 1e-3.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (perm, shift, cartesian_error).
            All three arrays are indexed by source atom i in the input order.

            perm is an integer array of shape (N,). If perm[i] == j, rotating
            source atom i lands on a periodic image of original atom j.
            For an array v with atom index on axis 0, v[perm][i] reads the
            target value v[j] for source i. To place source-indexed rotated
            values into target order, use out[perm] = rotated_values, so that
            out[j] receives rotated_values[i]. The inverse permutation is
            np.argsort(perm).

            shift is an integer array of shape (N, 3). Add shift[i] @ atoms.cell
            to atoms.positions[j] to obtain the target image for source i.

            cartesian_error is a float array of shape (N,), in Å. Entry i is
            the distance between the rotated position of i and that target image.

    Raises:
        ValueError: Invalid cell, coordinates, center or tolerances, or the
            rotation fails the element, threefold closure or distance checks.
    """
    if not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell).all():
        raise ValueError("Cell and positions must be finite.")
    if not len(atoms) or not np.all(atoms.pbc) or np.linalg.matrix_rank(atoms.cell) != 3:
        raise ValueError("Expected a nonempty, fully periodic 3D cell.")
    if not np.isfinite([lattice_atol, position_atol]).all() or min(lattice_atol, position_atol) <= 0:
        raise ValueError("Tolerances must be finite and positive.")
    R = rotmat_xy(2 * np.pi / 3)
    if center is not None and center_frac is not None:
        raise ValueError("Only one of center and center_frac should be provided.")
    if center is not None:
        center_frac = np.asarray(center, dtype=float) @ np.linalg.inv(atoms.cell)
    elif center_frac is not None:
        center_frac = np.array(center_frac, dtype=float)
    else:
        center_frac = np.array([0.0, 0.0, 0.0])
    if center_frac.shape != (3,) or not np.isfinite(center_frac).all():
        raise ValueError("Expected a finite rotation center with shape (3,).")
    # the rotational matrix of a commensurate symmetric operation should be integer in fractional coordinates
    R_frac_using_cell = atoms.cell @ R.T @ np.linalg.inv(atoms.cell)
    R_frac_op = np.rint(R_frac_using_cell).astype(int)
    lattice_error = np.max(np.abs(R_frac_using_cell - R_frac_op))
    if lattice_error > lattice_atol:
        raise ValueError(f"Fractional rotation mismatch {lattice_error:g} exceeds {lattice_atol:g}.")
    # after confirming the integer nature, we just use the R_frac_op to rotate the fractional coordinates
    frac_coord = atoms.get_scaled_positions(wrap=False)
    frac_coord_rot = (frac_coord - center_frac) @ R_frac_op + center_frac
    diff = frac_coord_rot[:, None, :] - frac_coord[None, :, :]  # shape (N, N, 3), first axis indexes rotated sites
    diff_wrapped = diff - np.rint(diff)  # wrap to [-0.5, 0.5]
    distances = np.linalg.norm(diff_wrapped @ atoms.cell, axis=-1)  # shape (N, N)
    distances[atoms.numbers[:, None] != atoms.numbers[None, :]] = np.inf
    perm = distances.argmin(axis=1)
    shift = np.rint(diff[np.arange(len(atoms)), perm]).astype(int)
    # check the mapping accuracy in cartesian coordinates
    center_cart = center_frac @ atoms.cell
    rotated_cart = (atoms.positions - center_cart) @ R.T + center_cart
    matched_cart = (frac_coord[perm] + shift) @ atoms.cell
    # Three rotations return every site to its original label.
    if not np.array_equal(perm[perm[perm]], np.arange(len(atoms))):
        raise ValueError("Element-preserving map does not close after three rotations.")
    cartesian_error = np.linalg.norm(rotated_cart - matched_cart, axis=1)
    if cartesian_error.max() >= position_atol:
        raise ValueError(
            f"Maximum same-element mismatch {cartesian_error.max():g} Å reaches tolerance {position_atol:g} Å."
        )
    return perm, shift, cartesian_error


def _rotate_fc(fc, perm, rotation):
    """Actively rotate each block and write it into target atom-pair order."""
    rotated = np.empty_like(fc)
    rotated[perm[:, None], perm[None, :]] = rotation @ fc @ rotation.T
    return rotated


def symmetrize_fc_c3(
    ph: Phonopy,
    center_frac_primitive=None,
    lattice_atol=1e-5,
    position_atol=1e-3,
) -> np.ndarray:
    r"""Average real-space force constants over the $C_3$ rotations about $z$.

    The Phonopy object supplies the primitive lattice and supercell atom
    order. Let $\Phi_{ij}$ denote the block fc[i, j] and $p(i)$ the target site
    perm[i]. A displacement $\mathbf{u}_j$ contributes the force
    $\mathbf{F}_i=-\Phi_{ij}\mathbf{u}_j$ on site $i$.
    Under the Cartesian rotation $R$ by $+120^\circ$ about $+z$,

    $$
    R\mathbf{F}_i
    =-\Phi^{\mathrm{rot}}_{p(i),p(j)}(R\mathbf{u}_j).
    $$

    Since this holds for every $\mathbf{u}_j$ and $R^{-1}=R^\mathsf{T}$,

    $$
    \Phi^{\mathrm{rot}}_{p(i),p(j)}=R\Phi_{ij}R^\mathsf{T}.
    $$

    The destination indices $p(i),p(j)$ select the rotated atom pair. The two
    matrix products rotate the force and displacement components of its block.

    Args:
        ph (Phonopy): Structure and its real full or compact force constants.
            Compact input is expanded to full size in memory using translations.
        center_frac_primitive (array-like, optional): Point on the rotation
            axis in fractional coordinates of ph.primitive.cell, shape (3,).
            Defaults to the origin. The mapping uses the corresponding Cartesian
            point as the rotation center in the supercell.

        lattice_atol (float): Absolute dimensionless lattice-operation tolerance.
        position_atol (float): Maximum allowed site mismatch in angstroms.

    Returns:
        np.ndarray: Elementwise average of the original, once-rotated and
        twice-rotated tensors, as float64 in the input shape, atom order, and units.

    Raises:
        ValueError: Force constants are missing, malformed or nonfinite, or the
            requested center and tolerances do not give a valid C3 mapping.
        RuntimeError: The averaged force constants fail the rotation check.
    """
    fc = np.asarray(ph.force_constants)
    atoms_sc = atoms_ph2ase(ph.supercell)
    n_sc = len(atoms_sc)
    full_shape = (n_sc, n_sc, 3, 3)
    compact_shape = (len(ph.primitive), n_sc, 3, 3)
    if fc.shape not in (full_shape, compact_shape) or fc.dtype.kind not in "iuf":
        raise ValueError(f"Expected real FCs of shape {full_shape} or {compact_shape}, got {fc.shape}, {fc.dtype}.")
    fc = np.asarray(fc, dtype=np.float64)
    if not np.isfinite(fc).all():
        raise ValueError("Force constants must be finite.")
    compact = fc.shape != full_shape
    if compact:
        fc = compact_fc_to_full_fc(ph.primitive, fc)

    center_cart = None
    if center_frac_primitive is not None:
        center_cart = np.asarray(center_frac_primitive, dtype=float) @ ph.primitive.cell
    perm, _, _ = map_atoms_c3(atoms_sc, center=center_cart, lattice_atol=lattice_atol, position_atol=position_atol)
    R = rotmat_xy(2 * np.pi / 3)

    fc_c3_1 = _rotate_fc(fc, perm, R)
    fc_c3_2 = _rotate_fc(fc_c3_1, perm, R)
    fc_c3_sym = (fc + fc_c3_1 + fc_c3_2) / 3

    residual = np.max(np.abs(_rotate_fc(fc_c3_sym, perm, R) - fc_c3_sym))
    if residual > 1e-12 * np.max(np.abs(fc)):
        raise RuntimeError(f"C3 invariance check failed: maximum residual {residual:g}.")
    if compact:
        return full_fc_to_compact_fc(ph.primitive, fc_c3_sym)
    return fc_c3_sym
