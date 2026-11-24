"""Functions for processing layer-mode eigenvectors and phonon characterization."""

import numpy as np
from ase import Atoms as aseAtoms


def layer_plane_wave_basis(
    atoms: aseAtoms,
    layers: np.ndarray,
    q_frac: list[float] | np.ndarray,
    g_frac: list[int] | np.ndarray = (0, 0, 0),
    gauge="atomic",
    normal_z=(0, 0, 1),
    longitudinal_direction=None,
):
    r"""Build six mass-weighted plane-wave patterns for one harmonic q + G.

    Args:
        atoms: ASE primitive cell in the eigenvector atom order.
        layers: Integer array of shape (2, N_layer) containing zero-based atom
            indices. Rows 0 and 1 define layers 1 and 2 in (b1 - b2)/sqrt(2).
            Each atom must appear once. The layers must have equal total mass.
        q_frac: Phonon wavevector in fractional reciprocal coordinates.
        g_frac: Integer reciprocal-lattice offset in the same basis as q_frac.
        gauge: "atomic" uses the phase $\exp(2 \pi i G \cdot r)$, and "cell" uses
            $\exp(2 \pi i (q+G) \cdot r)$, with Cartesian wavevectors in cycles/angstrom.
        normal_z: Cartesian layer normal defining the positive Z direction.
        longitudinal_direction: Cartesian direction of approach when q+G has
            zero in-plane component. Its in-plane projection defines L, and
            T = Z × L. At finite in-plane wavevector, q+G itself defines L.

    Returns:
        Complex (3*N, 6) array with orthonormal columns sL, sT, sZ, aL, aT, aZ.
        Rows 3*j through 3*j+2 are the Cartesian components of atom j.

    Raises:
        ValueError: If the layer partition, wavevectors, masses, gauge, or
            polarization directions are invalid.
    """

    def vector(value, name):
        value = np.asarray(value, dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite three-component vector")
        return value

    if not isinstance(layers, np.ndarray) or layers.ndim != 2 or layers.shape[0] != 2:
        raise ValueError("layers must be an ndarray of shape (2, N_layer)")
    if layers.shape[1] == 0 or layers.dtype.kind not in "iu":
        raise ValueError("Each layer must contain integer atom indices")
    if not np.array_equal(np.sort(layers.ravel()), np.arange(len(atoms))):
        raise ValueError("Each atom must belong to exactly one layer")

    # N is the total number of atoms. N_layer is the number in one layer.
    reciprocal = atoms.cell.reciprocal()  # (3, 3)
    q = vector(q_frac, "q_frac") @ reciprocal  # (3,)
    offset = vector(g_frac, "g_frac")
    if not np.allclose(offset, np.rint(offset), atol=1e-8, rtol=0):
        raise ValueError("g_frac must be an integer reciprocal-lattice offset")
    g = np.rint(offset) @ reciprocal  # (3,)
    masses = np.asarray(atoms.get_masses(), dtype=float)  # (N,)
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("atomic masses must be finite and positive")
    totals = masses[layers].sum(axis=1)  # (2,), one mass per layer
    if not np.isclose(*totals, rtol=1e-8, atol=0):
        raise ValueError("equal total layer masses are required for physical +/- templates")
    if gauge not in ("atomic", "cell"):
        raise ValueError("gauge must be 'atomic' or 'cell'")
    positions = np.asarray(atoms.positions)  # (N, 3), atom × xyz
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")
    z = vector(normal_z, "normal_z")
    if np.linalg.norm(z) == 0:
        raise ValueError("normal_z must be nonzero")
    z = z / np.linalg.norm(z)
    k = q + g
    k_inplane = k - (k @ z) * z
    if np.linalg.norm(k_inplane) < 1e-12:
        if longitudinal_direction is None:
            raise ValueError("Specify longitudinal_direction at zero in-plane wavevector")
        direction = vector(longitudinal_direction, "longitudinal_direction")
        k_inplane = direction - (direction @ z) * z
        if np.linalg.norm(k_inplane) < 1e-12:
            raise ValueError("longitudinal_direction must have an in-plane component")
    longitudinal = k_inplane / np.linalg.norm(k_inplane)  # (3,)
    transverse = np.cross(z, longitudinal)  # (3,)
    phase_vector = g if gauge == "atomic" else q + g  # (3,)
    phase = np.exp(2j * np.pi * (positions @ phase_vector))  # (N,), one phase per atom

    # Each column is one polarization p. Each row triplet is one atom's xyz.
    def single_layer_basis(atom_indices, layer_mass):  # atom_indices (N_layer,), layer_mass scalar
        basis = np.zeros((3 * len(atoms), 3), dtype=complex)  # (3*N, 3), or (xyz, LTZ)
        amplitude = np.sqrt(masses[atom_indices] / layer_mass) * phase[atom_indices]  # (N_layer,)
        for column, polarization in enumerate((longitudinal, transverse, z)):
            displacement = np.zeros((len(atoms), 3), dtype=complex)  # (N, 3), or (atom, xyz)
            displacement[atom_indices] = amplitude[:, None] * polarization  # (N_layer, 1) * (3,) → (N_layer, 3)
            basis[:, column] = displacement.ravel()  # (3*N,), N first and then 3 from xyz
        return basis

    b1 = single_layer_basis(layers[0], totals[0])  # (3*N, 3)
    b2 = single_layer_basis(layers[1], totals[1])  # (3*N, 3)
    symmetric = (b1 + b2) / np.sqrt(2)  # (3*N, 3), sL, sT, sZ
    antisymmetric = (b1 - b2) / np.sqrt(2)  # (3*N, 3), aL, aT, aZ
    return np.column_stack((symmetric, antisymmetric))  # (3*N, 6), sL, sT, sZ, aL, aT, aZ


def characterize_multi_g(
    atoms: aseAtoms,
    layers: np.ndarray,
    eigenvectors: np.ndarray,
    q_frac: list[float] | np.ndarray,
    g_frac: list[list[int]] | np.ndarray,
    gauge="atomic",
    normal_z=(0, 0, 1),
    longitudinal_direction=None,
    rtol=1e-8,
):
    """Measure six layer and polarization weights across multiple G offsets for each phonon band.

    Args:
        atoms: ASE primitive cell in Phonopy atom order.
        layers: Integer atom indices of shape (2, N_layer).
        eigenvectors: Normalized mass-weighted mode columns, shape (3*N, n_bands).
        q_frac: Phonon wavevector in fractional reciprocal coordinates, shape (3,).
        g_frac: Integer reciprocal offsets, one per row, shape (n_G, 3).
        gauge: Phase convention of the eigenvectors ("atomic" or "cell").
        normal_z: Cartesian layer normal, shape (3,).
        longitudinal_direction: Optional Cartesian L axis at zero in-plane wavevector.
        rtol: Relative singular-value cutoff for the span of the templates.

    Returns:
        Dictionary with channel order sL, sT, sZ, aL, aT, aZ.
        amplitudes[G, channel, band] is the complex overlap with one template.
        weights_by_g[G, channel, band] is its squared magnitude.
        weights[channel, band] is the projection weight in the span of that
        channel's templates. For orthonormal templates it equals the sum over G.
        coverage[band] is the projection weight in the span of all templates.
    """
    g_frac = np.asarray(g_frac)
    if g_frac.ndim != 2 or g_frac.shape[1] != 3 or len(g_frac) == 0:
        raise ValueError("g_frac must have shape (n_G, 3) with at least one offset")
    eigenvectors = np.asarray(eigenvectors)
    if eigenvectors.ndim != 2 or eigenvectors.shape[0] != 3 * len(atoms):
        raise ValueError("eigenvectors must have shape (3*N, n_bands)")
    if not np.allclose(np.linalg.norm(eigenvectors, axis=0), 1, atol=1e-8, rtol=0):
        raise ValueError("Each eigenvector column must have unit norm")
    if not 0 < rtol < 1:
        raise ValueError("rtol must be between zero and one")

    # One G at a time: project every band onto its six reference columns.
    bases = []
    amplitudes = []
    for offset in g_frac:
        basis = layer_plane_wave_basis(
            atoms,
            layers,
            q_frac,
            offset,
            gauge=gauge,
            normal_z=normal_z,
            longitudinal_direction=longitudinal_direction,
        )  # (3*N, 6)
        bases.append(basis)
        amplitudes.append(basis.conj().T @ eigenvectors)  # (6, n_bands)
    amplitudes = np.stack(amplitudes)  # (n_G, 6, n_bands)
    weights_by_g = np.abs(amplitudes) ** 2

    # Count each independent displacement direction once in a joint projection.
    def weight_in_span(templates):
        directions, singular_values, _ = np.linalg.svd(templates, full_matrices=False)
        keep = singular_values > (rtol * singular_values[0])
        overlaps = directions[:, keep].conj().T @ eigenvectors  # (rank, n_bands)
        return np.sum(np.abs(overlaps) ** 2, axis=0)  # (n_bands,)

    weights = []
    for channel in range(6):
        templates = np.column_stack([basis[:, channel] for basis in bases])  # (3*N, n_G)
        weights.append(weight_in_span(templates))
    weights = np.stack(weights)  # (6, n_bands)
    coverage = weight_in_span(np.column_stack(bases))  # basis shape (3*N, 6*n_G)

    return {
        "amplitudes": amplitudes,  # shape (n_G, 6, n_bands), 6 for sL, sT, sZ, aL, aT, aZ
        "weights_by_g": weights_by_g,  # shape (n_G, 6, n_bands)
        "weights": weights,  # shape (6, n_bands)
        "coverage": coverage,  # shape (n_bands,), coverage in the spann of all G input by g_frac
    }
