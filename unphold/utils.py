"""Internal utilities for atom matching and structure conversion.

These are implementation details used by the Unfold class.
They are not part of the public API and should not be imported directly by users.
"""

import numpy
from ase.atoms import Atoms as aseAtoms
from phonopy.structure.atoms import PhonopyAtoms


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


def match_two_atoms(
    a: aseAtoms,
    b: aseAtoms,
    spatial_tolerance: float = 1e-2,
) -> dict:
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


def match_two_2d_atoms_pbc_with_2d_frac_shift(
    a: aseAtoms,
    b: aseAtoms,
    shift_0_frac: float = 0.01,
    shift_0_seg: int = 11,
    shift_1_frac: float = 0.01,
    shift_1_seg: int = 11,
    spatial_tolerance: float = 1e-2,
    tolerance_xyz_scaler: numpy.ndarray | None = None,
    ignore_z: bool = True,
) -> dict:
    """Match atoms between two 2D-periodic ASE Atoms objects, searching over in-plane fractional shifts.

    Brute-force searches a grid of in-plane fractional shifts of ``b`` (spanned by
    ``shift_0_frac``/``shift_0_seg`` along lattice vector 0 and ``shift_1_frac``/``shift_1_seg``
    along lattice vector 1) and keeps the first shift for which every atom in ``a`` has a unique,
    species-matching neighbor in the shifted-and-wrapped ``b`` within ``spatial_tolerance``.
    Unlike [`match_two_atoms`][unphold.utils.match_two_atoms], this tolerates a rigid in-plane
    misalignment between the two structures (e.g. from moire relaxation) by scanning shifts instead
    of requiring cells to already coincide.

    Args:
        a (aseAtoms): First atoms object, 2D-periodic (PBC in xy, non-periodic/vacuum along z).
        b (aseAtoms): Second atoms object, same convention as ``a``.
        shift_0_frac (float): Half-width of the shift search range along lattice vector 0,
            in fractional coordinates.
        shift_0_seg (int): Number of shift samples along lattice vector 0 (linspace endpoints inclusive).
        shift_1_frac (float): Half-width of the shift search range along lattice vector 1,
            in fractional coordinates.
        shift_1_seg (int): Number of shift samples along lattice vector 1 (linspace endpoints inclusive).
        spatial_tolerance (float): Position-matching tolerance in Angstrom (after scaling by
            ``tolerance_xyz_scaler``).
        tolerance_xyz_scaler (numpy.ndarray, optional): Per-axis scaling applied to the real-space
            distance before comparing against ``spatial_tolerance``, shape ``(3,)``. Defaults to
            ``[1.0, 1.0, 1.0]`` (no scaling) if not given.
        ignore_z (bool): If True, only xy coordinates are used for matching; z is aligned separately
            by shifting ``b`` so the mean z-coordinates of ``a`` and ``b`` coincide.

    Returns:
        dict: On success (a shift is found), contains:

            - **shift_position** (numpy.ndarray): In-plane fractional shift plus z-shift applied
              to ``b``, shape ``(3,)``.
            - **atoms_indices_a2b** (numpy.ndarray): Index array such that ``b = a[atoms_indices_a2b]``.
            - **atoms_indices_b2a** (numpy.ndarray): Index array such that ``a = b[atoms_indices_b2a]``.
            - **a_wrapped** (aseAtoms): ``a``, wrapped but not shifted.
            - **b_shifted_wrapped** (aseAtoms): ``b``, shifted and wrapped to match ``a``.
            - **atoms_dist_in_real_space** (numpy.ndarray): Per-atom distance vectors in real space.
            - **atoms_dist_matched** (numpy.ndarray): Scaled distances between matched atoms.
            - **atoms_dist_list** (numpy.ndarray): Number of matched pairs at each tried shift.
            - **number_of_attempts** (int): Number of shifts tried before a match was found.

            On failure (no shift matched), contains only **atoms_dist_list** with the match counts
            attempted at every shift, to help diagnose the tolerance/search-range settings.

    Raises:
        AssertionError: If ``a`` and ``b`` have different lengths, lack z-direction PBC, or their
            cells are inconsistent with a 2D-periodic (xy) layer.
    """
    if tolerance_xyz_scaler is None:
        tolerance_xyz_scaler = numpy.array([1.0, 1.0, 1.0])
    a = a.copy()  # make sure we do not modify the original objects
    b = b.copy()
    assert len(a) == len(b), "Both Atoms objects must have the same number of atoms."
    assert a.pbc[2] and b.pbc[2], "Both Atoms objects must have PBC in the z direction."
    assert numpy.allclose(a.cell[:2], b.cell[:2], rtol=1e-5), "Cells in the xy plane must match."
    assert numpy.all(numpy.abs(a.cell[2, :2]) <= 1e-6) and numpy.all(numpy.abs(b.cell[2, :2]) <= 1e-6), (
        "Cells lattice c should have zero components in xy plane."
    )
    assert numpy.all(numpy.abs(a.cell[:2, 2]) <= 1e-6) and numpy.all(numpy.abs(b.cell[:2, 2]) <= 1e-6), (
        "Cells lattice ab should have zero components in z direction."
    )
    a.cell[2, 2], b.cell[2, 2] = 100.0, 100.0  # set a large value for the z-coordinate to avoid PBC issues

    if ignore_z:
        # calculate the minimum distance in xy plane
        atoms_dist_xy = numpy.linalg.norm(a.positions[:, None, :2] - b.positions[None, :, :2], axis=2)
        min_xy_dist = numpy.min(atoms_dist_xy)
        if min_xy_dist < spatial_tolerance:
            print(
                f"Warning: minimum distance in xy plane is {min_xy_dist:.3f} < spatial_tolerance={spatial_tolerance}, "
                "forced ignore_z=True may lead to wrong matching!"
            )

    z_shift_b2a = numpy.mean(a.positions[:, 2]) - numpy.mean(b.positions[:, 2])
    b.positions[:, 2] += z_shift_b2a  # align z-coordinates by mean value
    cell_avg = (a.cell + b.cell) / 2.0  # average cell
    shifts_frac = numpy.array(
        [
            [frac0, frac1, 0.0]
            for frac0 in numpy.linspace(-shift_0_frac, shift_0_frac, shift_0_seg)
            for frac1 in numpy.linspace(-shift_1_frac, shift_1_frac, shift_1_seg)
        ]
    )

    # then try to match atoms with wrapping and shifts
    a.wrap()
    b.wrap()
    if_matched_spatially = False
    b_shifted_list = []
    atoms_dist_list = []
    for this_idx, this_shift_realspace in enumerate(shifts_frac):
        this_b = b.copy()
        this_b_fracpos = this_b.get_scaled_positions()  # get fractional coordinates
        this_b_fracpos += this_shift_realspace  # apply the shift in fractional coordinates
        this_b.set_scaled_positions(this_b_fracpos)  # set the shifted fractional coordinates
        this_b.wrap()  # wrap the positions to the unit cell
        b_shifted_list.append(this_b)
        this_b_fracpos = this_b.get_scaled_positions()  # get the wrapped fractional coordinates
        fracpos_diff = a.get_scaled_positions()[:, None, :] - this_b_fracpos[None, :, :]  # shape (natoms, natoms, 3)
        fracpos_diff = fractional_part_around_zero(fracpos_diff)  # map to [-0.5, 0.5)
        if ignore_z:  # only compare xy coordinates
            realpos_diff = numpy.einsum(  # shape (natoms, natoms, 2)
                "abx,xy->aby", fracpos_diff[:, :, :2], cell_avg[:2, :2]
            )
            realpos_diff /= tolerance_xyz_scaler[:2].reshape(1, 1, 2)  # scale by tolerance
            atoms_dist = numpy.linalg.norm(realpos_diff, axis=2)  # shape (natoms, natoms)
        else:  # compare all coordinates
            realpos_diff = numpy.einsum(  # shape (natoms, natoms, 3)
                "abx,xy->aby", fracpos_diff, cell_avg
            )
            realpos_diff /= tolerance_xyz_scaler.reshape(1, 1, 3)  # scale by tolerance
            atoms_dist = numpy.linalg.norm(realpos_diff, axis=2)  # shape (natoms, natoms)
        atoms_dist_in_tolerance = numpy.sum(atoms_dist < spatial_tolerance)
        atoms_dist_list.append(atoms_dist_in_tolerance)
        atoms_indices_a2b = numpy.argmin(atoms_dist, axis=0)  # b = a[atoms_indices_a2b]
        atoms_indices_b2a = numpy.argmin(atoms_dist, axis=1)  # a = b[atoms_indices_b2a]
        atoms_dist_in_real_space = (
            a.get_scaled_positions() - this_b.get_scaled_positions()[atoms_indices_b2a]
        )  # shape (natoms, 3)
        atoms_dist_in_real_space = fractional_part_around_zero(atoms_dist_in_real_space)  # map to [-0.5, 0.5)
        atoms_dist_in_real_space = atoms_dist_in_real_space @ cell_avg  # convert to real space
        # print(numpy.sum(atoms_dist < spatial_tolerance), len(a))  # for debug

        ### check if matched spatially, without checking species
        # a bad candidate shift must not abort the search — just try the next one
        if_matched_spatially = True
        # step 1: check atoms' pairs within range
        if if_matched_spatially:
            if atoms_dist_in_tolerance != len(a):
                if_matched_spatially = False
        # step 2: check if unique mapping from a to b
        if if_matched_spatially:
            if len(numpy.unique(numpy.argmin(atoms_dist, axis=0))) != len(a):
                if_matched_spatially = False
        # step 3: check if atomic species match
        if if_matched_spatially:
            if numpy.all(
                numpy.array(a.get_chemical_symbols())[atoms_indices_a2b] == numpy.array(b.get_chemical_symbols())
            ):
                pass
            else:
                if_matched_spatially = False
                print(
                    "Atomic species do not match, please check the structures visually first, "
                    "or reduce spatial_tolerance."
                )

        if if_matched_spatially:
            # found a match, return the shift and indices
            atoms_dist_matched = atoms_dist[atoms_dist < spatial_tolerance]  # shape (natoms, )
            assert len(atoms_dist_matched) == len(a), (
                f"len(atoms_dist_matched)={len(atoms_dist_matched)} != len(a)={len(a)}, "
                "please check the structures visually first, shift the two structures to a nice starting position."
            )
            ret_dict = {
                "shift_position": this_shift_realspace + numpy.array([0.0, 0.0, z_shift_b2a]),  # shift in from b to a
                "atoms_indices_a2b": atoms_indices_a2b,  # b = a[atoms_indices_a2b]
                "atoms_indices_b2a": atoms_indices_b2a,  # a = b[atoms_indices_b2a]
                "a_wrapped": a.copy(),  # just wrapped, not shifted
                "b_shifted_wrapped": this_b.copy(),  # shifted and wrapped
                "atoms_dist_in_real_space": atoms_dist_in_real_space,  # distance in real space
                "atoms_dist_matched": atoms_dist_matched,  # distance between matched atoms, this is SCALED!
                "atoms_dist_list": numpy.array(atoms_dist_list),  # list of matched atoms for each shift
                "number_of_attempts": this_idx + 1,  # number of attempts to match
            }

            break
        else:
            ret_dict = {
                "atoms_dist_list": numpy.array(atoms_dist_list),  # list of distances for each shift
            }

    if not if_matched_spatially:  # if didn't match, provide suggestions
        print(
            "No match found, please check the structures visually first, shift the two structures to a nice "
            "starting position, and consider tuning spatial_tolerance."
        )
    return ret_dict


def fractional_part_around_zero(arr: numpy.ndarray) -> numpy.ndarray:
    """Convert fractional parts of array elements to range [-0.5, 0.5).

    This function takes the fractional part of each element in the input array
    and maps it to the range [-0.5, 0.5) by subtracting 1.0 from fractional
    parts that are >= 0.5.

    Args:
        arr: Input array of numeric values.

    Returns:
        numpy.ndarray: Array with fractional parts mapped to the range [-0.5, 0.5).

    Examples:
        >>> import numpy
        >>> arr = numpy.array([-1.5, -0.5, 0.5, 1.5, 2.5])
        >>> eps = 1e-15  # float64 minimum precision is about 2.22e-16
        >>> fractional_part_around_zero(arr), \
                fractional_part_around_zero(arr - eps)
        (array([-0.5, -0.5, -0.5, -0.5, -0.5]), array([0.5, 0.5, 0.5, 0.5, 0.5]))
    """
    # return arr - numpy.round(arr)  # this is wrong, because
    # "For values exactly halfway between rounded decimal values, NumPy rounds to the nearest even value.
    # Thus 1.5 and 2.5 round to 2.0, -0.5 and 0.5 round to 0.0, etc."
    return numpy.where((arr % 1.0) >= 0.5, (arr % 1.0) - 1.0, arr % 1.0)


class RelaxBySpring:
    """Relax z-direction waviness in 2D moire materials using spring forces.

    Smooths out atomic positions in layered 2D materials by installing spring-like restoring
    forces and running explicit-Euler energy minimization, particularly useful for moire
    superlattices where interlayer interactions cause unwanted z-direction corrugation.

    Three types of springs can be installed:

    - Origin springs: restoring forces towards each atom's original position.
    - Layer springs: forces pulling atoms of a given species towards a target z-coordinate.
    - Neighbor springs: harmonic interactions between nearby atoms, using PBC minimum image
      convention.

    All operations are vectorized and handle periodic boundary conditions in all three
    directions using fractional coordinates.

    Warning:
        This class was written by an LLM and should be tested more carefully before being
        relied on in production.

    Args:
        atoms (aseAtoms): The atomic structure to be relaxed. Must have periodic boundary
            conditions.

    Example::

        relax = RelaxBySpring(atoms)
        relax.install_origin_spring(k=0.1)
        relax.install_layer_spring_by_species("C", k=1.0)
        relax.install_neighbour_spring(k=0.1, r_cutoff=5.0)
        relax.relax(steps=100, delta=0.01)
        relaxed_atoms = relax.atoms
    """

    def __init__(self, atoms: aseAtoms):
        self._atoms = atoms.copy()
        self._pos_org = self._atoms.positions.copy()
        self._pos = self._atoms.positions.copy()

        # Store different types of springs as numpy arrays for efficient vectorized operations
        # Initialize with empty arrays of appropriate dtype
        self._origin_springs = {
            "indices": numpy.array([], dtype=int),
            "k_values": numpy.array([], dtype=float),
            "target_positions": numpy.empty((0, 3), dtype=float),
        }
        self._layer_springs = {
            "indices": numpy.array([], dtype=int),
            "k_values": numpy.array([], dtype=float),
            "z_targets": numpy.array([], dtype=float),
        }
        self._neighbor_springs = {
            "pairs": numpy.empty((0, 2), dtype=int),
            "k_values": numpy.array([], dtype=float),
            "r0_values": numpy.array([], dtype=float),
        }

    def install_origin_spring(self, k: float = 0.1) -> None:
        """Install springs between each atom and its original position.

        Args:
            k (float): Spring constant for origin springs. Higher values create stronger
                restoring forces towards original positions.
        """
        n_atoms = len(self._atoms)
        self._origin_springs["indices"] = numpy.arange(n_atoms, dtype=int)
        self._origin_springs["k_values"] = numpy.full(n_atoms, k, dtype=float)
        self._origin_springs["target_positions"] = self._pos_org.copy()

    def install_layer_spring_by_species(
        self,
        species: str,
        k: float = 1.0,
        z_target: float | None = None,
    ) -> None:
        """Install springs between each atom of a given species and a target z-coordinate.

        Args:
            species (str): Chemical species to apply springs to, e.g. ``"C"``, ``"Mo"``, ``"W"``.
            k (float): Spring constant for layer springs. Controls strength of z-direction alignment.
            z_target (float, optional): Target z-coordinate for the species. If None, uses the
                mean z-coordinate of atoms of this species in the original structure.
        """
        symbols = self._atoms.get_chemical_symbols()
        species_indices = numpy.array([i for i, sym in enumerate(symbols) if sym == species], dtype=int)

        if len(species_indices) == 0:
            print(f"Warning: No atoms of species '{species}' found.")
            return

        if z_target is None:
            z_target = numpy.mean(self._pos_org[species_indices, 2])

        # Concatenate new springs to existing arrays
        self._layer_springs["indices"] = numpy.concatenate([self._layer_springs["indices"], species_indices])
        self._layer_springs["k_values"] = numpy.concatenate(
            [self._layer_springs["k_values"], numpy.full(len(species_indices), k, dtype=float)]
        )
        self._layer_springs["z_targets"] = numpy.concatenate(
            [self._layer_springs["z_targets"], numpy.full(len(species_indices), z_target, dtype=float)]
        )

    def install_neighbour_spring(
        self,
        k: float = 0.1,
        k_scale_by_r: str | None = None,
        r_cutoff: float = 5.0,
        fix_distance: bool = True,
    ) -> None:
        """Install springs between neighbouring atoms with periodic boundary conditions.

        The spring force follows Hooke's law with optional distance-dependent scaling.
        All pairwise distances are computed using minimum image convention for PBC.

        Args:
            k (float): Base spring constant for neighbor interactions.
            k_scale_by_r (str, optional): Distance scaling for the spring constant. One of:

                - `None`: constant spring constant ``k``.
                - `"1/r"`: spring constant scales as ``k / r``.
                - `"1/r^2"`: spring constant scales as ``k / r**2``.

            r_cutoff (float): Maximum distance for neighbor detection, in Angstrom.
            fix_distance (bool): If True, the equilibrium distance is set to the original
                interatomic distance. If False, uses the current distance as equilibrium.

        Raises:
            AssertionError: If ``k_scale_by_r`` is not one of the allowed options, or if
                ``r_cutoff``/``k`` is not positive.
        """
        assert k_scale_by_r in [None, "1/r", "1/r^2"], "k_scale_by_r must be None, '1/r' or '1/r^2'"
        assert r_cutoff > 0, "r_cutoff must be positive"
        assert k > 0, "k must be positive"

        # Use current or original positions for reference distances
        ref_pos = self._pos_org if fix_distance else self._pos
        n_atoms = len(self._atoms)
        cell = self._atoms.cell

        # Vectorized approach for finding neighbors
        # Create all pairs indices
        i_indices, j_indices = numpy.triu_indices(n_atoms, k=1)

        # Calculate all pairwise vectors
        dr_vectors = ref_pos[j_indices] - ref_pos[i_indices]  # shape (n_pairs, 3)

        # Convert to fractional coordinates for PBC
        # Using broadcasting-friendly approach
        cell_inv = numpy.linalg.inv(cell.T)
        dr_frac = dr_vectors @ cell_inv.T  # shape (n_pairs, 3)

        # Apply minimum image convention
        dr_frac = fractional_part_around_zero(dr_frac)

        # Convert back to cartesian coordinates
        dr_cart = dr_frac @ cell  # shape (n_pairs, 3)

        # Calculate distances
        distances = numpy.linalg.norm(dr_cart, axis=1)  # shape (n_pairs,)

        # Find neighbors within cutoff
        neighbor_mask = distances < r_cutoff
        neighbor_indices = numpy.where(neighbor_mask)[0]

        if len(neighbor_indices) > 0:
            # Get the relevant pairs and distances
            i_neighbors = i_indices[neighbor_indices]
            j_neighbors = j_indices[neighbor_indices]
            neighbor_distances = distances[neighbor_indices]

            # Calculate spring constants with scaling
            k_values = numpy.full(len(neighbor_indices), k, dtype=float)
            if k_scale_by_r == "1/r":
                k_values = k / neighbor_distances
            elif k_scale_by_r == "1/r^2":
                k_values = k / (neighbor_distances**2)

            # Store neighbor springs using numpy concatenate
            new_pairs = numpy.column_stack([i_neighbors, j_neighbors])
            self._neighbor_springs["pairs"] = (
                numpy.concatenate([self._neighbor_springs["pairs"], new_pairs])
                if len(self._neighbor_springs["pairs"]) > 0
                else new_pairs
            )

            self._neighbor_springs["k_values"] = numpy.concatenate([self._neighbor_springs["k_values"], k_values])
            self._neighbor_springs["r0_values"] = numpy.concatenate(
                [self._neighbor_springs["r0_values"], neighbor_distances]
            )

    def relax(self, steps: int = 100, delta: float = 0.01) -> None:
        """Perform spring-based relaxation to minimize system energy.

        Uses explicit Euler integration with Hooke's law forces from all installed springs.
        Force calculation is fully vectorized and handles periodic boundary conditions.

        Args:
            steps (int): Number of relaxation steps to perform.
            delta (float): Integration time step. Position updates are proportional to
                force × delta. Smaller values give more stable integration but need more steps.
        """
        cell = self._atoms.cell
        cell_inv = numpy.linalg.inv(cell.T)

        for _step in range(steps):
            forces = numpy.zeros_like(self._pos)

            # Origin springs - vectorized
            if len(self._origin_springs["indices"]) > 0:
                indices = self._origin_springs["indices"]
                k_vals = self._origin_springs["k_values"]
                targets = self._origin_springs["target_positions"]

                dr = targets - self._pos[indices]  # shape (n_springs, 3)
                spring_forces = k_vals[:, numpy.newaxis] * dr  # shape (n_springs, 3)

                # Add forces to atoms
                numpy.add.at(forces, indices, spring_forces)

            # Layer springs - vectorized
            if len(self._layer_springs["indices"]) > 0:
                indices = self._layer_springs["indices"]
                k_vals = self._layer_springs["k_values"]
                z_targets = self._layer_springs["z_targets"]

                dz = z_targets - self._pos[indices, 2]  # shape (n_springs,)
                spring_forces_z = k_vals * dz  # shape (n_springs,)

                # Add z-forces to atoms
                numpy.add.at(forces[:, 2], indices, spring_forces_z)

            # Neighbor springs - vectorized where possible
            if len(self._neighbor_springs["pairs"]) > 0:
                pairs = self._neighbor_springs["pairs"]
                k_vals = self._neighbor_springs["k_values"]
                r0_vals = self._neighbor_springs["r0_values"]

                i_atoms = pairs[:, 0]
                j_atoms = pairs[:, 1]

                # Calculate current distances with PBC
                dr_vectors = self._pos[j_atoms] - self._pos[i_atoms]  # shape (n_pairs, 3)
                dr_frac = dr_vectors @ cell_inv.T  # shape (n_pairs, 3)
                dr_frac = fractional_part_around_zero(dr_frac)
                dr_cart = dr_frac @ cell  # shape (n_pairs, 3)

                distances = numpy.linalg.norm(dr_cart, axis=1)  # shape (n_pairs,)

                # Avoid division by zero
                valid_mask = distances > 1e-10
                if numpy.any(valid_mask):
                    valid_indices = numpy.where(valid_mask)[0]

                    # Calculate spring forces for valid pairs
                    force_magnitudes = k_vals[valid_indices] * (distances[valid_indices] - r0_vals[valid_indices])
                    force_directions = dr_cart[valid_indices] / distances[valid_indices, numpy.newaxis]
                    spring_forces = force_magnitudes[:, numpy.newaxis] * force_directions

                    # Apply forces (Newton's third law)
                    valid_i = i_atoms[valid_indices]
                    valid_j = j_atoms[valid_indices]

                    # For Hooke's law: when r > r0 (stretched), forces should be attractive
                    # dr_cart points from i to j, so for attractive forces:
                    # - Force on j should point toward i: -spring_forces
                    # - Force on i should point toward j: +spring_forces
                    numpy.add.at(forces, valid_j, -spring_forces)  # Force on j atoms (toward i)
                    numpy.add.at(forces, valid_i, +spring_forces)  # Force on i atoms (toward j)

            # Update positions
            self._pos += delta * forces

    @property
    def atoms(self) -> aseAtoms:
        """Return the atoms object with current relaxed positions.

        Returns:
            aseAtoms: The original Atoms object with positions updated to the current relaxed
                configuration. Cell and other properties remain unchanged.
        """
        self._atoms.positions = self._pos.copy()
        return self._atoms

    @property
    def positions(self) -> numpy.ndarray:
        """Return the current atomic positions.

        Returns:
            numpy.ndarray: Current atomic positions in Cartesian coordinates (Angstrom),
                shape ``(N, 3)``.
        """
        return self._pos.copy()

    def clear_springs(self) -> None:
        """Clear all installed springs and reset to empty arrays."""
        self._origin_springs = {
            "indices": numpy.array([], dtype=int),
            "k_values": numpy.array([], dtype=float),
            "target_positions": numpy.empty((0, 3), dtype=float),
        }
        self._layer_springs = {
            "indices": numpy.array([], dtype=int),
            "k_values": numpy.array([], dtype=float),
            "z_targets": numpy.array([], dtype=float),
        }
        self._neighbor_springs = {
            "pairs": numpy.empty((0, 2), dtype=int),
            "k_values": numpy.array([], dtype=float),
            "r0_values": numpy.array([], dtype=float),
        }

    def get_spring_info(self) -> dict:
        """Return counts of installed springs, by type.

        Returns:
            dict: Dictionary with keys:

                - **origin_springs** (int): number of origin springs installed.
                - **layer_springs** (int): number of layer springs installed.
                - **neighbor_springs** (int): number of neighbor spring pairs installed.
        """
        info = {
            "origin_springs": len(self._origin_springs["indices"]),
            "layer_springs": len(self._layer_springs["indices"]),
            "neighbor_springs": len(self._neighbor_springs["pairs"]),
        }
        return info
