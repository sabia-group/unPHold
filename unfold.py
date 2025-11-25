import os
import pickle
import time
from typing import Literal, Optional, Union

import ase
import numpy
from ase.atoms import Atoms as aseAtoms
from ase.build.supercells import make_supercell
from phonopy import Phonopy
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.band_structure import BandStructure
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.units import VaspToCm, VaspToEv, VaspToTHz
from tqdm import tqdm


def match_two_atoms(
    a: aseAtoms,
    b: aseAtoms,
    spatial_tolerance: float = 1e-2,
):
    """without checking species!"""

    ret_dict = {
        "atoms_indices_a2b": None,
        "atoms_indices_b2a": None,
        "fail_reason": None,
    }
    st = spatial_tolerance
    # check number of atoms
    if len(a) != len(b):
        ret_dict["fail_reason"] = f"len(a)={len(a)} != len(b)={len(b)}"
        return ret_dict
    # check number of atoms by species
    if sorted(a.get_chemical_symbols()) != sorted(b.get_chemical_symbols()):
        ret_dict["fail_reason"] = f"chemical_symbols mismatch"
        return ret_dict
    # check cell
    if not numpy.allclose(a.cell, b.cell, atol=st):
        ret_dict["fail_reason"] = f"cell mismatch"
        return ret_dict
    # check positions, maybe have to permute the atoms
    a = a.copy()
    a.wrap()
    b = b.copy()
    b.wrap()
    atoms_dist = numpy.linalg.norm(a.positions[:, None, :] - b.positions[None, :, :], axis=2)  # shape (natoms, natoms)
    if numpy.sum(atoms_dist < st) != len(a):
        ret_dict["fail_reason"] = f"atoms positions mismatch, too few/many atoms' pairs with distance < {st}"
        return ret_dict
    # find the permutation as an array of indices
    ret_dict["atoms_indices_a2b"] = numpy.argmin(atoms_dist, axis=0)  # b = a[atoms_indices_a2b]
    ret_dict["atoms_indices_b2a"] = numpy.argmin(atoms_dist, axis=1)  # a = b[atoms_indices_b2a]

    return ret_dict


def atoms_ase2ph(atoms: aseAtoms):
    if not numpy.all(atoms.get_pbc()):
        print("WARNING: for PhonopyAtoms the pbc must be T T T. Set to T T T.")
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell().array,
        positions=atoms.get_positions(),
    )


def atoms_ph2ase(atoms: PhonopyAtoms):
    return aseAtoms(
        symbols=atoms.symbols,
        cell=atoms.cell,
        positions=atoms.positions,
        pbc=True,
    )


def init_phonopy_paths(
    work_dpath: str,
    makedirs: bool = True,
):
    """
    Usage:
    ```python
    supercell_phonopy = 3
    args = AttrDict({
        "work_dpath": f"{PROJECT_ROOT}/fd_runs/sc_{supercell_phonopy}",
        "unitcell_fpath": f"{PROJECT_ROOT}/unitcell_3x3.xyz",
        "supercell": supercell_phonopy,
        "displacement": 0.01,
        "model_fpath": "/u/zklou/projects/tbg_epc/mace_models/model_compiled_1.model",
    })
    args._mydict.update(init_phonopy_paths(args.work_dpath))
    ```
    """
    if not os.path.exists(work_dpath):
        os.makedirs(work_dpath, exist_ok=True)
    ph_paths = {
        "config_yaml": os.path.join(work_dpath, "phonopy_params.yaml"),
        "geoms_dpath": os.path.join(work_dpath, "geoms_disp"),
        "aims_dpath": os.path.join(work_dpath, "aims_runs"),
        "mlip_dpath": os.path.join(work_dpath, "mlip_runs"),
        "fc_fpath": os.path.join(work_dpath, "force_constants.h5"),
        "logs_dpath": os.path.join(work_dpath, "logs"),
        "band_fpath": os.path.join(work_dpath, "band.h5"),
    }
    if makedirs:
        [os.makedirs(ph_paths[k], exist_ok=True) for k in ("geoms_dpath", "aims_dpath", "mlip_dpath", "logs_dpath")]
    return ph_paths


def compute_APR_from_phonopy(
    ph: Phonopy,
) -> list[numpy.ndarray]:
    """
    Compute acoustic participation ratio (APR) for each k-path segment from Phonopy
    Args:
        ph (Phonopy): Phonopy object, must have band structure computed.
    Return:
        APR (list[numpy.ndarray]): shape [nseg, (nqpoints, nbands)], nseg for kpath segments.
    Reference:
        N. Strasser, S. Wieser, and E. Zojer, Predicting Spin-Dependent Phonon Band
        Structures of HKUST-1 Using Density Functional Theory and Machine-Learned
        Interatomic Potentials, International Journal of Molecular Sciences 25, 5 (2024).
    """
    assert ph._band_structure is not None, "Band structure not computed."
    apr_list = []
    for kseg_idx in range(len(ph._band_structure.get_qpoints())):  # qpoints as frac
        apr_list.append(
            compute_APR(
                atoms=atoms_ph2ase(ph.unitcell),
                ph_eigvecs=ph._band_structure.get_eigenvectors()[kseg_idx],
            )
        )  # Each apr is of shape (nqpoints, nbands)
    return apr_list


def compute_APR(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,  # shape (nqpoints, natoms*3, nbands)
) -> numpy.ndarray:
    """
    Compute acoustic participation ratio (APR) according to the Eq.(2) in the reference.
    Args:
        atoms (aseAtoms): ASE Atoms object containing atomic information.
        ph_eigvecs (numpy.ndarray): Phonon eigenvectors with shape (nqpoints, natoms*3, nbands).
    Returns:
        numpy.ndarray: APR values with shape (nqpoints, nbands).
    """
    r"""
    Equation:
    $$
    APR_{q,n} = \frac{2}{N(N+1)} \frac{
        \left|
            \sum_{\alpha=1}^{N} \sum_{\beta=1}^{N}
            \frac{(e_{q,n}^\alpha)^\dagger e_{q,n}^\beta}{\sqrt{m_\alpha m_\beta}}
        \right|^2
    }{
        \sum_{\alpha=1}^{N} \sum_{\beta=1}^{N}
        \left|
            \frac{(e_{q,n}^\alpha)^\dagger e_{q,n}^\beta}{\sqrt{m_\alpha m_\beta}}
        \right|^2
    }
    $$
    q -> qpoints, n -> bands, N -> natoms, (alpha, beta) -> natoms*xyz
    $e_{q,n}^\alpha$ -> phonon eigenvector for atom alpha at qpoint q and band n, has (x,y,z) components
    """
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    assert natoms3 % 3 == 0, "Number of phonon displacement basis is not a multiple of 3."
    natoms = natoms3 // 3

    # Normalize by mass
    masses_sqrt = numpy.sqrt(atoms.get_masses())  # shape (natoms,), unit u/Dalton
    eigvec_div_mass_sqrt = (
        ph_eigvecs.reshape(nqpoints, natoms, 3, nbands) / masses_sqrt[None, :, None, None]
    )  # shape (nqpoints, natoms, 3, nbands)

    # Compute the inner product (shape: nqpoints, natoms*3, natoms*3, nbands)
    inner_prod = numpy.einsum(
        "qaxn,qbxn->qabn",  # a, b: atom index
        eigvec_div_mass_sqrt.conj(),
        eigvec_div_mass_sqrt,
    )

    # Consider only upper triangular part where beta >= alpha
    triu_indices = numpy.triu_indices(natoms)  # shape (2, ntriu), ntriu = number of elems in triu
    inner_prod_triu = inner_prod[:, triu_indices[0], triu_indices[1], :]  # shape (nqpoints, ntriu, nbands)

    # Compute numerator: squared sum over upper triangular elements
    numerator = numpy.abs(inner_prod_triu.sum(axis=1)) ** 2  # shape (nqpoints, nbands)

    # Compute denominator: sum over absolute values of upper triangular elements
    denominator = numpy.sum(numpy.abs(inner_prod_triu) ** 2, axis=1)  # shape (nqpoints, nbands)

    # Compute APR
    N = natoms
    apr = (2 / (N * (N + 1))) * (numerator / denominator)  # shape (nqpoints, nbands)
    del inner_prod
    return apr


def compute_L_from_phonopy(
    ph: Phonopy,
):
    assert ph._band_structure is not None, "Band structure not computed."
    L_list = []
    cell_reciprocal = atoms_ph2ase(ph.unitcell).cell.reciprocal()
    for kseg_idx in range(len(ph._band_structure.get_qpoints())):  # qpoints as frac
        L_list.append(
            compute_L(
                atoms=atoms_ph2ase(ph.unitcell),
                ph_eigvecs=ph._band_structure.get_eigenvectors()[kseg_idx],
                q=2 * numpy.pi * ph._band_structure.qpoints[kseg_idx] @ cell_reciprocal,
            )
        )  # Each apr is of shape (nqpoints, nbands)
    return L_list


def compute_L(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,  # shape (nqpoints, natoms*3, nbands)
    q: numpy.ndarray,  # shape (nqpoints, 3)
) -> numpy.ndarray:
    r"""longitudinality

    Equation:
    $$
    L_{q,n} = \left|
        \frac{1}{N} \sum_{\alpha=1}^{N}
        \frac{q e_{q,n}^{\alpha}}{|q| \sqrt{e_{q,n}^{\alpha *} e_{q,n}^{\alpha}}}
    \right|
    $$
    """

    # Normalize the ph_eigvec to 1
    ph_eigvec_normed = ph_eigvecs / numpy.linalg.norm(ph_eigvecs, axis=1)[:, None, :]
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    natoms = len(atoms)
    assert natoms3 == natoms * 3, "Number of phonon displacement basis is not a multiple of 3."
    ph_eigvec_normed = ph_eigvec_normed.reshape(nqpoints, natoms, 3, nbands)

    # Get the direction of q vector
    q_normed = q / (numpy.linalg.norm(q, axis=1)[:, None] + 1e-5)  # shape (nqpoints, 3), unit vector

    # Compute L, longitudinality
    lgt = numpy.einsum("qaxn,qx->qan", ph_eigvec_normed, q_normed)  # shape (nqpoints, natoms, nbands)
    lgt = lgt.mean(axis=1)  # shape (nqpoints, nbands)
    lgt = natoms**0.5 * numpy.abs(lgt)  # shape (nqpoints, nbands), need natoms**0.5 here to ensure max=1
    return lgt


def compute_V(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,  # shape (nqpoints, natoms*3, nbands)
) -> numpy.ndarray:
    r"""verticality

    Equation:
    $$
    V_{q,n} =
    \frac{1}{N} \sum_{\alpha=1}^{N}
    \left|
    \frac{\hat{z} e_{q,n}^{\alpha}}{\sqrt{e_{q,n}^{\alpha *} e_{q,n}^{\alpha}}}
    \right|
    $$

    Discussion:
    This metric will over-emphrasize the ones of small displacements.
    so function `compute_V_p2` is recommended over this one.
    """

    # Normalize the ph_eigvec to 1
    ph_eigvec_normed = ph_eigvecs / numpy.linalg.norm(ph_eigvecs, axis=1)[:, None, :]
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    natoms = len(atoms)
    assert natoms3 == natoms * 3, "Number of phonon displacement basis is not a multiple of 3."
    ph_eigvec_normed = ph_eigvec_normed.reshape(nqpoints, natoms, 3, nbands)

    # Compute V, verticality
    vtcl = numpy.abs(ph_eigvec_normed[:, :, 2, :])  # shape (nqpoints, natoms, nbands)
    vtcl = natoms**0.5 * vtcl.mean(axis=1)  # shape (nqpoints, nbands)
    return vtcl


def compute_V_p2(
    atoms: aseAtoms,
    ph_eigvecs: numpy.ndarray,  # shape (nqpoints, natoms*3, nbands)
) -> numpy.ndarray:
    r"""verticality (norm by p2, or collective)

    Equation:
    $$
    V_{q,n}^{p=2} =
    \frac{
        \sum_{\alpha=1}^{N} |\hat{z} e_{q,n}^{\alpha}|^2
    }{
        \sum_{\alpha=1}^{N} |e_{q,n}^{\alpha}|^2
    }
    $$

    Physical Interpretation:
    - $V_{q,n} = 0$: purely in-plane vibration
    - $V_{q,n} = 1$: purely out-of-plane vibration
    - $V_{q,n} = 0.5$: threshold value (e.g., single atom vibrating 45° from z-axis)
    """

    # Normalize the ph_eigvec to 1
    ph_eigvec_normed = ph_eigvecs / numpy.linalg.norm(ph_eigvecs, axis=1)[:, None, :]
    nqpoints, natoms3, nbands = ph_eigvecs.shape
    natoms = len(atoms)
    assert natoms3 == natoms * 3, "Number of phonon displacement basis is not a multiple of 3."
    ph_eigvec_normed = ph_eigvec_normed.reshape(nqpoints, natoms, 3, nbands)

    vtcl2 = numpy.abs(ph_eigvec_normed[:, :, 2, :])  # shape (nqpoints, natoms, nbands)
    vtcl2 = numpy.sum(vtcl2**2, axis=1)  # shape (nqpoints, nbands)
    return vtcl2


def rotmat_xOy(angle: float):
    """
    Args:
        angle: in radian
    """
    return numpy.array(
        [
            [
                numpy.cos(angle),
                -numpy.sin(angle),
                0,
            ],
            [
                numpy.sin(angle),
                numpy.cos(angle),
                0,
            ],
            [
                0,
                0,
                1,
            ],
        ]
    )


class Unfold:
    """
    Usage:
    ```python
    unitcell, tmat = ...  # prepare
    supercell = make_supercell(unitcell, tmat)
    unfold = Unfold(unitcell, supercell, tmat)
    special_points={  # for hexagon cell with gamma = 60.0 deg
        "G": [0.0, 0.0, 0.0],
        "M": [1 / 2, 1 / 2, 0.0],
        "K": [1 / 3, 2 / 3, 0.0],
        "K1": [-1 / 3, 1 / 3, 0.0],
        "K2": [1 / 3, 2 / 3, 0.0],
        "K3": [2 / 3, 1 / 3, 0.0],
        "A": [0.0, 0.0, 1 / 2],
    }
    bz_labels = ["G", "M", "K", "G"]
    kpath = [[special_points[label] for label in bz_labels]]  # must be 2-level list
    kpts_list, connections = get_band_qpoints_and_path_connections(kpath, npoints=41)
    kpts_concat, bz_labels_indices = concatenate_bands(kpts_list, connections)

    my_unfold = Unfold(
        unitcell = unitcell,
        supercell = atoms_ph2ase(ph_sc.unitcell),
        transformation_matrix = tmat,
        verbose = True,
    )
    unfold.set_kpts_in_unitcell(kpts, format="fractional")
    my_unfold.calculate_sc_phonon(ph_sc.dynamical_matrix, "meV")
    my_unfold.calulate_weights()
    my_unfold.calculate_band_expansion()
    en_min, en_max, en_delta = 0.0, 200.0, 0.1
    en_sigma = 5 * en_delta
    my_unfold.calculate_band_expansion(
        grid=numpy.arange(en_min, en_max+1e-6, en_delta),
        sigma=en_sigma,
    );
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    energies_on_grid = my_unfold.energies_on_grid.T
    # vmax = (en_sigma * numpy.sqrt(2 * numpy.pi))
    vmax = energies_on_grid[energies_on_grid>1e-1].mean()
    ax.imshow(
        energies_on_grid,
        cmap="Greys",
        aspect="auto",
        origin="lower",
        vmax=vmax,
        extent=[-0.5, len(my_unfold.kpts_uc_frac) - 0.5, en_min-en_delta/2, en_max+en_delta/2],
    )
    ax.set_xticks(bz_labels_indices, bz_labels)
    ax.vlines(
        bz_labels_indices,
        en_min-en_delta/2, en_max+en_delta/2,
        color="grey",
        linestyle="dashed",
        lw=1.0,
    )
    ax.set_xlabel("k-points")
    ax.set_ylabel("Energy (meV)")
    fig.tight_layout()
    plt.show()
    ```
    """

    def __init__(
        self,
        unitcell: ase.Atoms,  # very primitive cell
        supercell: ase.Atoms,  # one layer in the twisted bilayer, phonopy.unitcell
        transformation_matrix: numpy.ndarray,  # from unitcell to supercell
        transformation_matrix_ph: numpy.ndarray = None,  # from phonopy.unitcell to phonopy.supercell
        angle: Optional[float] = None,
        spatial_tolerance: float = 5e-2,
        unfold_atoms_indices: Optional[numpy.ndarray] = None,
        perm_idx_i2g: Optional[numpy.ndarray] = None,
        perm_idx_g2i: Optional[numpy.ndarray] = None,
        verbose: bool = False,
    ):
        """Unfold phonon band structure from phonopy supercell to a unit cell

        Args:
            unitcell (ase.Atoms): the unitcell
            supercell (ase.Atoms): the supercell from phonon calculation (you can get it by `phonopy.unitcell)
            transformation_matrix (numpy.ndarray): Transformation matrix to create supercell from unitcell
            angle (Optional[float]): Rotate the supercell generated from unitcell to align with the phonopy supercell,
                in degrees
            spatial_tolerance (float): Tolerance for spatial matching of atoms
            unfold_atoms_indices (Optional[numpy.ndarray]): Indices of atoms in the phonopy supercell to be unfolded,
                e.g. 2D bilayer case
            perm_idx_i2g (Optional[numpy.ndarray]): Permutation indices from phonopy supercell (or phonopy supercell
                sliced by unfold_atoms_indices) to the generated supercell by unit cell
            perm_idx_g2i (Optional[numpy.ndarray]): The inverse permutation of `perm_idx_i2g`
            verbose (bool): Whether to show progress bars and additional output

        Variables meanings:
        - `uc`: unitcell, should be stretched and rotated so that can be transformed to the supercell with only
            `make_supercell` and `tmat`
        - `sc`: supercell in phonopy
        - `sc_uf`: part of the supercell that will be unfolded, will only be used in case 2 and 3

        Cases in unfolding:
        - Case 0: no permutation required, project all atoms
            - no extra args required
            - use `sc`
        - Case 1: permutation required, project all atoms, e.g. atoms sequence changed
            - requires perm_idx_xxx
            - use `sc`
        - Case 2: no permutation required, project part of atoms, e.g. 2D bilayer case
            - requires unfold_atoms_indices
            - use `sc_uf`
        - Case 3: permutation required, project part of atoms, e.g. 2D bilayer case with atoms sequence changed
            - requires perm_idx_xxx and unfold_atoms_indices
            - use `sc_uf`

        Performance:
        - The most time consuming part is calculating band structure in the supercell, which is done by phonopy so cannot be parallelized.
        """

        self.uc = unitcell.copy()
        self.sc = supercell.copy()
        self.tmat = transformation_matrix
        self.tmat_ph = transformation_matrix_ph  # not necessarily required in current procedure
        self.angle = angle
        self.sc_by_mat = None
        self.unfold_atoms_indices = unfold_atoms_indices
        self.perm_idx_i2g = perm_idx_i2g
        self.perm_idx_g2i = perm_idx_g2i
        self.spa_tol = spatial_tolerance
        self.verbose = verbose
        self.prepare()

    def __repr__(self):
        return (
            f"Unfold(uc={self.uc.symbols}, sc={self.sc.symbols},tmat={self.tmat}, tmat_ph={self.tmat_ph},",
            f"angle={self.angle:.3f}, spa_tol={self.spa_tol:.2e}, "
            f"unfold_atoms_indices={self.unfold_atoms_indices}, "
            f"perm_idx_i2g={self.perm_idx_i2g}, perm_idx_g2i={self.perm_idx_g2i})",
        )

    def prepare(
        self,
    ):
        """(reciprocal) lattice vectors (row vectors)
        Relations:
        sc_la_vec = tmat @ uc_la_vec
        uc_bz_vec = tmat.T @ sc_bz_vec
        """

        """Check the supercell validity by transformation matrix"""
        self.sc_by_mat = make_supercell(self.uc, self.tmat, wrap=False)  # without wrapping
        if self.angle is not None:
            assert isinstance(self.angle, float)
            self.sc_by_mat.rotate(self.angle, "z", rotate_cell=True)
        if (self.perm_idx_g2i is not None) and (self.perm_idx_i2g is not None):
            # check the permutation indices a looooooot
            assert isinstance(self.perm_idx_g2i, numpy.ndarray) and isinstance(self.perm_idx_i2g, numpy.ndarray)
            assert self.perm_idx_g2i.shape == self.perm_idx_i2g.shape
            assert numpy.all(self.perm_idx_g2i >= 0) and numpy.all(self.perm_idx_i2g >= 0)
            if self.unfold_atoms_indices is None:
                assert len(self.perm_idx_g2i) == len(self.sc_by_mat) and len(self.perm_idx_i2g) == len(self.uc)
            else:
                assert len(self.perm_idx_g2i) == len(self.unfold_atoms_indices) and len(self.perm_idx_i2g) == len(
                    self.unfold_atoms_indices
                )
                assert len(self.unfold_atoms_indices) == len(
                    self.sc_by_mat
                )  # indices the part to be unfolded in phonopy supercell
                assert len(numpy.unique(self.unfold_atoms_indices)) == len(
                    self.unfold_atoms_indices
                )  # should be unique
        else:
            print("WARNING: it is strongly recommended to provide permutation indices perm_idx_i2g and perm_idx_g2i")
            _match_scs = match_two_atoms(self.sc, self.sc_by_mat, spatial_tolerance=self.spa_tol)
            if _match_scs["fail_reason"] is not None:
                raise Exception(_match_scs["fail_reason"])
            self.perm_idx_i2g = _match_scs["atoms_indices_a2b"]  # input to generated
            self.perm_idx_g2i = _match_scs["atoms_indices_b2a"]  # generated to input
        self.nucs_in_sc = len(self.sc_by_mat) // len(self.uc)  # nubmer of unit cells in the supercell

        """prepare the lattice / BZ vectors"""
        self.uc_la = numpy.array(self.uc.cell)  # lattice vector, shape (3=la_vec, 3=xyz)
        self.uc_bz = numpy.array(self.uc.cell.reciprocal())  # BZ vector, shape (3=bz_vec, 3=xyz)
        self.sc_la = numpy.array(self.sc.cell)  # lattice vector, shape (3=la_vec, 3=xyz)
        self.sc_bz = numpy.array(self.sc.cell.reciprocal())  # BZ vector, shape (3=bz_vec, 3=xyz)
        assert numpy.allclose(self.sc_la[:2, :2], (self.tmat @ self.uc_la)[:2, :2], atol=3e-2)  # only compare xy
        assert numpy.allclose(self.uc_bz[:2, :2], (self.tmat.T @ self.sc_bz)[:2, :2], atol=3e-2)  # only compare xy

    def set_kpts_in_unitcell(self, kpts: numpy.ndarray, format: Literal["fractional", "cartesian"] = "fractional"):
        """fractional kpoints in scBZ of the ucBZ (self.kpts_uc_frac)
        k_pos = k-points in k-space
              = kpts_uc_frac @ uc_bz_vec
              = kpts_uc_frac @ tmat.T @ sc_bz_vec
              =          kpts_sc_frac @ sc_bz_vec
        For cartesian: without 2pi !!!
        """
        if format == "fractional":
            self.kpts_cart = numpy.matmul(kpts, self.uc_bz)  # without 2pi
            self.kpts_uc_frac = kpts
            self.kpts_sc_frac = numpy.matmul(self.kpts_uc_frac, self.tmat.T)
        elif format == "cartesian":
            self.kpts_cart = kpts  # without 2pi
            self.kpts_uc_frac = numpy.matmul(kpts, numpy.linalg.inv(self.uc_bz))
            self.kpts_sc_frac = numpy.matmul(kpts, numpy.linalg.inv(self.sc_bz))
        else:
            raise ValueError(f"format={format} not supported")
        assert numpy.allclose(self.kpts_uc_frac @ self.uc_bz, self.kpts_cart)
        assert numpy.allclose(self.kpts_sc_frac @ self.sc_bz, self.kpts_cart)

    def calculate_sc_phonon(
        self,
        dyn_sc: Union[DynamicalMatrix, DynamicalMatrixNAC],
        factor: Union[float, str] = VaspToEv,
        save_fpath: Optional[str] = None,
    ):
        """Calculate phonon band structure in the supercell.

        This step might take a long time because it diagonalizes the dynamical matrix,
        please be patient. Progress bar cannot be added here because it is implemented
        inside phonopy.

        Args:
            dyn_sc (Union[DynamicalMatrix, DynamicalMatrixNAC]): Dynamical matrix for the supercell.
            factor (Union[float, str]): Energy conversion factor. Can be a float or string.
                Supported strings: "ev", "mev", "thz", "cm".
                Defaults to `VaspToEv`, or equivalently `"ev"`.
            save_fpath (Optional[str]): Path to save the band structure data, should be a .npz file.
                If None, the data will not be saved. Defaults to None.
        """

        if isinstance(factor, float):
            pass
        elif isinstance(factor, str):
            factor = factor.lower()
            assert factor in (
                "ev",
                "mev",
                "thz",
                "cm",
            ), f"factor={factor} not supported"
            factor = {"ev": VaspToEv, "mev": VaspToEv * 1e3, "thz": VaspToTHz, "cm": VaspToCm}[factor]
        else:
            raise ValueError(f"factor={factor} not supported")

        # # calculate band structure in the unitcell
        # bs_uc = BandStructure(
        #     paths=[self.kpts_uc_frac],
        #     dynamical_matrix=dyn_uc,
        #     with_eigenvectors=True,
        #     factor=factor,
        # )
        # self.bs_uc_energies = bs_uc.frequencies[0]  # shape (nkpts, nbands)
        # self.bs_sc_eigenvecs = bs_uc.eigenvectors[0]  # shape (nkpts, natoms*xyz, nbands)
        # nkpts, natoms3_uc, nbands_uc = self.bs_sc_eigenvecs.shape
        # self.bs_sc_eigenvecs = self.bs_sc_eigenvecs.reshape(nkpts, natoms3_uc // 3, 3, nbands_uc)

        # calculate band structure in the supercell
        time_start = time.time()
        bs_sc = BandStructure(
            paths=[self.kpts_sc_frac],
            dynamical_matrix=dyn_sc,
            with_eigenvectors=True,
            factor=factor,
        )
        time_end = time.time()
        self.bs_sc_energies = bs_sc.frequencies[0]
        self.bs_sc_eigenvecs = bs_sc.eigenvectors[0]  # shape (nkpts, natoms*xyz, nbands)
        print(
            f"Band structure: {time_end - time_start:.2f} seconds for {len(self.kpts_sc_frac)} k-points,",
            f"so {(time_end - time_start) / len(self.kpts_sc_frac):.2f} seconds per k-point.",
        )
        if save_fpath is None:
            pass
        else:
            if not save_fpath.endswith(".npz"):
                print("WARNING: save_fpath should end with .npz, adding it automatically.")
                save_fpath += ".npz"
            os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
            numpy.savez(
                file=save_fpath,
                bs_sc_energies=bs_sc.frequencies[0],  # array, shape (nkpts, nbands)
                bs_sc_eigenvecs=bs_sc.eigenvectors[0],  # array, shape (nkpts, natoms*xyz, nbands)
                kpts_sc_frac=self.kpts_sc_frac,  # array, shape (nkpts, 3)
                factor=factor,  # float
            )

    def load_sc_phonon(
        self,
        save_fpath: str,
    ):
        """Load phonon band structure in the supercell from file.

        Args:
            save_fpath (str): Path to the saved phonon band structure file.
        """
        assert os.path.exists(save_fpath), f"File {save_fpath} does not exist."
        data = numpy.load(save_fpath, allow_pickle=True)
        assert data["bs_sc_energies"].shape[0] == self.kpts_sc_frac.shape[0]
        assert data["bs_sc_energies"].shape[1] == len(self.sc) * 3
        assert data["bs_sc_eigenvecs"].shape[1] == len(self.sc) * 3
        self.bs_sc_energies = data["bs_sc_energies"]  # array, shape (nkpts, nbands)
        self.bs_sc_eigenvecs = data["bs_sc_eigenvecs"]  # array, shape (nkpts, natoms*xyz, nbands)
        assert numpy.allclose(self.kpts_sc_frac, data["kpts_sc_frac"])  # array, shape (nkpts, 3)
        return data["factor"]  # ref to VaspToXXXX

    def _calculate_weights_one_kpt(self, kpt_idx: int):
        # kpt_cart = self.kpts_cart[kpt_idx]
        # kpt_uc_frac = self.kpts_uc_frac[kpt_idx]
        # kpt_sc_frac = self.kpts_sc_frac[kpt_idx]
        # uc2sc_bloch_phases = numpy.exp(-2.0j*numpy.pi * numpy.matmul(self.sc.positions, kpt_cart))  # incorrect
        # uc2sc_bloch_phases = numpy.ones(len(self.sc))  # phonopy applied the Bloch phase atom by atom, so no phase here
        uc_natoms = len(self.uc)
        sc_natoms = len(self.sc)
        sc_uf_natoms = len(self.unfold_atoms_indices) if self.unfold_atoms_indices is not None else sc_natoms
        """ prepare unitcell modes, be aware of the permutation """
        uc_modes = numpy.diag(numpy.ones(3 * len(self.uc))).reshape(
            uc_natoms, 3, 3 * uc_natoms
        )  # uc_nmodes = uc_natoms*xyz
        uc2sc_modes = numpy.tile(uc_modes, (self.nucs_in_sc, 1, 1))  # shape (sc_natoms, xyz, uc_nmodes)
        # uc2sc_modes = numpy.einsum("n,nxb->nxb", uc2sc_bloch_phases, uc2sc_modes)  # shape (sc_natoms, xyz, uc_nmodes)
        if self.perm_idx_g2i is not None:  # permutate the uc2sc, so unfold_atoms_indices doesn't matter here
            uc2sc_modes = uc2sc_modes[self.perm_idx_g2i]  # permute the modes to match the supercell
            uc2sc_modes = uc2sc_modes.reshape(
                3 * len(self.perm_idx_g2i), 3 * uc_natoms
            )  # shape (sc_natoms*xyz, uc_nmodes)
        else:
            uc2sc_modes = uc2sc_modes.reshape(3 * sc_natoms, 3 * uc_natoms)  # shape (sc_natoms*xyz, uc_nmodes)
        """ prepare supercell modes """
        sc_modes = self.bs_sc_eigenvecs[kpt_idx]  # shape (sc_natoms*xyz, sc_nbands)
        if self.unfold_atoms_indices is not None:
            """
            unfolding only involves part of the phonopy supercell
            uc_modes: unitcell modes, but tiled to supercell format
                - sc_uf_natoms = uc2sc_natoms: the number of atoms in the supercell created by transformation matrix, i.e. number of atoms in the layer to be projected
                - shape (sc_uf_natoms*xyz, uc_nmodes=sc_uf_natoms*xyz)
                - eigen_uc2sc (atoms_)
            sc_modes: supercell modes, containing all atoms
                - shape (sc_natoms*xyz, sc_nbands=sc_natoms*xyz)
                - eigen_sc: (atom*3, atom*3)
            sc_uf_modes: supercell modes for those atoms to be projected
                - shape (sc_uf_natoms*)
            """
            sc_uf_modes = sc_modes.reshape(sc_natoms, 3, 3 * sc_natoms)
            sc_uf_modes = sc_uf_modes[self.unfold_atoms_indices]  # take only the motion of atoms to be unfolded
            sc_uf_modes = sc_uf_modes.reshape(3 * sc_uf_natoms, 3 * sc_natoms)  # shape (sc_uf_natoms*xyz, sc_nmodes)
            weights = numpy.einsum("in,ib->nb", uc2sc_modes.conj(), sc_uf_modes)  # shape (uc_nmodes, sc_nbands)
            weights = (numpy.abs(weights) ** 2).sum(axis=0)  # shape (sc_nbands,)
        else:
            """
            what's doing here: unfolding involves the whole phonopy supercell
            band: phonon displacement
            mode: just xyz
            | uc_band > = C * | uc_mode >
            < sc_band | uc_mode > < uc_mode | sc_band >: overlap between one uc mode (not uc mode) and one sc band
            < sc_band | uc_mode > < uc_mode | sc_band >
                = < sc_band | C^{-1} | uc_band > < uc_band | C | sc_band >
            \sum_{uc} < sc_band | uc_mode > < uc_mode | sc_band >: `weights` here, contribution of this sc band at this kpoint.
            \sum_{uc} < sc_band | uc_mode > < uc_mode | sc_band >
                = \sum_{uc} < sc_band | C^{-1} | uc_band > < uc_band | C | sc_band >
            """
            weights = numpy.einsum("in,ib->nb", uc2sc_modes.conj(), sc_modes)  # shape (uc_nmodes, sc_nbands)
            weights = (numpy.abs(weights) ** 2).sum(axis=0)  # shape (sc_nbands,)
            _weights_check = numpy.einsum("ji,ij->j", weights.conj().T, weights)  # shape (sc_nbands,)
            assert numpy.allclose(weights, _weights_check)  # double check
            # weights = numpy.diag(numpy.matmul(weights.conj().T, weights))  # shape (sc_nbands,)
        return weights / self.nucs_in_sc

    def calulate_weights(self):
        """Calculate the weights of each band in the supercell phonon band structure."""
        weights = []
        if self.verbose:
            ktp_idx_iterator = tqdm(
                range(len(self.kpts_uc_frac)),
                desc="Projecting",
                ncols=64,
            )
        else:
            ktp_idx_iterator = range(len(self.kpts_uc_frac))
        for kpt_idx in ktp_idx_iterator:
            weights.append(self._calculate_weights_one_kpt(kpt_idx))
        self.weights = numpy.array(weights)  # shape (nkpts, nbands)

    def _calulate_band_expansion_one_kpt(self, kpt_idx: int, grid: numpy.ndarray, sigma: float):
        weights = self.weights[kpt_idx]  # shape (sc_nbands, )
        sc_energies = self.bs_sc_energies[kpt_idx]  # shape (sc_nbands, )
        sc_energies_expanded = band_expansion(energies=sc_energies, grid=grid, sigma=sigma)  # shape (sc_nbands, ngrid)
        proj_energies_expanded = numpy.einsum("j,jk->k", weights, sc_energies_expanded)  # shape (ngrid, )
        return proj_energies_expanded

    def calculate_band_expansion(self, grid: Optional[numpy.ndarray] = None, sigma: Optional[float] = None):
        """Calculate the band structure expansion on a grid in band space."""
        if (grid is None) and (sigma is None):
            # grid = numpy.linspace(0, 0.2, 201)
            _div = (self.bs_sc_energies.max() - self.bs_sc_energies.min()) / 2000
            _overshoot = self.bs_sc_energies.max() * 0.05
            grid = numpy.arange(self.bs_sc_energies.min() - _overshoot, self.bs_sc_energies.max() + _overshoot, _div)
            sigma = 5 * _div
        else:
            assert isinstance(grid, numpy.ndarray) and grid.ndim == 1
            assert isinstance(sigma, float) and sigma > 0
        if self.verbose:
            ktp_idx_iterator = tqdm(
                range(len(self.kpts_uc_frac)),
                desc="Building Grids",
                ncols=64,
            )
        else:
            ktp_idx_iterator = range(len(self.kpts_uc_frac))
        energies_on_grid = []
        for kpt_idx in ktp_idx_iterator:
            energies_on_grid.append(self._calulate_band_expansion_one_kpt(kpt_idx, grid, sigma))
        self.energies_on_grid = numpy.stack(energies_on_grid, axis=0)  # shape (nkpt, ngrid)
        return grid, sigma

    def save(self, fpath: str):
        """Save the Unfold object to a file using pickle"""
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fpath: str):
        """Load an Unfold object from a file using pickle"""
        with open(fpath, "rb") as f:
            loaded_obj = pickle.load(f)
        if not isinstance(loaded_obj, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return loaded_obj


def gaussian_function(x: numpy.ndarray, mu: Union[int, numpy.ndarray] = 0, sigma: int = 1e-2):
    """Calculate Gaussian function values for given parameters.

    This function computes the Gaussian (normal) distribution function:
    f(x) = exp(-(x-mu)^2/(2*sigma^2))/(sigma*sqrt(2*pi))

    Args:
        x (numpy.ndarray): Input array for which to calculate Gaussian values.
        mu (Union[int, numpy.ndarray], optional): Mean(s) of the Gaussian distribution.
            If int, a single Gaussian is calculated.
            If numpy.ndarray, multiple Gaussians are calculated. Defaults to 0.
        sigma (int, optional): Standard deviation of the Gaussian distribution. Defaults to 1e-2.

    Raises:
        ValueError: If mu is neither int nor numpy.ndarray.

    Returns:
        numpy.ndarray: Gaussian function values.
            If mu is int: output shape matches x.shape
            If mu is numpy.ndarray: output shape is mu.shape times x.shape (tensor product)
    """
    if isinstance(mu, int):
        pass  # return shape: x.shape
    elif isinstance(mu, numpy.ndarray):
        assert x.ndim == 1
        mu = mu[..., numpy.newaxis]  # shape (..., 1), return shape: mu.shape \otimes x.shape
    else:
        raise ValueError("mu should be int or numpy.ndarray, but got " + str(type(mu)))
    return numpy.exp(-((x - mu) ** 2) / (2 * sigma**2)) / (sigma * numpy.sqrt(2 * numpy.pi))


def band_expansion(energies: numpy.ndarray, grid: numpy.ndarray, sigma=1e-2):
    """expand the band structure to a grid by Gaussian function, i.e. add an extra"""
    grid_delta_min = numpy.min(numpy.diff(grid))
    if grid_delta_min > sigma:
        print(f"Warning: grid delta is smaller than sigma: {grid_delta_min:.3e} < {sigma:.3e}")
    return gaussian_function(grid, energies, sigma)  # shape (nbands, ngrids)


def concatenate_bands(
    kpts: list[numpy.ndarray],
    connections: list[bool],
):
    """take the output of phonopy.phonon.band_structure.get_band_qpoints_and_path_connections
    and remove the duplicated kpoints if two neighboring band kpath has the same start/end point
    connections[i] means kpts[i] and kpts[i+1] are/ain't connected

    WARN: this function can only deal with one-contour kpath, no interruped kpath allowed
    """
    assert len(kpts) == len(connections)
    kpts_new: list[numpy.ndarray] = []
    for i in range(len(kpts)):
        if connections[i]:
            kpts_new.append(kpts[i][:-1])  # kpts[i][-1] == kpts_arr[i+1][0]
        else:
            kpts_new.append(kpts[i])
    # calculate the indices corresponds to bz_labels
    bz_labels_indices = [
        0,
    ]
    next_seg_begin_index = 0
    for i in range(len(kpts)):
        if connections[i]:
            next_seg_begin_index += kpts[i].shape[0] - 1
            bz_labels_indices.append(next_seg_begin_index)
        else:
            next_seg_begin_index += kpts[i].shape[0]
            if i != len(kpts) - 1:
                bz_labels_indices.append(next_seg_begin_index - 1)
                bz_labels_indices.append(next_seg_begin_index)
            else:
                bz_labels_indices.append(next_seg_begin_index - 1)
    return numpy.concatenate(kpts_new, axis=0), bz_labels_indices


class UnfoldTwistBilayer:
    """unfold twisted bilayer to one layer in a monolayer unitcell"""

    def __init__(
        self,
    ):
        """to be implemented"""
        pass
