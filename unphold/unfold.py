"""Core phonon unfolding classes.

Provides:

- [`Unfold`][unphold.unfold.Unfold]: general supercell → unitcell phonon band unfolding
"""

import os
import pickle
import time

import numpy
from ase.atoms import Atoms as aseAtoms
from ase.build.supercells import make_supercell
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix, DynamicalMatrixNAC
from phonopy.phonon.band_structure import BandStructure
from phonopy.physical_units import get_physical_units as _get_phonopy_units
from tqdm import tqdm

from .utils import band_expansion, match_two_atoms

_pu = _get_phonopy_units()
VASP_TO_THZ = _pu.DefaultToTHz
VASP_TO_EV = VASP_TO_THZ * _pu.THzToEv
VASP_TO_CM = VASP_TO_THZ * _pu.THzToCm


class Unfold:
    """Unfold phonon band structure from a Phonopy supercell to a primitive unitcell.

    The spectral weight at primitive-cell k-point **k** for supercell band *n* is:

    $$w_{k,n} = \\frac{1}{N_{uc}} \\sum_i |\\langle \\phi^{uc}_{k,i} | \\Psi^{sc}_{k,n} \\rangle|^2$$

    where $N_{uc}$ is the number of primitive cells in the supercell.

    The correspondence between the ideal, unit-cell-generated supercell (``sc_by_tmat``,
    built from ``unitcell`` and ``transformation_matrix``) and the real supercell
    (``supercell``, which Phonopy actually diagonalised) is given by a single array,
    ``perm_sc2gen``: for each atom in ``sc_by_tmat``, the index of the corresponding atom
    in ``supercell``, or ``-1`` if there is none.

    This one array covers what used to be three separate mechanisms:

    - **Atom reordering**: ``supercell`` and ``sc_by_tmat`` list the same atoms in a
      different order (the common case) - ``perm_sc2gen`` just encodes the permutation.
    - **Projecting a subset of atoms** (e.g. one layer of a bilayer): pass ``unitcell``/
      ``transformation_matrix`` for that one layer, and let ``perm_sc2gen`` map its ideal
      sites onto the corresponding atoms in the *full* ``supercell`` - atoms belonging to
      other layers simply never appear as values.
    - **Vacancies**: an ideal site with no real counterpart (e.g. a missing atom) is
      marked with ``-1``. Its row contributes a zero vector to the projection instead of
      raising an error. See [Handling defects](#handling-defects) below.

    If ``perm_sc2gen`` is not supplied, it is computed automatically by matching
    ``supercell`` and ``sc_by_tmat`` position-by-position (see
    [`match_two_atoms`][unphold.utils.match_two_atoms]); this only works when the two
    have identical atom counts and no vacancies.

    ## Handling defects

    For a supercell with vacancies, mark the corresponding ideal sites in
    ``perm_sc2gen`` with ``-1``. The projector is built by zero-padding: an ideal site
    with no real counterpart contributes nothing to the inner product, rather than being
    excluded from the basis. One consequence is that the captured spectral weight is
    then no longer exactly conserved - for $n_v$ point vacancies,

    $$\\sum_n w_{k,n} = 3\\,N_{atoms}^{uc} - \\frac{3 n_v}{N_{uc}}$$

    instead of $3\\,N_{atoms}^{uc}$ exactly, with the deficit vanishing as the supercell
    size $N_{uc}$ grows (dilute-defect limit). This is expected, not a bug: the missing
    atom's phonon character is genuinely absent from the diagonalised system, so it
    cannot be captured by this projector.

    Example::

        from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
        from unphold import Unfold
        from unphold.utils import concatenate_bands

        special_points = {
            "G": [0.0, 0.0, 0.0],
            "M": [0.5, 0.5, 0.0],
            "K": [1/3, 2/3, 0.0],
        }
        bz_labels = ["G", "M", "K", "G"]
        kpath = [[special_points[l] for l in bz_labels]]
        kpts_list, connections = get_band_qpoints_and_path_connections(kpath, npoints=41)
        kpts, bz_label_indices = concatenate_bands(kpts_list, connections)

        unfold = Unfold(unitcell, supercell, tmat, verbose=True)
        unfold.set_kpts_in_unitcell(kpts, format="fractional")
        unfold.calculate_sc_phonon(ph.dynamical_matrix, "meV")
        unfold.calculate_weights()
        grid, sigma = unfold.calculate_spectral_function_on_grid()
    """

    def __init__(
        self,
        unitcell: aseAtoms,
        supercell: aseAtoms,
        transformation_matrix: numpy.ndarray,
        transformation_matrix_ph: numpy.ndarray = None,
        angle: float | None = None,
        spatial_tolerance: float = 5e-2,
        perm_sc2gen: numpy.ndarray | None = None,
        verbose: bool = False,
    ):
        """
        Args:
            unitcell (aseAtoms): Primitive unitcell.
            supercell (aseAtoms): Supercell from the phonon calculation
                (retrieve via ``phonopy.unitcell`` after converting with ``atoms_ph2ase``).
            transformation_matrix (numpy.ndarray): Integer matrix mapping unitcell → supercell.
            transformation_matrix_ph (numpy.ndarray, optional): Phonopy-internal transformation
                matrix (not required by the current algorithm).
            angle (float, optional): Rotation angle in degrees to align the generated supercell
                with the Phonopy supercell (moiré systems).
            spatial_tolerance (float): Atom-matching tolerance in Angstrom.
            perm_sc2gen (numpy.ndarray, optional): Index array of shape
                ``(nucs_in_sc * len(unitcell),)``, one entry per atom of the ideal
                Phonopy-generated supercell (``sc_by_tmat``), giving the index of the
                corresponding atom in ``supercell``, or ``-1`` if there is none (vacancy,
                or an atom outside the region being projected - e.g. the other layer of
                a bilayer). If ``None``, computed automatically by matching ``supercell``
                to ``sc_by_tmat`` position-by-position (requires equal atom counts and no
                vacancies).
            verbose (bool): Show progress bars.
        """
        self.uc = unitcell.copy()
        self.sc = supercell.copy()
        self.tmat = transformation_matrix
        self.tmat_ph = transformation_matrix_ph
        self.angle = angle
        self.sc_by_tmat = None
        self.perm_sc2gen = perm_sc2gen
        self.spatial_tolerance = spatial_tolerance
        self.verbose = verbose
        self.prepare()

    def __repr__(self):
        return (
            f"Unfold(uc={self.uc.symbols}, sc={self.sc.symbols}, tmat={self.tmat}, "
            f"angle={self.angle}, spatial_tolerance={self.spatial_tolerance:.2e})"
        )

    def prepare(self):
        """Validate geometry and precompute lattice/BZ vectors.

        Called automatically on construction. Re-call if you modify ``tmat`` or ``angle``.

        Relations::

            sc_lattice = tmat @ uc_lattice
            uc_BZ      = tmat.T @ sc_BZ
        """
        self.sc_by_tmat = make_supercell(self.uc, self.tmat, wrap=False)
        if self.angle is not None:
            assert isinstance(self.angle, float)
            self.sc_by_tmat.rotate(self.angle, "z", rotate_cell=True)

        if self.perm_sc2gen is not None:
            assert isinstance(self.perm_sc2gen, numpy.ndarray)
            assert self.perm_sc2gen.shape == (len(self.sc_by_tmat),), (
                f"perm_sc2gen should have shape ({len(self.sc_by_tmat)},) "
                f"(one entry per atom of sc_by_tmat), got {self.perm_sc2gen.shape}"
            )
            _valid = self.perm_sc2gen >= 0
            assert numpy.all(self.perm_sc2gen[_valid] < len(self.sc)), "perm_sc2gen has out-of-range entries"
            assert len(numpy.unique(self.perm_sc2gen[_valid])) == _valid.sum(), (
                "perm_sc2gen must be injective on its non-vacant (>= 0) entries"
            )
        else:
            print("WARNING: it is strongly recommended to provide perm_sc2gen")
            _match = match_two_atoms(self.sc, self.sc_by_tmat, spatial_tolerance=self.spatial_tolerance)
            if _match["fail_reason"] is not None:
                raise ValueError(_match["fail_reason"])
            self.perm_sc2gen = _match["atoms_indices_a2b"]

        self.nucs_in_sc = len(self.sc_by_tmat) // len(self.uc)

        self.uc_la = numpy.array(self.uc.cell)
        self.uc_bz = numpy.array(self.uc.cell.reciprocal())
        self.sc_la = numpy.array(self.sc.cell)
        self.sc_bz = numpy.array(self.sc.cell.reciprocal())
        assert numpy.allclose(self.sc_la[:2, :2], (self.tmat @ self.uc_la)[:2, :2], atol=3e-2)
        assert numpy.allclose(self.uc_bz[:2, :2], (self.tmat.T @ self.sc_bz)[:2, :2], atol=3e-2)

    def set_kpts_in_unitcell(
        self,
        kpts: numpy.ndarray,
        format: str = "fractional",
    ):
        """Set the k-points to evaluate, given in the primitive-cell BZ.

        Cartesian coordinates are without the 2π prefactor (i.e. in units of Å⁻¹).

        Args:
            kpts (numpy.ndarray): K-points, shape ``(nkpts, 3)``.
            format (str): ``"fractional"`` (default) or ``"cartesian"``.
        """
        if format == "fractional":
            self.kpts_cart = numpy.matmul(kpts, self.uc_bz)
            self.kpts_uc_frac = kpts
            self.kpts_sc_frac = numpy.matmul(self.kpts_uc_frac, self.tmat.T)
        elif format == "cartesian":
            self.kpts_cart = kpts
            self.kpts_uc_frac = numpy.matmul(kpts, numpy.linalg.inv(self.uc_bz))
            self.kpts_sc_frac = numpy.matmul(kpts, numpy.linalg.inv(self.sc_bz))
        else:
            raise ValueError(f"format={format!r} not supported")
        assert numpy.allclose(self.kpts_uc_frac @ self.uc_bz, self.kpts_cart)
        assert numpy.allclose(self.kpts_sc_frac @ self.sc_bz, self.kpts_cart)

    def calculate_sc_phonon(
        self,
        dyn_sc: DynamicalMatrix | DynamicalMatrixNAC,
        factor: float | str = VASP_TO_EV,
        save_fpath: str | None = None,
        show_progress: bool = False,
    ):
        """Diagonalise the supercell dynamical matrix along the set k-path.

        This is the most expensive step.

        Args:
            dyn_sc: Dynamical matrix from ``phonopy.dynamical_matrix``.
            factor (float or str): Energy unit conversion. Strings: ``"ev"``, ``"mev"``,
                ``"thz"``, ``"cm"``. Default: ``VASP_TO_EV``.
            save_fpath (str, optional): Path to save results as ``.npz``.
            show_progress (bool): If True, diagonalise k-points one at a time with a
                progress bar. If False (default), diagonalise all k-points in a
                single Phonopy call (faster, but progress cannot be tracked since
                it is internal to Phonopy).
        """
        if isinstance(factor, float):
            pass
        elif isinstance(factor, str):
            factor = factor.lower()
            assert factor in ("ev", "mev", "thz", "cm"), f"factor={factor!r} not supported"
            factor = {"ev": VASP_TO_EV, "mev": VASP_TO_EV * 1e3, "thz": VASP_TO_THZ, "cm": VASP_TO_CM}[factor]
        else:
            raise ValueError(f"factor={factor!r} not supported")

        time_start = time.time()
        if show_progress:
            iterator = tqdm(self.kpts_sc_frac, desc="Diagonalizing") if self.verbose else self.kpts_sc_frac
            energies_list = []
            eigenvecs_list = []
            for kpt in iterator:
                bs_sc = BandStructure(
                    paths=[[kpt]],
                    dynamical_matrix=dyn_sc,
                    with_eigenvectors=True,
                    factor=factor,
                )
                energies_list.append(bs_sc.frequencies[0][0])
                eigenvecs_list.append(bs_sc.eigenvectors[0][0])
            self.bs_sc_energies = numpy.array(energies_list)
            self.bs_sc_eigenvecs = numpy.array(eigenvecs_list)
        else:
            bs_sc = BandStructure(
                paths=[self.kpts_sc_frac],
                dynamical_matrix=dyn_sc,
                with_eigenvectors=True,
                factor=factor,
            )
            self.bs_sc_energies = bs_sc.frequencies[0]
            self.bs_sc_eigenvecs = bs_sc.eigenvectors[0]
        time_end = time.time()
        print(
            f"Band structure: {time_end - time_start:.2f}s for {len(self.kpts_sc_frac)} k-points "
            f"({(time_end - time_start) / len(self.kpts_sc_frac):.3f}s/k-point)."
        )
        if save_fpath is not None:
            if not save_fpath.endswith(".npz"):
                print("WARNING: save_fpath should end with .npz - appending.")
                save_fpath += ".npz"
            os.makedirs(os.path.dirname(save_fpath), exist_ok=True)
            numpy.savez(
                file=save_fpath,
                bs_sc_energies=self.bs_sc_energies,
                bs_sc_eigenvecs=self.bs_sc_eigenvecs,
                kpts_sc_frac=self.kpts_sc_frac,
                factor=factor,
            )

    def load_sc_phonon(self, save_fpath: str) -> float:
        """Load a previously saved supercell phonon band structure.

        Args:
            save_fpath (str): Path to the ``.npz`` file written by :meth:`calculate_sc_phonon`.

        Returns:
            float: The energy conversion factor used when the file was saved.
        """
        assert os.path.exists(save_fpath), f"File {save_fpath} does not exist."
        data = numpy.load(save_fpath, allow_pickle=True)
        assert data["bs_sc_energies"].shape[0] == self.kpts_sc_frac.shape[0]
        assert data["bs_sc_energies"].shape[1] == len(self.sc) * 3
        assert data["bs_sc_eigenvecs"].shape[1] == len(self.sc) * 3
        self.bs_sc_energies = data["bs_sc_energies"]
        self.bs_sc_eigenvecs = data["bs_sc_eigenvecs"]
        assert numpy.allclose(self.kpts_sc_frac, data["kpts_sc_frac"])
        return float(data["factor"])

    def _calculate_weights_one_kpt(self, kpt_idx: int) -> numpy.ndarray:
        uc_natoms = len(self.uc)
        sc_natoms = len(self.sc)
        gen_natoms = len(self.sc_by_tmat)

        uc_modes = numpy.diag(numpy.ones(3 * uc_natoms)).reshape(uc_natoms, 3, 3 * uc_natoms)
        uc2gen_modes = numpy.tile(uc_modes, (self.nucs_in_sc, 1, 1)).reshape(3 * gen_natoms, 3 * uc_natoms)

        sc_modes = self.bs_sc_eigenvecs[kpt_idx]
        nbands = sc_modes.shape[-1]
        sc_modes = sc_modes.reshape(sc_natoms, 3, nbands)

        # Gather sc atoms into generated-supercell order, zero-padding vacant sites
        # (where perm_sc2gen == -1) so they contribute nothing to the projection.
        sc2gen_modes = numpy.zeros((gen_natoms, 3, nbands), dtype=sc_modes.dtype)
        valid = self.perm_sc2gen >= 0
        sc2gen_modes[valid] = sc_modes[self.perm_sc2gen[valid]]
        sc2gen_modes = sc2gen_modes.reshape(3 * gen_natoms, nbands)

        weights = numpy.einsum("in,ib->nb", uc2gen_modes.conj(), sc2gen_modes)
        weights = (numpy.abs(weights) ** 2).sum(axis=0)

        return weights / self.nucs_in_sc

    def calculate_weights(self):
        """Calculate spectral weights for all k-points.

        Results are stored in ``self.weights``, shape ``(nkpts, sc_nbands)``.
        """
        weights = []
        iterator = (
            tqdm(range(len(self.kpts_uc_frac)), desc="Projecting") if self.verbose else range(len(self.kpts_uc_frac))
        )
        for kpt_idx in iterator:
            weights.append(self._calculate_weights_one_kpt(kpt_idx))
        self.weights = numpy.array(weights)

    def _calculate_spectral_function_on_grid_one_kpt(
        self,
        kpt_idx: int,
        grid: numpy.ndarray,
        sigma: float,
    ) -> numpy.ndarray:
        weights = self.weights[kpt_idx]
        sc_energies = self.bs_sc_energies[kpt_idx]
        sc_energies_expanded = band_expansion(energies=sc_energies, grid=grid, sigma=sigma)
        return numpy.einsum("j,jk->k", weights, sc_energies_expanded)

    def calculate_spectral_function_on_grid(
        self,
        grid: numpy.ndarray | None = None,
        sigma: float | None = None,
    ) -> tuple[numpy.ndarray, float]:
        """Project weighted supercell bands onto an energy grid.

        If ``grid`` and ``sigma`` are None, sensible defaults are chosen automatically
        from the range of ``bs_sc_energies``.

        Args:
            grid (numpy.ndarray, optional): 1D energy grid.
            sigma (float, optional): Gaussian broadening width (same units as energies).

        Returns:
            tuple: ``(grid, sigma)`` the grid and broadening used.

        After calling, ``self.spectral_function_on_grid`` has shape ``(nkpts, ngrid)``.
        """
        if grid is None and sigma is None:
            _div = (self.bs_sc_energies.max() - self.bs_sc_energies.min()) / 2000
            _overshoot = self.bs_sc_energies.max() * 0.05
            grid = numpy.arange(
                self.bs_sc_energies.min() - _overshoot,
                self.bs_sc_energies.max() + _overshoot,
                _div,
            )
            sigma = 5 * _div
        else:
            assert isinstance(grid, numpy.ndarray) and grid.ndim == 1
            assert isinstance(sigma, float) and sigma > 0

        iterator = (
            tqdm(range(len(self.kpts_uc_frac)), desc="Building Grids")
            if self.verbose
            else range(len(self.kpts_uc_frac))
        )
        spectral_function_on_grid = []
        for kpt_idx in iterator:
            spectral_function_on_grid.append(self._calculate_spectral_function_on_grid_one_kpt(kpt_idx, grid, sigma))
        self.spectral_function_on_grid = numpy.stack(spectral_function_on_grid, axis=0)
        return grid, sigma

    def save(self, fpath: str):
        """Serialise the Unfold object to disk (pickle).

        Args:
            fpath (str): Output path.
        """
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, fpath: str) -> "Unfold":
        """Load a serialised Unfold object.

        Args:
            fpath (str): Path written by :meth:`save`.

        Returns:
            Unfold: Deserialised object.

        Raises:
            TypeError: If the loaded object is not an Unfold instance.
        """
        with open(fpath, "rb") as f:
            loaded_obj = pickle.load(f)
        if not isinstance(loaded_obj, cls):
            raise TypeError(f"Loaded object is not an instance of {cls.__name__}")
        return loaded_obj
