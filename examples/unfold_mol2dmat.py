"""Unfold the phonons of a MePTCDI molecular layer on graphene onto both sublattices.

The input is a fully relaxed (atoms and cell) commensurate cell: 232 graphene C atoms
plus four MePTCDI (N,N'-dimethylperylene-3,4,9,10-tetracarboxylic diimide) molecules of
46 atoms each. Force constants come from MACE-MH-1 finite displacements on a 3x3x1
supercell. The same supercell phonons are unfolded twice:

1. Graphene primitive cell (PC). The relaxed lattice is slightly sheared, so the PC is
   simply ``inv(TMAT_GRAPHENE) @ cell`` (det 116, 232 C atoms) with the two basis atoms
   at ideal honeycomb fractional sites; the small registry offset of the real layer is
   absorbed by the position matching that builds ``perm_sc2gen``.
2. Molecular PC (one molecule). The molecule centers sit on an oblique lattice,
   ``TMAT_MOL = [[2, -1], [0, 2]]`` (det 4), not a naive 2x2. Each molecule is
   individually rotated in-plane and neighbors alternate in 180-degree flips, so the
   atom correspondence is built molecule by molecule: rotate the isolated symmetrized
   molecule (``relaxed_sym.xyz``) onto each one and match species-resolved nearest
   neighbors, which keeps the CH3 hydrogens unambiguous.

Graphene is unfolded along a k-path (G-M-K-G by default). The molecular bands are
nearly flat, so the molecular unfolding runs on a uniform mesh instead, and the
k-averaged unfolded density of states (DOS) is compared with the DOS of the isolated
molecule.

The case dir needs ``phonopy.yaml`` + ``force_constants.h5``; the molecule dir
additionally provides ``relaxed_sym.xyz``. The derived graphene PC and the
transformation matrices are written back into the case dir, figures go to
``examples/output``.

Usage:
    python examples/unfold_mol2dmat.py                          # quick test settings
    python examples/unfold_mol2dmat.py --kpts 101 --mol-mesh 6  # production
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from ase import Atoms
from ase.build import make_supercell
from ase.io import read, write
from ase.neighborlist import NeighborList, natural_cutoffs
from matplotlib.colors import Normalize
from phonopy import Phonopy
from phonopy.cui.load import load as load_phonopy
from phonopy.file_IO import read_force_constants_hdf5
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from unphold import Unfold
from unphold.utils import (
    atoms_ph2ase,
    band_expansion,
    concatenate_bands,
    match_two_2d_atoms_pbc_with_2d_frac_shift,
)
from unphold.visualize import visualize_BZ_2d, visualize_cell_2d, visualize_kpath_2d

_TESTS_DATA = Path(__file__).parent.parent / "tests" / "data"
DATA_DEFAULT = _TESTS_DATA / "mol2dmat" / "graphene_MePTCDI_2x2_mace"
DATA_MOL_DEFAULT = _TESTS_DATA / "mol2dmat" / "MePTCDI_mace"
OUTPUT_DEFAULT = Path(__file__).parent / "output"

A_GRAPHENE = 2.46  # ideal graphene lattice constant (Angstrom), 60-degree cell convention

# graphene PC -> supercell, in-plane; det = 116 -> 232 C atoms
TMAT_GRAPHENE = numpy.array([[8, -4, 0], [-1, 15, 0], [0, 0, 1]])
# molecular PC -> supercell; det = 4 -> 4 molecules (COM lattice (1/2, 1/4) and (0, 1/2))
TMAT_MOL = numpy.array([[2, -1, 0], [0, 2, 0], [0, 0, 1]])

# graphene hexagonal BZ (60-degree cell convention), fractional coordinates
GR_SPECIAL_POINTS = {
    "G": [0.0, 0.0, 0.0],
    "M": [1 / 2, 0.0, 0.0],
    "K": [2 / 3, 1 / 3, 0.0],
}
GR_KPATH_DEFAULT = "GMKG"

DETAILS_FREQ_MAX = 16.0  # THz, upper limit of the low-frequency detail figures and DOS zoom


def load(case_dir: Path):
    """simple wrapper to load the phonopy object with force constants"""
    ph = load_phonopy(case_dir / "phonopy.yaml", produce_fc=False)
    ph.force_constants = read_force_constants_hdf5(case_dir / "force_constants.h5")
    return ph


def cell_params_2d(cell: numpy.ndarray) -> tuple:
    """In-plane lattice parameters (a, b, gamma in degrees) of a 3x3 cell matrix."""
    la = numpy.linalg.norm(cell[0, :2])
    lb = numpy.linalg.norm(cell[1, :2])
    gamma = numpy.degrees(numpy.arccos(numpy.dot(cell[0, :2], cell[1, :2]) / (la * lb)))
    return la, lb, gamma


def split_layers(atoms_sc: Atoms) -> tuple:
    """Split graphene layer and molecular layer by the largest gap in z."""
    z = atoms_sc.positions[:, 2]
    z_sorted = numpy.sort(z)
    i_gap = numpy.argmax(numpy.diff(z_sorted))  # index of the largest gap in z
    z_cut = (z_sorted[i_gap] + z_sorted[i_gap + 1]) / 2  # mean as the cutting plane
    idx_graphene = numpy.where(z < z_cut)[0]
    idx_mol = numpy.where(z >= z_cut)[0]
    symbols = numpy.array(atoms_sc.get_chemical_symbols())
    assert set(symbols[idx_graphene]) == {"C"}, "graphene layer is expected to be pure carbon"
    return idx_graphene, idx_mol


def build_graphene_pc(atoms_sc: Atoms, idx_graphene: numpy.ndarray) -> Atoms:
    """find grpahene primitive cell by inverse of the transformation matrix"""
    gp_pc_cell_from_atoms_sc = numpy.linalg.solve(TMAT_GRAPHENE, atoms_sc.cell.array)
    print("graphene primitive cell by TMAT^{-1}:\n", gp_pc_cell_from_atoms_sc)

    # human readable data for the graphene primitive cell
    la, lb, gamma = cell_params_2d(gp_pc_cell_from_atoms_sc)
    print(f"graphene primitive cell: a={la:.4f} b={lb:.4f}, a/b difference: {200*abs(la - lb)/(la+lb):.4f} %")
    print(f"graphene primitive cell: gamma {gamma:.3f} degrees, expected 60 degrees")

    # construct the graphene primitive cell, intentionally displace atomic positions in PC
    gp_pc_positions = numpy.zeros((2, 3))
    gp_pc_positions[0,:2] = (gp_pc_cell_from_atoms_sc[0,:2] + gp_pc_cell_from_atoms_sc[1,:2]) * 2 / 3
    gp_pc_positions[1,:2] = (gp_pc_cell_from_atoms_sc[0,:2] + gp_pc_cell_from_atoms_sc[1,:2])
    gp_pc_positions[:,2] = atoms_sc.positions[idx_graphene].mean(axis=0)[2]  # actually not necessary, just to be safe
    return Atoms("C2", cell=gp_pc_cell_from_atoms_sc, positions=gp_pc_positions, pbc=True)

def find_molecules(atoms_sc: Atoms, idx_mol: numpy.ndarray) -> list:
    """Identify molecules as connected components and unwrap each across the PBC.

    Returns:
        list of Atoms: One Atoms per molecule with COM-centered positions;
        ``atoms.info`` holds ``idx_global`` (supercell atom indices), ``com``
        (center in the supercell frame), and ``angle`` (in-plane principal-axis
        angle of the heavy atoms, in [0, pi)).
    """
    atoms_layer = atoms_sc[idx_mol]
    cell = atoms_sc.cell.array
    inv_cell = numpy.linalg.inv(cell)
    nl = NeighborList(natural_cutoffs(atoms_layer, mult=1.2), self_interaction=False, bothways=True)
    nl.update(atoms_layer)
    component = numpy.full(len(atoms_layer), -1)
    n_mol = 0
    for seed in range(len(atoms_layer)):
        if component[seed] >= 0:
            continue
        stack = [seed]
        component[seed] = n_mol
        while stack:
            i = stack.pop()
            for j in nl.get_neighbors(i)[0]:
                if component[j] < 0:
                    component[j] = n_mol
                    stack.append(j)
        n_mol += 1

    molecules = []
    symbols_layer = numpy.array(atoms_layer.get_chemical_symbols())
    for c in range(n_mol):
        local = numpy.where(component == c)[0]
        pos = atoms_layer.positions[local].copy()
        # unwrap: a molecule cut by the cell boundary is reassembled by moving every
        # atom to its periodic image nearest to the first atom (valid while the
        # molecule fits within half a lattice vector in each periodic direction)
        d = (pos - pos[0]) @ inv_cell  # displacements from the first atom, fractional
        d[:, :2] -= numpy.round(d[:, :2])  # this finds the small displacement against a integer lattice vector
        pos = pos[0] + d @ cell  # back to Cartesian, molecule now contiguous since d[:2] is in [-0.5, 0.5)
        symbols = symbols_layer[local]
        heavy = pos[symbols != "H", :2]
        _, _, vt = numpy.linalg.svd(heavy - heavy.mean(axis=0), full_matrices=False)  # find major principal axis
        com = pos.mean(axis=0)
        molecule = Atoms(symbols=symbols, positions=pos - com)  # COM-centered; supercell-frame COM kept in info
        molecule.info = {
            "idx_global": idx_mol[local],
            "com": com,
            "angle": numpy.mod(numpy.arctan2(vt[0, 1], vt[0, 0]), numpy.pi),
        }
        molecules.append(molecule)
    print(f"molecular layer: {n_mol} molecules, {len(atoms_layer)} atoms")
    return molecules


def build_molecular_pc(atoms_sc: Atoms, molecules: list, atoms_iso: Atoms) -> tuple:
    """Build the molecular PC and each molecule's atom correspondence to it.

    Rotations are wrapped to [-90, 90) degrees and anchored to molecule 0, never
    undoing a 180-degree flip of the nearly C2-symmetric molecule.

    Args:
        atoms_sc: The relaxed supercell.
        molecules: COM-centered molecules from `find_molecules`.
        atoms_iso: The isolated symmetrized molecule used as the common reference.

    Returns:
        tuple: `(atoms_pc_mol, molecules)`. The PC is the reference molecule rotated
        onto molecule 0; each molecule gains `info["corr"]`, where `corr[kappa]` is
        the molecule-local atom matching reference atom `kappa`.
    """
    pc_cell = numpy.linalg.solve(TMAT_MOL, atoms_sc.cell.array)
    print("molecular primitive cell by TMAT^{-1}:\n", pc_cell)

    iso_pos = atoms_iso.positions - atoms_iso.positions.mean(axis=0)
    iso_sym = numpy.array(atoms_iso.get_chemical_symbols())
    heavy = iso_pos[iso_sym != "H", :2]
    _, _, vt = numpy.linalg.svd(heavy - heavy.mean(axis=0), full_matrices=False)
    iso_angle = numpy.mod(numpy.arctan2(vt[0, 1], vt[0, 0]), numpy.pi)
    print(
        "molecular primitive cell: angles of the molecules in the layer (degrees):",
        numpy.degrees([m.info["angle"] for m in molecules]),
    )
    print("molecular primitive cell: angle of the isolated molecule (degrees):", numpy.degrees(iso_angle))

    def wrap_pm90(angle: float) -> float:
        return (angle + numpy.pi / 2) % numpy.pi - numpy.pi / 2

    def rotate_xy(pos: numpy.ndarray, angle: float) -> numpy.ndarray:
        rot = numpy.array([[numpy.cos(angle), -numpy.sin(angle)], [numpy.sin(angle), numpy.cos(angle)]])
        out = pos.copy()
        out[:, :2] = out[:, :2] @ rot.T
        return out

    """
    rotate the reference onto each molecule and find the atom correspondence; anchor
    all rotations to molecule 0, so that a molecule whose angle falls on the other
    side of the +-90 degree wrap boundary cannot come out C2-flipped
    """
    d_angle_0 = wrap_pm90(molecules[0].info["angle"] - iso_angle)
    for m in molecules:
        d_angle = d_angle_0 + wrap_pm90(m.info["angle"] - molecules[0].info["angle"])
        aligned = rotate_xy(iso_pos, d_angle)  # both COM-centered, no translation needed
        symbols = numpy.array(m.get_chemical_symbols())
        corr = numpy.full(len(iso_sym), -1)
        for species in numpy.unique(iso_sym):  # per atomic species, avoid the -CH3 being ambiguous
            ia = numpy.where(iso_sym == species)[0]
            ib = numpy.where(symbols == species)[0]
            dist = numpy.linalg.norm(aligned[ia][:, None] - m.positions[ib][None, :], axis=2)
            nearest = numpy.argmin(dist, axis=1)
            assert len(numpy.unique(nearest)) == len(ia), (
                f"nearest-neighbor correspondence not bijective for species {species}; "
                "the molecule deviates too much from the reference orientation"
            )
            corr[ia] = ib[nearest]
        m.info["corr"] = corr
        residual = numpy.linalg.norm(aligned - m.positions[corr], axis=1)
        print(
            f"  molecule at {numpy.degrees(m.info['angle']):.2f} degrees: reference rotated by "
            f"{numpy.degrees(d_angle):+.2f} degrees, distortion vs gas phase max {residual.max():.3f} Angstrom"
        )

    # PC basis: the reference molecule rotated onto molecule 0, placed at its center
    basis = rotate_xy(iso_pos, d_angle_0) + molecules[0].info["com"]
    return Atoms(symbols=iso_sym, positions=basis, cell=pc_cell, pbc=True), molecules


def build_perm_graphene(atoms_sc: Atoms, idx_graphene: numpy.ndarray, atoms_pc: Atoms) -> numpy.ndarray:
    """Match the generated graphene supercell to the sliced layer, return ``perm_sc2gen``."""
    sc_generated = make_supercell(atoms_pc, TMAT_GRAPHENE, wrap=False)
    match = match_two_2d_atoms_pbc_with_2d_frac_shift(
        atoms_sc[idx_graphene],
        sc_generated,
        shift_0_frac=0.001,
        shift_0_seg=1,
        shift_1_frac=0.001,
        shift_1_seg=1,
        spatial_tolerance=0.3,
        ignore_z=False,
    )
    assert "atoms_indices_a2b" in match, "graphene layer matching failed"
    print(f"graphene layer matched, max residual {match['atoms_dist_matched'].max():.4f} Angstrom")
    return idx_graphene[match["atoms_indices_a2b"]]


def build_perm_molecular(atoms_sc: Atoms, atoms_pc_mol: Atoms, molecules: list) -> numpy.ndarray:
    """Assign each generated PC copy to a relaxed molecule by COM, return ``perm_sc2gen``."""
    cell = atoms_sc.cell.array
    inv_cell = numpy.linalg.inv(cell)
    sc_generated = make_supercell(atoms_pc_mol, TMAT_MOL, wrap=False)  # DO NOT WRAP! Or COM gets wrong
    n_copies = len(sc_generated) // len(atoms_pc_mol)
    perm = numpy.full(len(sc_generated), -1)
    assigned = set()
    for copy in range(n_copies):
        segment = slice(copy * len(atoms_pc_mol), (copy + 1) * len(atoms_pc_mol))
        distances = []
        for m in molecules:  # find which molecule is closest to this generated copy by COM
            d = (m.info["com"] - sc_generated.positions[segment].mean(axis=0)) @ inv_cell
            d[:2] -= numpy.round(d[:2])  # min-image in-plane: now d[:2] in [-0.5, 0.5)
            distances.append(numpy.linalg.norm(d @ cell))
        c = int(numpy.argmin(distances))
        assert c not in assigned, "two generated copies mapped to the same molecule"
        assigned.add(c)
        perm[segment] = molecules[c].info["idx_global"][molecules[c].info["corr"]]  # brain drained for this slice...
    assert len(numpy.unique(perm)) == len(perm), "perm_sc2gen is not injective"
    return perm


def make_kpath(atoms_pc: Atoms, special_points: dict, path_labels: list, npoints: int) -> dict:
    """Sample the k-path and compute the plot axis in the PC BZ."""
    kpath = [[special_points[label] for label in path_labels]]
    kpts_segs, connections = get_band_qpoints_and_path_connections(kpath, npoints=npoints)
    kpts_flat, bz_idx = concatenate_bands(kpts_segs, connections)
    kpts_cart = kpts_flat @ numpy.array(atoms_pc.cell.reciprocal())
    k_dist = numpy.concatenate([[0.0], numpy.cumsum(numpy.linalg.norm(numpy.diff(kpts_cart, axis=0), axis=1))])
    return {
        "kpts_segs": kpts_segs,
        "kpts_flat": kpts_flat,
        "k_dist": k_dist,
        "hsp_x": k_dist[bz_idx],
    }


def _decorate_ax(ax, k_dist, hsp_x, labels):
    for x in hsp_x:
        ax.axvline(x, color="gray", linestyle=":", linewidth=0.7)
    ax.set_xticks(hsp_x)
    ax.set_xticklabels(labels)
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylabel("Frequency (THz)")


def plot_match(atoms_generated: Atoms, atoms_layer: Atoms, titles: tuple, fpath: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    visualize_cell_2d(atoms_generated, ax=axes[0])
    axes[0].set_title(f"{titles[0]}, {len(atoms_generated)} atoms")
    visualize_cell_2d(atoms_layer, ax=axes[1])
    axes[1].set_title(f"{titles[1]}, {len(atoms_layer)} atoms")
    fig.tight_layout()
    fig.savefig(fpath, dpi=300)
    print(f"Saved {fpath}")
    plt.close(fig)


def plot_bz_kpath(atoms_pc: Atoms, atoms_sc: Atoms, kpts_segs: list, tmat: numpy.ndarray, fpath: Path):
    n_rep = int(numpy.ceil(numpy.sqrt(abs(numpy.linalg.det(tmat)) / 3)))
    fig, ax = plt.subplots(figsize=(3, 3))
    visualize_BZ_2d(atoms_pc, plt_kwargs={"color": "C0"}, ax=ax)
    visualize_BZ_2d(atoms_sc, plt_kwargs={"color": "C1"}, ax=ax, repeat=(int(n_rep/2**0.5), int(n_rep*2**0.5)))
    visualize_kpath_2d(atoms_pc, fractional_coordinate=numpy.concatenate(kpts_segs), plt_kwargs={"color": "r"}, ax=ax)
    ax.autoscale(tight=True)
    ax.margins(0)
    fig.tight_layout()
    fig.savefig(fpath, dpi=300)
    print(f"Saved {fpath}")
    plt.close(fig)


def visualize_gamma_motion(
    unfold: Unfold,
    kpt_idx: int,
    atoms_sc: Atoms,
    idx_graphene: numpy.ndarray,
    idx_mol: numpy.ndarray,
    band_indices: list,
    out: Path,
    prefix: str,
):
    """Draw the Gamma-point displacement pattern of selected bands, one figure each.

    Reuses the k-path unfolding data at ``kpt_idx`` (the Gamma point of the path), so
    frequencies, eigenvectors, and unfolding weights come from one diagonalization.
    Each figure has the graphene layer on the left and the molecular layer on the
    right (top view): in-plane motion as arrows, out-of-plane motion as the marker
    color. Displacements are eigenvectors divided by sqrt(mass), normalized to the
    largest amplitude in the supercell, so the two panels share the same scale. The
    figure title carries the unfolding weight of the band.

    Args:
        unfold: The k-path unfolding with weights already calculated.
        kpt_idx: Index of the Gamma point in the unfolding k-point list.
        atoms_sc: The supercell.
        idx_graphene: Supercell indices of the graphene layer.
        idx_mol: Supercell indices of the molecular layer.
        band_indices: Bands to draw, ascending frequency order at Gamma (0 = first
            acoustic mode).
        out: Output directory.
        prefix: Output filename prefix.
    """
    frequencies = unfold.bs_sc_energies[kpt_idx]
    eigenvectors = unfold.bs_sc_eigenvecs[kpt_idx]
    weights = unfold.weights[kpt_idx]
    # numpy.savez(out / f"{prefix}_gamma_weights.npz", weights=weights, frequencies=frequencies)
    n_bands = len(frequencies)
    masses = atoms_sc.get_masses()
    cell = atoms_sc.cell.array
    corners = numpy.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ cell[:2, :2]

    for band in band_indices:
        assert 0 <= band < n_bands, f"band index {band} out of range 0..{n_bands - 1}"
        mode = eigenvectors[:, band]
        mode = mode * numpy.exp(-1j * numpy.angle(mode[numpy.argmax(numpy.abs(mode))]))  # fix the global phase
        disp = mode.real.reshape(-1, 3) / numpy.sqrt(masses)[:, None]  # mass-weighted eigvec -> displacement
        disp /= numpy.abs(disp).max()

        fig, axes = plt.subplots(1, 2, figsize=(9.5, 6.0), constrained_layout=True)
        panels = ((axes[0], idx_graphene, "graphene layer"), (axes[1], idx_mol, "molecular layer"))
        for ax, idx, title in panels:
            pos = atoms_sc.positions[idx]
            scatter = ax.scatter(
                pos[:, 0], pos[:, 1], c=disp[idx, 2], cmap="coolwarm", vmin=-1, vmax=1, s=14, zorder=2
            )
            ax.quiver(
                pos[:, 0], pos[:, 1], disp[idx, 0], disp[idx, 1],
                angles="xy", scale_units="xy", scale=1 / 3.0, width=0.004, color="black", zorder=3,
            )
            ax.plot(corners[:, 0], corners[:, 1], color="grey", lw=0.8, zorder=1)
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.set_xlabel("x (Angstrom)")
        axes[0].set_ylabel("y (Angstrom)")
        fig.colorbar(scatter, ax=axes, shrink=0.8, label="out-of-plane displacement (normalized)")
        fig.suptitle(f"Gamma mode {band}: {frequencies[band]:.3f} THz, unfolding weight {weights[band]:.3f}")
        fpath = out / f"{prefix}_gamma_mode_{band:03d}.png"
        fig.savefig(fpath, dpi=300)
        print(f"Saved {fpath}")
        plt.close(fig)


def unfold_and_plot(
    label: str,
    atoms_pc: Atoms,
    atoms_sc: Atoms,
    tmat: numpy.ndarray,
    perm: numpy.ndarray,
    ph: Phonopy,
    path: dict,
    path_labels: list,
    out: Path,
    prefix: str,
    viz_idx: list = None,
    idx_graphene: numpy.ndarray = None,
    idx_mol: numpy.ndarray = None,
) -> Unfold:
    """Run the unfolding and plot folded supercell bands plus the spectral function.

    With ``viz_idx``, additionally draw the Gamma-point motion of those bands, reusing
    this unfolding's eigenvectors and weights (the k-path must contain Gamma).
    """
    unfold = Unfold(atoms_pc, atoms_sc, tmat, perm_sc2gen=perm, verbose=True)
    unfold.set_kpts_in_unitcell(path["kpts_flat"], format="fractional")
    unfold.calculate_sc_phonon(dyn_sc=ph.dynamical_matrix, factor="thz", show_progress=True)
    unfold.calculate_weights()
    weight_sum = unfold.weights.sum(axis=1)
    print(
        f"{label}: total weight per k-point {weight_sum.min():.3f}..{weight_sum.max():.3f} (expect {3 * len(atoms_pc)})"
    )

    if viz_idx:
        at_gamma = numpy.where(numpy.linalg.norm(path["kpts_flat"], axis=1) < 1e-9)[0]
        assert len(at_gamma) > 0, "the k-path must contain Gamma to visualize Gamma-point modes"
        visualize_gamma_motion(unfold, int(at_gamma[0]), atoms_sc, idx_graphene, idx_mol, viz_idx, out, prefix)

    k_dist, hsp_x = path["k_dist"], path["hsp_x"]

    fig, ax = plt.subplots(figsize=(5, 4))
    for band in range(unfold.bs_sc_energies.shape[1]):
        ax.plot(k_dist, unfold.bs_sc_energies[:, band], color="k", linewidth=0.3)
    _decorate_ax(ax, k_dist, hsp_x, path_labels)
    ax.set_xlabel(f"k-path in the {label} PC BZ")
    ax.set_title(f"Supercell phonon bands (folded), {label} path")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_folded_bands_{label}.png", dpi=300)
    print(f"Saved {out / f'{prefix}_folded_bands_{label}.png'}")
    plt.close(fig)

    spectral, grid, _ = unfold.calculate_spectral_function_on_grid()
    norm = Normalize(vmin=0, vmax=numpy.percentile(spectral, 99.5))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.pcolormesh(k_dist, grid, spectral.T, cmap="Blues", norm=norm, shading="nearest")
    fig.colorbar(im, ax=ax, label=r"$A(\mathbf{k}, \omega)$ [arb.]")
    _decorate_ax(ax, k_dist, hsp_x, path_labels)
    ax.set_xlabel(f"k-path in the {label} PC BZ")
    ax.set_ylim(grid[0], grid[-1])
    ax.set_title(f"Unfolded onto the {label} PC")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_unfolded_{label}.png", dpi=300)
    print(f"Saved {out / f'{prefix}_unfolded_{label}.png'}")
    plt.close(fig)

    # low-frequency detail on a finer grid: this is where the intermolecular and flexural dispersion lives
    grid_fine = numpy.arange(min(-1.0, unfold.bs_sc_energies.min()), DETAILS_FREQ_MAX, 0.01)
    spectral_fine, grid_fine, _ = unfold.calculate_spectral_function_on_grid(grid=grid_fine, sigma=0.05)
    norm = Normalize(vmin=0, vmax=numpy.percentile(spectral_fine, 99.5))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.pcolormesh(k_dist, grid_fine, spectral_fine.T, cmap="Blues", norm=norm, shading="nearest")
    fig.colorbar(im, ax=ax, label=r"$A(\mathbf{k}, \omega)$ [arb.]")
    _decorate_ax(ax, k_dist, hsp_x, path_labels)
    ax.set_xlabel(f"k-path in the {label} PC BZ")
    ax.set_ylim(grid_fine[0], grid_fine[-1])
    ax.set_title(f"Unfolded onto the {label} PC, low-frequency detail")
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_unfolded_{label}_details.png", dpi=300)
    print(f"Saved {out / f'{prefix}_unfolded_{label}_details.png'}")
    plt.close(fig)
    return unfold


def unfold_mol_dos(
    atoms_pc_mol: Atoms,
    atoms_sc: Atoms,
    perm: numpy.ndarray,
    ph:Phonopy,
    ph_iso:Phonopy,
    mesh: int,
    out: Path,
    prefix: str,
) -> Unfold:
    """Unfold on a uniform molecular-BZ k-mesh and compare the DOS with the isolated molecule.

    The k-averaged unfolded spectral function is the molecular layer's DOS as seen by
    the molecular PC projector; the reference is the Gaussian-broadened mode spectrum of
    the isolated MePTCDI molecule. Both integrate to 3 x 46 = 138 modes, so the two
    curves are directly comparable.
    """
    kpts = numpy.array([[i / mesh, j / mesh, 0.0] for i in range(mesh) for j in range(mesh)])
    print(f"molecular DOS: {mesh}x{mesh} Gamma-centered mesh in the molecular BZ")
    unfold = Unfold(atoms_pc_mol, atoms_sc, TMAT_MOL, perm_sc2gen=perm, verbose=True)
    unfold.set_kpts_in_unitcell(kpts, format="fractional")
    unfold.calculate_sc_phonon(dyn_sc=ph.dynamical_matrix, factor="thz", show_progress=True)
    unfold.calculate_weights()
    weight_sum = unfold.weights.sum(axis=1)
    print(
        f"mol: total weight per k-point {weight_sum.min():.3f}..{weight_sum.max():.3f} (expect {3 * len(atoms_pc_mol)})"
    )

    freqs_iso = ph_iso.get_frequencies((0.0, 0.0, 0.0))

    e_min = min(unfold.bs_sc_energies.min(), freqs_iso.min())
    e_max = max(unfold.bs_sc_energies.max(), freqs_iso.max())
    windows = (
        (numpy.arange(e_min - 2.0, e_max + 3.0, 0.05), 0.15, "full range"),
        (numpy.arange(min(-1.0, e_min), DETAILS_FREQ_MAX, 0.01), 0.05, "low-frequency detail"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.5))
    for ax, (grid, sigma, title) in zip(axes, windows, strict=True):
        spectral, grid, sigma = unfold.calculate_spectral_function_on_grid(grid=grid, sigma=sigma)
        dos_unfolded = spectral.mean(axis=0)
        dos_iso = band_expansion(energies=freqs_iso, grid=grid, sigma=sigma).sum(axis=0)
        ax.plot(grid, dos_unfolded, color="C0", label="unfolded molecular layer")
        ax.plot(grid, dos_iso, color="C3", alpha=0.7, label="isolated MePTCDI")
        ax.set_xlim(grid[0], grid[-1])
        ax.set_xlabel("Frequency (THz)")
        ax.set_ylabel(r"DOS (arb.)")
        ax.set_title(f"Molecular DOS, {title} (sigma = {sigma:.2f} THz)")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{prefix}_dos_mol.png", dpi=300)
    print(f"Saved {out / f'{prefix}_dos_mol.png'}")
    plt.close(fig)
    return unfold


def main(
    data_dir: Path,
    data_mol_dir: Path,
    gr_path_labels: list,
    npoints: int,
    mol_mesh: int,
    output: Path,
    prefix: str,
    viz_idx: list = None,
):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    ph = load(data_dir)
    atoms_sc = atoms_ph2ase(ph.unitcell)
    idx_graphene, idx_mol = split_layers(atoms_sc)
    print(f"supercell: {len(atoms_sc)} atoms, graphene {len(idx_graphene)}, molecules {len(idx_mol)}")

    #### graphene sublattice
    atoms_pc_gp = build_graphene_pc(atoms_sc, idx_graphene)
    perm_gr = build_perm_graphene(atoms_sc, idx_graphene, atoms_pc_gp)
    plot_match(
        make_supercell(atoms_pc_gp, TMAT_GRAPHENE),
        atoms_sc[idx_graphene],
        ("tiled from the sheared graphene PC", "sliced from the supercell"),
        out / f"{prefix}_match_graphene.png",
    )

    #### molecular sublattice, referenced to the isolated symmetrized molecule
    atoms_iso = read(data_mol_dir / "relaxed_sym.xyz")
    molecules = find_molecules(atoms_sc, idx_mol)
    atoms_pc_mol, molecules = build_molecular_pc(atoms_sc, molecules, atoms_iso)
    perm_mol = build_perm_molecular(atoms_sc, atoms_pc_mol, molecules)
    plot_match(
        make_supercell(atoms_pc_mol, TMAT_MOL, wrap=False),
        atoms_sc[idx_mol],
        ("tiled from the molecular PC", "sliced from the supercell"),
        out / f"{prefix}_match_mol.png",
    )


    # deposit the derived graphene PC and the transformation matrices next to the
    # phonon data, so the dataset is self-contained for tests and downstream scripts
    numpy.savez(data_dir / "tmat.npz", tmat_graphene=TMAT_GRAPHENE, tmat_mol=TMAT_MOL)
    print(f"Saved tmat.npz to {data_dir}")

    # --- graphene: k-path unfolding ---
    path_gp = make_kpath(atoms_pc_gp, GR_SPECIAL_POINTS, gr_path_labels, npoints)
    plot_bz_kpath(atoms_pc_gp, atoms_sc, path_gp["kpts_segs"], TMAT_GRAPHENE, out / f"{prefix}_bz_kpath_graphene.png")
    unfold_and_plot(
        "graphene",
        atoms_pc_gp,
        atoms_sc,
        TMAT_GRAPHENE,
        perm_gr,
        ph,
        path_gp,
        gr_path_labels,
        out,
        prefix,
        viz_idx=viz_idx,
        idx_graphene=idx_graphene,
        idx_mol=idx_mol,
    )

    # --- molecule: DOS comparison against the isolated molecule ---
    ph_iso = load(data_mol_dir)
    unfold_mol_dos(atoms_pc_mol, atoms_sc, perm_mol, ph, ph_iso, mol_mesh, out, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molecule-on-graphene phonon unfolding example")
    parser.add_argument(
        "--data",
        default=str(DATA_DEFAULT),
        help=f"Case dir with phonopy.yaml + force_constants.h5 (default: {DATA_DEFAULT})",
    )
    parser.add_argument(
        "--data-molecule",
        default=str(DATA_MOL_DEFAULT),
        help=f"Isolated-molecule dir with phonopy.yaml + force_constants.h5 + relaxed_sym.xyz "
        f"(default: {DATA_MOL_DEFAULT})",
    )
    parser.add_argument(
        "--gp-kpath",
        default=GR_KPATH_DEFAULT,
        help=f"Graphene k-path, letters from {sorted(GR_SPECIAL_POINTS)} (default: {GR_KPATH_DEFAULT})",
    )
    parser.add_argument(
        "--kpts",
        default=4,
        type=int,
        help="k-points per path segment (default: 4, small for testing; use 21 or more for production)",
    )
    parser.add_argument(
        "--mol-mesh",
        default=3,
        type=int,
        help="molecular-BZ mesh for the DOS, NxN Gamma-centered (default: 3; use 6 or more for production)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DEFAULT),
        help=f"Directory for output figures (default: {OUTPUT_DEFAULT})",
    )
    parser.add_argument(
        "--output-prefix",
        default="mol2dmat",
        help="Prefix for output figure filenames (default: mol2dmat)",
    )
    parser.add_argument(
        "--viz-idx",
        nargs="+",
        type=int,
        default=None,
        help="Gamma-point band indices to visualize, one figure each, e.g. --viz-idx 0 3 6 (default: none)",
    )
    args = parser.parse_args()

    gp_labels = list(args.gp_kpath)
    if len(gp_labels) < 2:
        parser.error(f"--gp-kpath must have at least 2 points, got {args.gp_kpath!r}")
    unknown = [label for label in gp_labels if label not in GR_SPECIAL_POINTS]
    if unknown:
        parser.error(f"--gp-kpath has unknown label(s) {unknown}; must be from {sorted(GR_SPECIAL_POINTS)}")

    main(
        data_dir=Path(args.data),
        data_mol_dir=Path(args.data_molecule),
        gr_path_labels=gp_labels,
        npoints=args.kpts,
        mol_mesh=args.mol_mesh,
        output=Path(args.output),
        prefix=args.output_prefix,
        viz_idx=args.viz_idx,
    )
