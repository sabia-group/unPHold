"""Graphene monolayer vacancy unfolding: recover the 2-atom primitive-cell dispersion from a 9x9 cell
with one missing atom.

Data: tests/data/graphene/
    uc_1_sc_9_mace/          - pristine cell, force constants from a 9x9x1 supercell (reference bands)
    vacancy_uc_9_sc_1_mace/  - 9x9x1 cell (161 atoms) with a single carbon vacancy, used as the unfolding source
Each phonon calculation directory holds its relaxed primitive cell as gp_pc.xyz, alongside
phonopy.yaml and force_constants.h5.

A defect cell has fewer atoms than the ideal primitive-cell tiling it derives from, so
``Unfold``'s automatic (equal-count) atom matching cannot be used. Instead we build the
``perm_sc2gen`` index array explicitly with ``match_atoms_with_vacancies``, which marks the
vacancy site with ``-1``. This matching step is the crux of unfolding a defective structure.

Writes figures to examples/output/, regardless of the current working directory:
    python examples/unfold_graphene_vacancy.py --kpts 21
    # denser k-point sampling for better visual quality of unfolded spectral function:
    python examples/unfold_graphene_vacancy.py --kpts 201
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.build import make_supercell
from ase.io import read
from matplotlib.colors import Normalize
from phonopy.cui.load import load as load_phonopy
from phonopy.file_IO import read_force_constants_hdf5
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from unphold import Unfold
from unphold.utils import atoms_ph2ase, concatenate_bands, match_atoms_with_vacancies
from unphold.visualize import visualize_cell_2d

DATA = Path(__file__).parent.parent / "tests" / "data" / "graphene"
OUTPUT_DEFAULT = Path(__file__).parent / "output"
TMAT = np.diag([9, 9, 1])  # 2-atom primitive cell -> ideal 9x9x1 tiling (162 sites)

# Hexagonal monolayer: Γ–M–K–Γ
KPATH = [
    [
        [0.0, 0.0, 0.0],  # Γ
        [1 / 2, 0.0, 0.0],  # M
        [2 / 3, 1 / 3, 0.0],  # K
        [0.0, 0.0, 0.0],  # Γ
    ]
]
KLABELS = ["Γ", "M", "K", "Γ"]

ENERGY_GRID = np.arange(-5.0, 55.0001, 0.05)
SIGMA = 0.15


def load(name, primitive_matrix=None):
    d = DATA / name
    ph = load_phonopy(d / "phonopy.yaml", primitive_matrix=primitive_matrix)
    ph.force_constants = read_force_constants_hdf5(d / "force_constants.h5")
    return ph


def _decorate_ax(ax, k_dist, hsp_x, grid):
    for x in hsp_x:
        ax.axvline(x, color="gray", linestyle=":", linewidth=0.7)
    ax.set_xticks(hsp_x)
    ax.set_xticklabels(KLABELS)
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylim(grid[0], grid[-1])
    ax.set_ylabel("Frequency (THz)")
    ax.set_xlabel("k-path in primitive cell BZ")


def _overlay_bands(ax, bs, color="red", alpha=0.5, lw=0.8):
    for dist_seg, freq_seg in zip(bs.distances, bs.frequencies, strict=True):
        for b in range(freq_seg.shape[1]):
            ax.plot(dist_seg, freq_seg[:, b], color=color, alpha=alpha, linewidth=lw)


def main(output: Path, npoints: int = 21):
    atoms_pc = read(DATA / "vacancy_uc_9_sc_1_mace" / "gp_pc.xyz")  # relaxed 2-atom primitive cell
    ph_pc = load("uc_1_sc_9_mace")  # only for the PC reference bands
    ph_vac = load("vacancy_uc_9_sc_1_mace")

    sc_real = atoms_ph2ase(ph_vac.unitcell)  # 161-atom cell with one vacancy

    # Build the ideal, defect-free 9x9x1 tiling, then match the real (relaxed, vacancy-bearing)
    # cell against it. perm is ideal-indexed with real-valued entries, -1 at the vacancy site.
    sc_by_tmat = make_supercell(atoms_pc, TMAT, wrap=False, order="cell-major")
    match = match_atoms_with_vacancies(ideal=sc_by_tmat, real=sc_real, spatial_tolerance=0.5)
    assert match["fail_reason"] is None, match["fail_reason"]
    perm = match["perm_real2ideal"]
    vac_ideal_idx = match["vacancy_indices"]

    n_ideal = len(sc_by_tmat)
    n_real = len(sc_real)
    n_vac = len(vac_ideal_idx)
    n_matched = int((perm >= 0).sum())
    print(f"Ideal tiling sites: {n_ideal}")
    print(f"Real cell atoms:    {n_real}")
    print(f"Vacancies:          {n_vac}")
    print(f"Matched atoms:      {n_matched}")

    unfold = Unfold(
        unitcell=atoms_pc,
        supercell=sc_real,
        transformation_matrix=TMAT,
        perm_sc2gen=perm,
        verbose=True,
    )

    kpts_uc, connections = get_band_qpoints_and_path_connections(KPATH, npoints=npoints)
    kpts_flat, bz_idx = concatenate_bands(kpts_uc, connections)

    unfold.set_kpts_in_unitcell(kpts_flat, format="fractional")
    unfold.calculate_sc_phonon(dyn_sc=ph_vac.dynamical_matrix, factor="thz", show_progress=True)
    unfold.calculate_weights()

    # Weight conservation with a vacancy
    weight_sums = unfold.weights.sum(axis=1)
    weight_sum_expected = 3 * len(atoms_pc) - 3 * n_vac / unfold.nucs_in_sc
    print(f"N_uc (cells in sc): {unfold.nucs_in_sc}")
    print(f"Weight sum expected: {weight_sum_expected:.5f}")
    print(f"Weight sum actual:   {weight_sums.mean():.5f}")
    print(f"Max |deviation| over k-points: {np.abs(weight_sums - weight_sum_expected).max():.2e}")

    grid, _ = unfold.calculate_spectral_function_on_grid(grid=ENERGY_GRID, sigma=SIGMA)
    spectral = unfold.spectral_function_on_grid  # (nkpts, ngrid)

    # Pristine primitive-cell bands, for the physics reference overlay.
    ph_pc.run_band_structure(kpts_uc, path_connections=connections)
    bs_pc = ph_pc._band_structure

    k_dist = np.concatenate([d[:-1] if c else d for d, c in zip(bs_pc.distances, connections, strict=True)])
    hsp_x = k_dist[bz_idx]
    norm = Normalize(vmin=0, vmax=np.percentile(spectral, 99.5))

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    # Figure: atomic-structure matching
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    vac_xy = sc_by_tmat.get_positions()[vac_ideal_idx, :2]
    ax = axes[0]
    visualize_cell_2d(sc_by_tmat, ax=ax)
    for i, xy in enumerate(sc_by_tmat.get_positions()[:, :2]):
        if perm[i] >= 0:
            ax.annotate(str(perm[i]), xy, textcoords="offset points", xytext=(2, 2), fontsize=12)
        else:
            ax.annotate("-1", xy, textcoords="offset points", xytext=(8, 5), fontsize=12, color="red")
    ax.scatter(vac_xy[:, 0], vac_xy[:, 1], marker="x", s=120, color="red", zorder=5, label="vacancy")
    ax.legend(loc="upper left")
    ax.set_title("Ideal 9x9 tiling, labels = perm_sc2gen values")
    ax = axes[1]
    visualize_cell_2d(sc_real, ax=ax)
    for i, xy in enumerate(sc_real.get_positions()[:, :2]):
        ax.annotate(str(i), xy, textcoords="offset points", xytext=(2, 2), fontsize=12)
    ax.set_title("Relaxed cell (1 vacancy), labels = atom indices")
    fig.tight_layout()
    fig.savefig(out / "graphene_vacancy_matching.png", dpi=300)
    print(f"Saved {out / 'graphene_vacancy_matching.png'}")
    plt.close(fig)

    # Figure: unfolded spectral function + pristine PC bands
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.pcolormesh(k_dist, grid, spectral.T, cmap="Blues", norm=norm, shading="nearest")
    plt.colorbar(im, ax=ax, label=r"$A(\mathbf{k}, \omega)$ [arb.]")
    _overlay_bands(ax, bs_pc)
    _decorate_ax(ax, k_dist, hsp_x, grid)
    ax.set_title("Unfolded vacancy phonons + pristine bands (red)")
    fig.tight_layout()
    fig.savefig(out / "graphene_vacancy_unfolded.png", dpi=300)
    print(f"Saved {out / 'graphene_vacancy_unfolded.png'}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graphene monolayer vacancy phonon unfolding example")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DEFAULT),
        help=f"Directory for output figures (default: {OUTPUT_DEFAULT})",
    )
    parser.add_argument(
        "--kpts",
        type=int,
        default=21,
        help="Number of k-points per k-path segment (default: 21)",
    )
    args = parser.parse_args()
    main(output=Path(args.output), npoints=args.kpts)
