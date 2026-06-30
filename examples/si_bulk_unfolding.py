"""Si bulk phonon unfolding: recover 2-atom primitive-cell dispersion from a 2×2×2 supercell.

Data: tests/data/si/
    uc_1_sc_2_aims/  — 2-atom primitive cell, force constants from 2×2×2 SC displacements
    uc_2_sc_1_aims/  — 2×2×2 SC as unit cell, used as the source for unfolding

Run from unPHold/:
    python examples/si_bulk_unfolding.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from phonopy.cui.load import load as load_phonopy
from phonopy.file_IO import read_force_constants_hdf5
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from unphold import Unfold
from unphold.utils import atoms_ph2ase, concatenate_bands

DATA = Path(__file__).parent.parent / "tests" / "data" / "si"
TMAT = np.diag([2, 2, 2])

# FCC Si: Γ–X–U|K–Γ–L
KPATH = [
    [
        [0.0, 0.0, 0.0],  # Γ
        [0.5, 0.0, 0.5],  # X
        [0.625, 0.25, 0.625],  # U
        [1.0, 1.0, 1.0],  # Γ (zone boundary continuation)
        [0.5, 0.5, 0.5],  # L
    ]
]
KLABELS = ["Γ", "X", "U|K", "Γ", "L"]


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
    for dist_seg, freq_seg in zip(bs.distances, bs.frequencies):
        for b in range(freq_seg.shape[1]):
            ax.plot(dist_seg, freq_seg[:, b], color=color, alpha=alpha, linewidth=lw)


def main(output: Path):
    ph_uc = load("uc_1_sc_2_aims")
    ph_sc = load("uc_2_sc_1_aims", primitive_matrix="P")

    kpts_uc, connections = get_band_qpoints_and_path_connections(KPATH, npoints=51)
    kpts_flat, bz_idx = concatenate_bands(kpts_uc, connections)
    kpts_sc = [k @ TMAT.T for k in kpts_uc]

    ph_uc.run_band_structure(kpts_uc, path_connections=connections)
    ph_sc.run_band_structure(kpts_sc, path_connections=connections)

    bs_uc = ph_uc._band_structure
    bs_sc = ph_sc._band_structure
    k_dist = np.concatenate([d[:-1] if c else d for d, c in zip(bs_sc.distances, connections)])

    unfold = Unfold(
        unitcell=atoms_ph2ase(ph_uc.unitcell),
        supercell=atoms_ph2ase(ph_sc.unitcell),
        transformation_matrix=TMAT,
        verbose=True,
    )
    unfold.set_kpts_in_unitcell(kpts_flat, format="fractional")
    unfold.calculate_sc_phonon(dyn_sc=ph_sc.dynamical_matrix, factor="thz")
    unfold.calculate_weights()

    grid, _ = unfold.calculate_spectral_function_on_grid(grid=np.arange(-1.0, 18.0, 0.01), sigma=0.1)
    spectral = unfold.spectral_function_on_grid  # (nkpts, ngrid)

    hsp_x = k_dist[bz_idx]
    norm = Normalize(vmin=0, vmax=np.percentile(spectral, 99.5))

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: UC band structure (reference) ---
    fig, ax = plt.subplots(figsize=(5, 4))
    _overlay_bands(ax, bs_uc, color="k", alpha=1.0, lw=1.0)
    _decorate_ax(ax, k_dist, hsp_x, grid)
    ax.set_title("Si primitive cell phonon bands")
    fig.tight_layout()
    fig.savefig(out / "si_uc_bands.png", dpi=300)
    print(f"Saved {out / 'si_uc_bands.png'}")
    plt.close(fig)

    # --- Figure 2: SC band structure (showing band folding) ---
    fig, ax = plt.subplots(figsize=(5, 4))
    _overlay_bands(ax, bs_sc, color="k", alpha=1.0, lw=0.6)
    _decorate_ax(ax, k_dist, hsp_x, grid)
    ax.set_title("Si 2×2×2 supercell bands (folded)")
    fig.tight_layout()
    fig.savefig(out / "si_sc_bands.png", dpi=300)
    print(f"Saved {out / 'si_sc_bands.png'}")
    plt.close(fig)

    # --- Figure 3: unfolded spectral function ---
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.pcolormesh(k_dist, grid, spectral.T, cmap="Blues", norm=norm, shading="nearest")
    plt.colorbar(im, ax=ax, label=r"$A(\mathbf{k}, \omega)$ [arb.]")
    _overlay_bands(ax, bs_sc)
    _decorate_ax(ax, k_dist, hsp_x, grid)
    ax.set_title("Unfolded + SC bands (red)")
    fig.tight_layout()
    fig.savefig(out / "si_unfolded_vs_sc.png", dpi=300)
    print(f"Saved {out / 'si_unfolded_vs_sc.png'}")
    plt.close(fig)

    # --- Figure 4: unfolded spectral function + UC bands overlay ---
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.pcolormesh(k_dist, grid, spectral.T, cmap="Blues", norm=norm, shading="nearest")
    plt.colorbar(im, ax=ax, label=r"$A(\mathbf{k}, \omega)$ [arb.]")
    _overlay_bands(ax, bs_uc)
    _decorate_ax(ax, k_dist, hsp_x, grid)
    ax.set_title("Unfolded + UC bands (red)")
    fig.tight_layout()
    fig.savefig(out / "si_unfolded_vs_uc.png", dpi=300)
    print(f"Saved {out / 'si_unfolded_vs_uc.png'}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Si bulk phonon unfolding example")
    parser.add_argument(
        "--output",
        default="output",
        help="Directory for output figures (default: output); use ../docs/assets to update tutorial figures",
    )
    args = parser.parse_args()
    main(output=args.output)
