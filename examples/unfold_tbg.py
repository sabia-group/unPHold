"""Twisted bilayer graphene (TBG) phonon unfolding: unfold a moire supercell onto one layer's primitive cell.

Physics check: unfolding the TBG phonon band structure onto one layer's primitive cell (PC)
should approximately reproduce Bernal bilayer graphene (BLG) bands, since each individual
layer of a lightly twisted bilayer still looks locally like a graphene monolayer. A
two-layer breathing (ZO') mode near ~2.3 THz should also show up as an out-of-plane,
optical (low APR), high-weight candidate at Gamma.

Data (MACE MLIP force constants, from finite-difference phonon runs):
    --data-blg   - Bernal bilayer case dir, 4-atom unit cell == its own SC
                   (phonopy.yaml, force_constants.h5)
    --data-tbg   - TBG moire case dir, treated as the phonopy "unitcell"
                   (phonopy.yaml, force_constants.h5, tmat.npz, gp_pc.xyz)

Defaults to the tests/data/{blg,tbg}/... fixtures and writes figures to examples/output/
(gitignored), regardless of the current working directory:
    python examples/unfold_tbg.py
    # m_6_r_1_sc_1_mace (508 atoms) is much more expensive to diagonalize; restrict to G->K only:
    python examples/unfold_tbg.py --data-tbg tests/data/tbg/m_6_r_1_sc_1_mace --kpath GK --output-prefix tbg_m6r1
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
from ase.build import make_supercell
from ase.io import read
from matplotlib.colors import Normalize
from phonopy.cui.load import load as load_phonopy
from phonopy.file_IO import read_force_constants_hdf5
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from unphold import Unfold
from unphold.metrics import compute_APR, compute_V, compute_V_p2
from unphold.utils import (
    atoms_ph2ase,
    calculate_pc_rotation_angle,
    concatenate_bands,
    match_two_2d_atoms_pbc_with_2d_frac_shift,
)
from unphold.visualize import plot_layer_mode_2d, visualize_BZ_2d, visualize_cell_2d, visualize_kpath_2d

_TESTS_DATA = Path(__file__).parent.parent / "tests" / "data"
DATA_BLG_DEFAULT = _TESTS_DATA / "blg" / "AB_uc_1_sc_9_mace"
DATA_TBG_DEFAULT = _TESTS_DATA / "tbg" / "m_2_r_1_sc_4_mace"
OUTPUT_DEFAULT = Path(__file__).parent / "output"

# graphene hexagonal BZ, fractional coordinates
GP_SPECIAL_POINTS = {
    "G": [0.0, 0.0, 0.0],
    "M": [1 / 2, 0.0, 0.0],
    "K": [2 / 3, 1 / 3, 0.0],
}
KPATH_DEFAULT = "GKMG"

FREQ_WINDOW = (2.0, 2.6)  # THz, window around the expected layer-breathing-mode frequency
WEIGHT_MIN = 0.01  # only keep Gamma modes with non-negligible layer0 unfolding weight
DETAILS_FREQ_MAX = 8.0  # THz, y-limit for the G->K/2 low-energy detail plot


def load(case_dir: Path, primitive_matrix=None):
    ph = load_phonopy(case_dir / "phonopy.yaml", primitive_matrix=primitive_matrix)
    ph.force_constants = read_force_constants_hdf5(case_dir / "force_constants.h5")
    return ph


def _decorate_ax(ax, k_dist, hsp_x, labels):
    for x in hsp_x:
        ax.axvline(x, color="gray", linestyle=":", linewidth=0.7)
    ax.set_xticks(hsp_x)
    ax.set_xticklabels(labels)
    ax.set_xlim(k_dist[0], k_dist[-1])
    ax.set_ylabel("Frequency (THz)")


def _overlay_bands(ax, distances, frequencies, color="k", alpha=1.0, lw=0.8):
    for dist_seg, freq_seg in zip(distances, frequencies, strict=True):
        for b in range(freq_seg.shape[1]):
            ax.plot(dist_seg, freq_seg[:, b], color=color, alpha=alpha, linewidth=lw)


def _overlay_bands_flat(ax, k_dist, freqs, color="k", alpha=1.0, lw=0.8):
    for b in range(freqs.shape[1]):
        ax.plot(k_dist, freqs[:, b], color=color, alpha=alpha, linewidth=lw)


def main(
    blg_dir: Path,
    tbg_dir: Path,
    path_labels: list,
    npoints: int,
    output: Path,
    output_prefix: str,
):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    atoms_gp_pc = read(tbg_dir / "gp_pc.xyz")
    ph_blg = load(blg_dir)
    ph_tbg = load(tbg_dir)
    tmat_l0 = dict(**numpy.load(tbg_dir / "tmat.npz", allow_pickle=True))["tmat_l0"]
    atoms_tbg_uc = atoms_ph2ase(ph_tbg.unitcell)

    kpath = [[GP_SPECIAL_POINTS[label] for label in path_labels]]
    kpts_uc_segs, connections = get_band_qpoints_and_path_connections(kpath, npoints=npoints)
    kpts_uc_flat, bz_idx = concatenate_bands(kpts_uc_segs, connections)

    # --- Figure 1: BLG (reference) band structure, direct diagonalization of the 4-atom UC ---
    ph_blg.run_band_structure(kpts_uc_segs, path_connections=connections, labels=path_labels)
    bs_blg = ph_blg._band_structure
    k_dist = numpy.concatenate([d[:-1] if c else d for d, c in zip(bs_blg.distances, connections, strict=True)])
    hsp_x = k_dist[bz_idx]

    fig, ax = plt.subplots(figsize=(5, 4))
    _overlay_bands(ax, bs_blg.distances, bs_blg.frequencies)
    _decorate_ax(ax, k_dist, hsp_x, path_labels)
    ax.set_xlabel("k-path in Bernal bilayer BZ")
    ax.set_title("Bernal bilayer graphene phonon bands")
    fig.tight_layout()
    fig.savefig(out / f"{output_prefix}_blg_uc_bands.png", dpi=300)
    print(f"Saved {out / f'{output_prefix}_blg_uc_bands.png'}")
    plt.close(fig)

    # --- Align the graphene PC's orientation with the TBG layer0 supercell ---
    ret_pc_rot = calculate_pc_rotation_angle(atoms_gp_pc, tmat_l0)
    atoms_pc_rot = ret_pc_rot["atoms_pc_rot"]
    print(f"PC rotation angle (deg): {ret_pc_rot['rot_angle_deg']:.3f}")

    sc_from_pc = make_supercell(atoms_gp_pc, tmat_l0)
    sc_from_pc_rot = make_supercell(atoms_pc_rot, tmat_l0, order="cell-major")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    visualize_cell_2d(sc_from_pc, ax=axes[0])
    axes[0].set_title("from PC")
    visualize_cell_2d(sc_from_pc_rot, ax=axes[1])
    axes[1].set_title("from PC with rotation")
    visualize_cell_2d(atoms_tbg_uc, ax=axes[2])
    axes[2].set_title("target SC (TBG)")
    fig.tight_layout()
    fig.savefig(out / f"{output_prefix}_tmat_pc_and_pc_rot.png", dpi=300)
    print(f"Saved {out / f'{output_prefix}_tmat_pc_and_pc_rot.png'}")
    plt.close(fig)

    # --- Figure: BZ + k-path comparison, rotated PC vs TBG SC (repeated-zone scheme) ---
    n_rep = int(numpy.ceil(numpy.sqrt(abs(numpy.linalg.det(tmat_l0)) / 3)))  # SC BZ area ~ 1/det(tmat) of PC BZ area
    fig, ax = plt.subplots(figsize=(5, 5))
    visualize_BZ_2d(atoms_pc_rot, plt_kwargs={"color": "C0"}, ax=ax)
    visualize_BZ_2d(atoms_tbg_uc, plt_kwargs={"color": "C1"}, ax=ax, repeat=(n_rep, n_rep))
    visualize_kpath_2d(
        atoms_pc_rot, fractional_coordinate=numpy.concatenate(kpts_uc_segs), plt_kwargs={"color": "r"}, ax=ax
    )
    ax.autoscale(tight=True)
    ax.margins(0)
    fig.tight_layout()
    fig.savefig(out / f"{output_prefix}_vis_kpath_blg_tbg_2d.png", dpi=300)
    print(f"Saved {out / f'{output_prefix}_vis_kpath_blg_tbg_2d.png'}")
    plt.close(fig)

    # --- Slice layer0 out of the TBG SC and match it to the rotated-PC supercell image ---
    layer0_indices = numpy.where(atoms_tbg_uc.positions[:, 2] < atoms_tbg_uc.positions[:, 2].mean())[0]
    atoms_layer0 = atoms_tbg_uc[layer0_indices]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    visualize_cell_2d(sc_from_pc_rot, ax=axes[0])
    axes[0].set_title(f"supercell from PC, {len(sc_from_pc_rot)} atoms")
    visualize_cell_2d(atoms_layer0, ax=axes[1])
    axes[1].set_title(f"layer0 sliced from TBG, {len(atoms_layer0)} atoms")
    fig.tight_layout()
    fig.savefig(out / f"{output_prefix}_tbg_l0.png", dpi=300)
    print(f"Saved {out / f'{output_prefix}_tbg_l0.png'}")
    plt.close(fig)

    match_result = match_two_2d_atoms_pbc_with_2d_frac_shift(
        atoms_layer0,
        sc_from_pc_rot,
        shift_0_frac=0.005,
        shift_0_seg=6,
        shift_1_frac=0.005,
        shift_1_seg=6,
        spatial_tolerance=1.0,  # moire relaxation causes real out-of-plane bending
        tolerance_xyz_scaler=numpy.array([0.5, 0.5, 1.5]),
        ignore_z=False,
    )
    perm_sc2gen_l0 = layer0_indices[match_result["atoms_indices_a2b"]]

    # --- Unfold TBG SC phonons onto the layer0 PC ---
    unfold = Unfold(
        unitcell=atoms_pc_rot,
        supercell=atoms_tbg_uc,
        transformation_matrix=tmat_l0,
        perm_sc2gen=perm_sc2gen_l0,
        verbose=True,
    )
    unfold.set_kpts_in_unitcell(kpts_uc_flat, format="fractional")
    unfold.calculate_sc_phonon(dyn_sc=ph_tbg.dynamical_matrix, factor="thz", show_progress=True)
    unfold.calculate_weights()

    # --- Figure: TBG SC band structure (raw diagonalization, folded) ---
    # unfold.bs_sc_energies was diagonalized at kpts_uc_flat, which is aligned 1:1 with k_dist.
    fig, ax = plt.subplots(figsize=(5, 4))
    _overlay_bands_flat(ax, k_dist, unfold.bs_sc_energies, lw=0.4)
    _decorate_ax(ax, k_dist, hsp_x, path_labels)
    ax.set_xlabel("k-path in layer0 PC BZ")
    ax.set_title("TBG phonon bands (folded)")
    fig.tight_layout()
    fig.savefig(out / f"{output_prefix}_tbg_uc_bands.png", dpi=300)
    print(f"Saved {out / f'{output_prefix}_tbg_uc_bands.png'}")
    plt.close(fig)

    # --- Figure: unfolded spectral function vs BLG reference bands, full k-path ---
    grid, _ = unfold.calculate_spectral_function_on_grid()
    spectral = unfold.spectral_function_on_grid  # (nkpts, ngrid)
    norm = Normalize(vmin=0, vmax=numpy.percentile(spectral, 99.5))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.pcolormesh(k_dist, grid, spectral.T, cmap="Blues", norm=norm, shading="nearest")
    fig.colorbar(im, ax=ax, label=r"$A(\mathbf{k}, \omega)$ [arb.]")
    _overlay_bands(ax, bs_blg.distances, bs_blg.frequencies, color="red", alpha=0.5)
    _decorate_ax(ax, k_dist, hsp_x, path_labels)
    ax.set_xlabel("k-path in layer0 PC BZ")
    ax.set_ylim(grid[0], grid[-1])
    ax.set_title("Unfolded (TBG -> layer0) + BLG bands (red)")
    fig.tight_layout()
    fig.savefig(out / f"{output_prefix}_tbg_unfolded_vs_blg.png", dpi=300)
    print(f"Saved {out / f'{output_prefix}_tbg_unfolded_vs_blg.png'}")
    plt.close(fig)

    # --- Figure: same data, zoomed to the first half-segment and low energy (breathing-mode region) ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.pcolormesh(k_dist, grid, spectral.T, cmap="Blues", norm=norm, shading="nearest")
    fig.colorbar(im, ax=ax, label=r"$A(\mathbf{k}, \omega)$ [arb.]")
    _overlay_bands(ax, bs_blg.distances, bs_blg.frequencies, color="red", alpha=0.5)
    _decorate_ax(ax, k_dist, hsp_x, path_labels)
    ax.set_xlabel("k-path in layer0 PC BZ")
    half_dist = hsp_x[1] / 2  # halfway along the first path segment
    ax.set_xlim(k_dist[0], half_dist)
    ax.set_ylim(0.0, DETAILS_FREQ_MAX)
    seg_label = f"{path_labels[0]}{path_labels[1]}"
    ax.set_title(f"Unfolded (TBG -> layer0) + BLG bands (red), {path_labels[0]}->{path_labels[1]}/2 detail")
    fig.tight_layout()
    fig.savefig(out / f"{output_prefix}_tbg_unfolded_vs_blg_{seg_label}_details.png", dpi=300)
    print(f"Saved {out / f'{output_prefix}_tbg_unfolded_vs_blg_{seg_label}_details.png'}")
    plt.close(fig)

    # --- Identify layer-breathing-mode candidates at Gamma ---
    gamma_freqs = unfold.bs_sc_energies[0]
    gamma_weights = unfold.weights[0]
    cand_mask = (gamma_freqs > FREQ_WINDOW[0]) & (gamma_freqs < FREQ_WINDOW[1]) & (gamma_weights > WEIGHT_MIN)
    cand_idx = numpy.where(cand_mask)[0]

    cand_eigvecs = unfold.bs_sc_eigenvecs[0:1, :, cand_idx]  # (1, natoms3, ncand)
    apr_cand = compute_APR(atoms=unfold.sc, ph_eigvecs=cand_eigvecs)[0]
    v_cand = compute_V(atoms=unfold.sc, ph_eigvecs=cand_eigvecs)[0]
    vp2_cand = compute_V_p2(atoms=unfold.sc, ph_eigvecs=cand_eigvecs)[0]

    print(f"candidates in {FREQ_WINDOW} THz window with weight > {WEIGHT_MIN}: {len(cand_idx)}")
    print(f"{'band':>5} {'freq (THz)':>12} {'APR':>8} {'V':>8} {'V_p2':>8} {'weight':>8}")
    for i, b in enumerate(cand_idx):
        print(
            f"{b:5d} {gamma_freqs[b]:12.4f} {apr_cand[i]:8.4f} {v_cand[i]:8.4f} {vp2_cand[i]:8.4f} "
            f"{gamma_weights[b]:8.4f}"
        )

    # --- Figure(s): candidate breathing-mode displacement patterns, layer0 vs layer1 ---
    sc_positions = unfold.sc.get_positions()
    z_mean = sc_positions[:, 2].mean()
    idx_bottom = numpy.where(sc_positions[:, 2] < z_mean)[0]
    idx_top = numpy.where(sc_positions[:, 2] >= z_mean)[0]
    atoms_bottom = unfold.sc[idx_bottom]
    atoms_top = unfold.sc[idx_top]

    for b in cand_idx:
        b = int(b)
        eigvec = unfold.bs_sc_eigenvecs[0, :, b]
        imag_frac = numpy.abs(eigvec.imag).max() / (numpy.abs(eigvec.real).max() + 1e-12)
        if imag_frac > 1e-3:
            print(f"warning: band {b} has non-negligible imaginary part (max|Im|/max|Re| = {imag_frac:.2e})")
        disp = eigvec.real.reshape(-1, 3)
        freq = unfold.bs_sc_energies[0, b]

        vmax = numpy.abs(disp[:, 2]).max()  # shared out-of-plane amplitude range across both layers
        mode_norm = Normalize(vmin=-vmax, vmax=vmax)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        plot_layer_mode_2d(atoms_bottom, disp[idx_bottom], freqs=freq, norm=mode_norm, add_colorbar=False, axes=axes[0])
        mappables = plot_layer_mode_2d(
            atoms_top, disp[idx_top], freqs=freq, norm=mode_norm, add_colorbar=False, axes=axes[1]
        )
        axes[0].set_title(f"band {b}, freq = {freq:.4f} THz, bottom layer")
        axes[1].set_title(f"band {b}, freq = {freq:.4f} THz, top layer")
        fig.colorbar(mappables[0], ax=axes, label="out-of-plane amplitude (z)", shrink=0.8)
        fig.savefig(out / f"{output_prefix}_viz_lbm_{b}.png", dpi=300)
        print(f"Saved {out / f'{output_prefix}_viz_lbm_{b}.png'}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TBG phonon unfolding example")
    parser.add_argument(
        "--data-blg",
        default=str(DATA_BLG_DEFAULT),
        help=f"Path to the Bernal bilayer case dir, containing phonopy.yaml + force_constants.h5 "
        f"(default: {DATA_BLG_DEFAULT})",
    )
    parser.add_argument(
        "--data-tbg",
        default=str(DATA_TBG_DEFAULT),
        help="Path to the TBG moire case dir, containing phonopy.yaml + force_constants.h5 + "
        f"tmat.npz + gp_pc.xyz (default: {DATA_TBG_DEFAULT})",
    )
    parser.add_argument(
        "--kpath",
        default=KPATH_DEFAULT,
        help=f"High-symmetry k-path, one letter per point from {sorted(GP_SPECIAL_POINTS)} "
        f"(default: {KPATH_DEFAULT}); e.g. GK for just Gamma->K (much cheaper for large cells like m6_r1)",
    )
    parser.add_argument(
        "--kpts",
        default=21,
        type=int,
        help="Number of k-points per k-path segment (default: 21)",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DEFAULT),
        help=f"Directory for output figures (default: {OUTPUT_DEFAULT}); use ../docs/assets to update tutorial figures",
    )
    parser.add_argument(
        "--output-prefix",
        default="tbg",
        help="Prefix for output figure filenames (default: tbg)",
    )
    args = parser.parse_args()
    path_labels = list(args.kpath)
    if len(path_labels) < 2:
        parser.error(f"--kpath must have at least 2 points, got {args.kpath!r}")
    unknown = [label for label in path_labels if label not in GP_SPECIAL_POINTS]
    if unknown:
        parser.error(f"--kpath has unknown label(s) {unknown}; must be one of {sorted(GP_SPECIAL_POINTS)}")

    main(
        blg_dir=Path(args.data_blg),
        tbg_dir=Path(args.data_tbg),
        path_labels=path_labels,
        npoints=args.kpts,
        output=Path(args.output),
        output_prefix=args.output_prefix,
    )
