"""Phonon mode character metrics on monolayer graphene.

For each metric (APR, longitudinality L, verticality V_p2) the script plots the
primitive-cell band structure colored by the metric value, with the full band
structure drawn in grey underneath and a colorbar for the metric.

Data: tests/data/graphene/uc_1_sc_9_mace/
Description: 2-atom primitive cell, force constants for a 9x9x1 supercell

Writes figures to examples/output/ (gitignored), regardless of the current working directory:
    python examples/metrics_graphene.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from phonopy.cui.load import load as load_phonopy
from phonopy.file_IO import read_force_constants_hdf5
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections

from unphold.metrics import compute_APR_from_phonopy, compute_L_from_phonopy, compute_V_from_phonopy

DATA = Path(__file__).parent.parent / "tests" / "data" / "graphene" / "uc_1_sc_9_mace"
OUTPUT_DEFAULT = Path(__file__).parent / "output"

KPATH = [
    [
        [0.0, 0.0, 0.0],  # G
        [0.5, 0.0, 0.0],  # M
        [2 / 3, 1 / 3, 0.0],  # K
        [0.0, 0.0, 0.0],  # G
    ]
]
KLABELS = ["G", "M", "K", "G"]

CMAP = LinearSegmentedColormap.from_list("Reds_trunc", plt.get_cmap("Reds")(np.linspace(0.25, 1.0, 256)))


def _decorate_ax(ax, hsp_x, freq_max):
    for x in hsp_x[1:-1]:
        ax.axvline(x, color="gray", linestyle=":", linewidth=0.7)
    ax.set_xticks(hsp_x)
    ax.set_xticklabels(KLABELS)
    ax.set_xlim(hsp_x[0], hsp_x[-1])
    ax.set_ylim(-2.0, freq_max)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.7)
    ax.set_ylabel("Frequency (THz)")
    ax.set_xlabel("k-path in primitive cell BZ")


def _colored_bands(ax, bs, metric_segs, norm, cmap=CMAP):
    lc = None
    for dist_seg, freq_seg, metric_seg in zip(bs.distances, bs.frequencies, metric_segs, strict=True):
        for b in range(freq_seg.shape[1]):
            points = np.column_stack([dist_seg, freq_seg[:, b]])[:, None, :]
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm, zorder=2)
            lc.set_array(0.5 * (metric_seg[:-1, b] + metric_seg[1:, b]))
            lc.set_linewidth(1.2)
            ax.add_collection(lc)
    return lc


def plot_metric(bs, metric_segs, cbar_label, hsp_x, freq_max, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    lc = _colored_bands(ax, bs, metric_segs, norm=Normalize(vmin=0, vmax=1))
    fig.colorbar(lc, ax=ax, label=cbar_label)
    _decorate_ax(ax, hsp_x, freq_max)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"wrote {path}")


def main(output: Path):
    output.mkdir(parents=True, exist_ok=True)

    ph = load_phonopy(DATA / "phonopy.yaml")
    ph.force_constants = read_force_constants_hdf5(DATA / "force_constants.h5")

    kpts, connections = get_band_qpoints_and_path_connections(KPATH, npoints=101)
    ph.run_band_structure(kpts, path_connections=connections, with_eigenvectors=True)
    bs = ph.band_structure

    apr = compute_APR_from_phonopy(ph)
    lgt = compute_L_from_phonopy(ph)
    vp2 = compute_V_from_phonopy(ph)

    hsp_x = [d[0] for d in bs.distances] + [bs.distances[-1][-1]]
    freq_max = 1.05 * max(f.max() for f in bs.frequencies)

    plot_metric(bs, apr, "APR", hsp_x, freq_max, output / "graphene_metrics_apr.png")
    plot_metric(bs, lgt, "Longitudinality $L$", hsp_x, freq_max, output / "graphene_metrics_L.png")
    plot_metric(bs, vp2, "Verticality $V^{p=2}$", hsp_x, freq_max, output / "graphene_metrics_Vp2.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT, help="output directory for figures")
    args = parser.parse_args()
    main(args.output)
