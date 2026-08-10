# unPHold

**unPHold** unfolds phonon band structures from a supercell calculation onto a primitive-cell Brillouin zone, and characterizes each mode.

GitHub repo: [sabia-group/unPHold](https://github.com/sabia-group/unPHold)

## What unPHold does

A supercell has a smaller Brillouin zone than the primitive cell it is built from, so its phonon branches appear as folded copies of the primitive-cell dispersion.
Unfolding reverses this down-folding: from a supercell phonon calculation it recovers the effective primitive-cell dispersion.

<figure markdown>
  ![Folded TBG bands](assets/tbg_tbg_uc_bands.png){ width=320 }
  ![Unfolded TBG spectrum](assets/tbg_tbg_unfolded_vs_blg.png){ width=320 }
  <figcaption>
    <strong>Unfolding twisted bilayer graphene phonon bands.</strong>
    <strong>Left:</strong> the raw phonon bands of the moiré supercell, where every mode is folded into the small moiré Brillouin zone, giving dense, hard-to-read bands.
    <strong>Right:</strong> after unfolding onto one layer's primitive cell (blue), the effective dispersion is recovered and can be compared against a Bernal bilayer reference (red).
  </figcaption>
</figure>

Beyond the unfolding supercell phonon bands, unPHold also characterizes individual modes and visualizes their real-space displacement pattern.

<figure markdown>
  ![Layer-breathing mode of twisted bilayer graphene](assets/tbg_m6r1_viz_lbm_24.png){ width=700 }
  <figcaption>
    <strong>A special layer-breathing mode (LBM) of twisted bilayer graphene at &Gamma;</strong>.
    The bottom layer (left) and the top layer (right) move antiphase out-of-plane (colour, red for +z and blue for -z), with moire-modified amplitude at AA and AB stacking regions.
  </figcaption>
</figure>

## Scope

- 3D and 2D supercells, moiré superlattices, defected structures
- Relaxed / deregistered structures
- Phonon mode characterization

## Features

- Atom matching for defective, de-registered, and layer-resolved systems.
- Per-mode unfolding weights and unfolded band structure visualization.
- Mode-character metrics: acoustic participation ratio (APR), longitudinality (L), and out-of-plane verticality (V).
- Built on [phonopy](https://phonopy.github.io/phonopy/): unPHold reuses phonopy's force constants and structural metadata, so it works with any force-constant source, including DFT codes and machine-learning interatomic potentials (MLIPs).

## Getting started

- [Installation](installation.md)
- Theory: [unfolding](theory/unfolding.md), [mode-character metrics](theory/metrics.md)
- Tutorials: [Si bulk](tutorials/si.md), [twisted bilayer graphene](tutorials/twisted_bilayer.md),
  and [defect graphene monolayer](tutorials/defect.md)
