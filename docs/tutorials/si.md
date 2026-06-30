# Si: 3D Bulk Unfolding

In this section, we demonstrate how to unfold silicon phonon bands from a 2x2x2 supercell back to the primitive cell.
The full runnable script is embedded in the [Complete example](#complete-example) section below, and also `examples/si_bulk_unfolding.py`.

<figure markdown>
  ![Si FCC unit cell](https://vasp.at/tutorials/latest/bulk/part1/e01_fcc-Si/fcc-unit-cell.png){ width=300 }
  <figcaption>Si unit cell (left) and primitive cell (right). Image: <a href="https://vasp.at/tutorials/latest/bulk/part1/">VASP tutorials</a>.</figcaption>
</figure>

---

## Preparing the phonopy inputs

unPHold only needs `Phonopy` object of the supercell (usually reload from a `phonopy.yaml` file) with its force constants as input.
One can refer to [this tutorial](https://how-tos.readthedocs.io/en/latest/phonopy_simple/phonopy_in_python.html) for a complete phonopy workflow in Python.

We have prepared the 2x2x2 supercell data (`tests/data/si/uc_2_sc_1_aims`) for unfolding, as well as the primitive cell data (`tests/data/si/uc_1_sc_2_aims`) for reference.
Since the respective supercells for finite difference are the same, the the unfolded supercell phonon bands should match the primitive cell bands exactly.

We can load above data and plot the phonon bands for both the primitive cell and the supercell:

<figure markdown>
  ![Si UC bands](../assets/si_uc_bands.png){ width=300 }
  ![Si SC bands](../assets/si_sc_bands.png){ width=300 }
  <figcaption>Left: Si primitive-cell phonon bands, calculated with 2x2x2 supercell. Right: Si 2x2x2 supercell bands plotted in the primitive-cell BZ.</figcaption>
</figure>

### K-point path

To unfold the supercell bands to desired primitive cell k-path, we shall first define the k-path in the primitive-cell BZ. The standard high-symmetry points for FCC are:

<figure markdown>
  ![FCC BZ](https://fhi-aims-club.gitlab.io/tutorials/phonons-with-fhi-vibes/figures/BZ_fcc.png){ width=300 }
  <figcaption>FCC Brillouin zone with standard high-symmetry points. Image: <a href="https://fhi-aims-club.gitlab.io/tutorials/phonons-with-fhi-vibes/phonons/2_phonopy_basics/exercise-2/">FHI-vibes tutorial</a>.</figcaption>
</figure>

We shall use the k-path Γ-X-U|K-Γ-L in primitive cell BZ.
The code for generating the kpath in UC BZ fractional coordinate (used in phonopy) and transforming to SC BZ fractional coordinate is as follows:
```python
kpts_uc, connections = get_band_qpoints_and_path_connections(KPATH, npoints=51)
kpts_flat, bz_idx = concatenate_bands(kpts_uc, connections)
kpts_sc = [k @ TMAT.T for k in kpts_uc]
```

`concatenate_bands` removes the duplicate k-point at each segment boundary
(the shared Γ between X-Γ and Γ-L), so `kpts_flat` has no repeated points.
`bz_idx` records where each high-symmetry point falls in the flat array, used later
for axis tick marks.

The k-path lives in the **primitive-cell BZ** (FCC):


`kpts_sc` transforms the same physical k-points from primitive-cell fractional
coordinates to supercell fractional coordinates via
$\mathbf{k}_{SC} = \mathbf{k}_{UC} \cdot M^T$,
where $M$ is `TMAT = diag([2, 2, 2])`, encoding
$\mathbf{a}_{SC} = M\,\mathbf{a}_{UC}$.
[`Unfold.set_kpts_in_unitcell()`][unphold.unfold.Unfold.set_kpts_in_unitcell]
handles this conversion internally when you pass the flat k-path.

### Running the unfolding

```python
    unfold = Unfold(
        unitcell=atoms_ph2ase(ph_uc.unitcell),
        supercell=atoms_ph2ase(ph_sc.unitcell),
        transformation_matrix=TMAT,
        verbose=True,
    )
unfold.set_kpts_in_unitcell(kpts_flat, format="fractional")
unfold.calculate_sc_phonon(dyn_sc=ph_sc.dynamical_matrix, factor="thz")
unfold.calculate_weights()
```

[`calculate_sc_phonon`][unphold.unfold.Unfold.calculate_sc_phonon] diagonalises
the supercell dynamical matrix at each k-point (~10 s for 201 k-points on a laptop).

[`calculate_weights`][unphold.unfold.Unfold.calculate_weights] projects each SC
eigenvector onto the primitive-cell plane waves, yielding
`unfold.weights` of shape `(nkpts, n_sc_modes)`. As a sanity check, the weight sum
per k-point should equal $3 N_{UC} = 6$ for the 2-atom primitive cell.

### Interpreting the result

The unfolded spectral function recovers the primitive-cell dispersion from the 16-atom supercell:

<figure markdown>
  ![Unfolded spectral function](../assets/si_unfolded_vs_sc.png){ width=420 }
  <figcaption>Unfolded spectral function. The 48 SC modes collapse onto the 6 primitive-cell
  branches, recovering the well-known Si phonon dispersion.</figcaption>
</figure>

To validate, overlay the directly computed primitive-cell bands (red):

<figure markdown>
  ![Unfolded vs UC bands](../assets/si_unfolded_vs_uc.png){ width=420 }
  <figcaption>Unfolded spectral function with primitive-cell bands overlaid in red.
  For a perfect supercell the red lines sit exactly on the spectral-weight maxima.</figcaption>
</figure>

Any deviation between the red lines and the spectral weight would indicate a
mismatch — e.g. a wrong transformation matrix or inconsistent DFT settings between
the two runs.

---

## Notes for large calculations

[`calculate_sc_phonon`][unphold.unfold.Unfold.calculate_sc_phonon] supports saving
intermediate results to disk via the `save_fpath` argument, which allows restarting
without rerunning the expensive diagonalisation step. Contact us or open an issue
if you need this for your system.

---

## Complete example

The full python script is available at [`examples/si_bulk_unfolding.py`](https://github.com/sabia-group/unPHold/blob/main/examples/si_bulk_unfolding.py).

