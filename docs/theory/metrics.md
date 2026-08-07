# Metrics for mode characterization

Beyond the unfolding weight, unPHold provides multiple scalar metrics that characterize the eigenvector of each phonon mode.
They are computed from the supercell eigenvectors and are used.
A hands-on tutorial computing and plotting all three metrics on monolayer graphene is available at [Graphene phonon mode characterization](../tutorials/metrics.md).

As a reminder for notations: \(e^\kappa_{\mathbf{q},n}\) is the norm\(=1\) eigenvector component of phonon mode \(n\) at wavevector \(\mathbf{q}\) on atom \(\kappa\), and \(N\) is the number of atoms.

## Acoustic participation ratio (APR)

The acoustic participation ratio[^kamencek] measures how in-phase the atomic displacements are, distinguishing acoustic-like from optic-like modes:

\[
\mathrm{APR}_{\mathbf{q},n} =
\frac{2}{N(N+1)}
\frac{
\left| \sum_{\kappa,\kappa'} \dfrac{(e^\kappa_{\mathbf{q},n})^\dagger e^{\kappa'}_{\mathbf{q},n}}{\sqrt{m_\kappa m_{\kappa'}}} \right|^2
}{
\sum_{\kappa,\kappa'} \left| \dfrac{(e^\kappa_{\mathbf{q},n})^\dagger e^{\kappa'}_{\mathbf{q},n}}{\sqrt{m_\kappa m_{\kappa'}}} \right|^2
}
\]

where \(m_\kappa\) are the atomic masses and the sum runs over unique atom pairs.
\(\mathrm{APR}=1\) indicates an acoustic-like mode (atoms moving in phase), while \(\mathrm{APR}\to 0\) indicates an optic-like mode.
Computed by `unphold.metrics.compute_APR`.

## Longitudinality (L)

Longitudinality[^legenstein] measures how much a mode's displacement aligns with its propagation direction
\(\hat{\mathbf{q}}\):

\[
L_{\mathbf{q},n} =
\frac{1}{N} \left| \sum_{\kappa=1}^{N} \hat{\mathbf{q}} \cdot e^\kappa_{\mathbf{q},n} \right|
\]

\(L=1\) marks a purely longitudinal mode and \(L=0\) a purely transverse one.
Computed by `unphold.metrics.compute_L`.

## Verticality (V)

Verticality measures how much of a mode's motion is out of the xy-plane, along \(\hat{\mathbf{z}}\).
The preferred definition is the \(p=2\) form,

\[
V^{p=2}_{\mathbf{q},n} =
\sum_\kappa |\hat{\mathbf{z}} \cdot e^\kappa_{\mathbf{q},n}|^2,
\]

with \(V=1\) purely out-of-plane, \(V=0\) purely in-plane, and \(V=0.5\) as the threshold (for example a single atom vibrating 45 degrees from the z-axis).
The complement \((1-V)\) measures the in-plane character of the mode.
Computed by `unphold.metrics.compute_V_p2`.

[^kamencek]: T. Kamencek, *Understanding Phonon-Related Properties in Metal-Organic Frameworks for Controlling Their Mechanical and Thermal Characteristics*, PhD thesis, Technische Universität Graz (2022).
[^legenstein]: L. Legenstein, L. Reicht, T. Kamencek, and E. Zojer, *Anisotropic Phonon Bands in H-Bonded Molecular Crystals: The Instructive Case of α-Quinacridone*, ACS Mater. Au **3**, 371 (2023).
