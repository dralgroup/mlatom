# Changelog

All notable changes to MLatom are summarized here. For the complete, detailed
release notes see the [official release notes](https://mlatom.com/docs/releases.html).

Dates are given as DD.MM.YYYY. Versions are available on
[PyPI](https://pypi.org/project/mlatom/) and
[GitHub](https://github.com/dralgroup/mlatom).

## [3.23.3] – 27.06.2026
## [3.23.2] – 26.06.2026
## [3.23.1] – 25.06.2026
- Bug fixes and performance improvements, including faster NAMD.

## [3.23.0] – 15.06.2026
- AIQM3 is now available as a public add-on (`pip install aitomic-addons`).
- Startup banner now reports the running version together with its commit and
  build date; MLatom checks once a day whether a newer release is available.
- Direct Gaussian workflows: MLatom generates Gaussian input and runs Gaussian
  for Gaussian-internal methods combined with geometry optimizations,
  frequencies, IRC, and excited-state (TD) calculations.
- Orientation fixes for normal modes and dipole moments.
- Refined ORCA 6 interface (dispersion and ground-state energy parsing).
- Bug fixes across MNDO gradients, FSSH non-adiabatic couplings, NAMD,
  dihedral-angle handling, and active learning.

## [3.22.0] – 09.03.2026
- Released OMNI-P2x with tutorials.
- Improved performance of NAMD simulations.

## [3.21.0] – 13.02.2026
- Refactored ORCA interface supporting many more excited-state simulations
  (e.g., QD-NEVPT2).
- Quality-of-life improvements for analyzing UV/vis absorption spectra.

## [3.20.0] – 26.12.2025
- TDBA and other improvements in FSSH.

## [3.19.0 – 3.19.1] – 23.10.2025 / 14.11.2025
- See the full release notes.

## [3.18.0]
- FSSH; KRR in Julia; MDtrajNet-1 (universal model for directly predicting MD
  trajectories); ECTS (a diffusion model for generating transition states).

## Earlier releases
For 3.0.0 through 3.17.x and detailed per-version notes, see the
[official release notes](https://mlatom.com/docs/releases.html).

[3.23.3]: https://pypi.org/project/mlatom/3.23.3/
[3.23.2]: https://pypi.org/project/mlatom/3.23.2/
[3.23.1]: https://pypi.org/project/mlatom/3.23.1/
[3.23.0]: https://pypi.org/project/mlatom/3.23.0/
[3.22.0]: https://pypi.org/project/mlatom/3.22.0/
[3.21.0]: https://pypi.org/project/mlatom/3.21.0/
[3.20.0]: https://pypi.org/project/mlatom/3.20.0/
