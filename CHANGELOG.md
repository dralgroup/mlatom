# Changelog

All notable changes to MLatom are summarized here. For the complete, detailed
release notes see the [official release notes](http://mlatom.com/docs/releases.html).

Dates are given as DD.MM.YYYY. Versions are available on
[PyPI](https://pypi.org/project/mlatom/) and
[GitHub](https://github.com/dralgroup/mlatom).

## [3.25.1] – 14.08.2026
- Fixed: the ANI `-D4` models still selected their dispersion parameters by
  functional name, so on dftd4 4.0.0 and newer they used a different fit — 3.9
  kcal/mol on an ethanol total energy, silently. 3.25.0 fixed this for AIQM1,
  AIQM2 and OMNI-P1 but not for these. No method now depends on how dftd4
  resolves a functional name. By Pavlo O. Dral.
- Fixed: `import mlatom` required PyTorch to be installed, because `aiqm1.py`
  imported it at module level without using it. By Pavlo O. Dral.
- AIQM3's D3(BJ) term is given explicitly too. Results are unchanged. By
  Pavlo O. Dral.
- We still recommend **dftd4 3.6.0**: that is the version MLatom is tested
  against. The parameter pinning above makes the D4 *energies* identical on
  4.x, but the test suite has not been run against 4.x, so Hessians,
  thermochemistry and the rest are unverified there.

## [3.25.0] – 13.08.2026
- Fine-tuning of the universal models on your own data. ANI, AIQM1, AIQM2,
  AIQM3, UAIQM, OMNI-P1 and OMNI-P2x are fine-tuned through the same `train()`
  call, and the result saves and loads like any other model. See the
  [tutorial](http://mlatom.com/docs/tutorial_finetuning.html).
- Dispersion is now declared once and handled on both sides: the term is
  subtracted from the reference labels before training and added back at
  prediction, so a fine-tuned model keeps the long-range behaviour of the model
  it started from.
- One geometry can carry labels from several methods. A molecule records where a
  label came from (`molecule.label_source`), a property can be addressed by its
  source (`mol.get_property('wb97x.energy_gradients')`), and the database keeps
  a registry of the sources it holds (`molecular_database.label_sources`).
  Databases can be written to and read back from HDF5.
- Molecules missing the property being learned are dropped before training
  rather than entering the loss as NaN, for the training and the validation set
  alike (`molecular_database.without_missing_labels()`).
- Fixed: a `weighted_sum` model tree applied its weights to only one molecule of
  a database, so every multi-molecule weighted sum was in effect unweighted.
  This affected all DENS predictions.
- Fixed: the D4(wB97X) term of AIQM1, AIQM2, OMNI-P1 and the ANI `-D4` variants
  is now selected by its damping parameters rather than by the functional name.
  dftd4 4.0.0 renamed `wb97x` to `wb97x-2008` and gave the name `wb97x` to a
  different functional, so on dftd4 >= 4.0.0 these models silently used another
  functional's damping: AIQM2 shifted by 3.5 kcal/mol on one geometry, and the
  methane-dimer well deepened from -0.65 to -1.16 kcal/mol. Results on
  dftd4 3.x are unchanged.
- Fixed: re-downloading a model died on the files already present, leaving the
  model half-extracted.
- Fixed: `CISD` and `CCSD` were concatenated in the PySCF interface's list of
  supported methods, which removed both from dispatch.

## [3.24.0] – 03.08.2026
- MLatom now prints the references to cite for AIQM3, UAIQM, OMNI-P2x,
  ANI-1ccx-gelu, DFT ensembles, and for IR spectra with AIQM models.
- Fixed the state energies read from MNDO's `fort.15` when the requested
  gradient list does not start at the ground state. This affected the energies
  reported along a molecular dynamics trajectory, not the trajectory itself.
- Fixed the filtering of initial conditions, where the molecules kept in the
  filtered database were all the same object.
- Downloading a model no longer hangs indefinitely on an unresponsive server,
  and a download that did not succeed is recognized as such: MLatom continues
  with the next link and, if none work, reports where to get the files by hand.
- Fixed the type annotation of `molecule.nstates`, which used `np.int`, removed
  in NumPy 1.24.

## [3.23.5] – 27.07.2026
- Sampling now writes the training/subtraining/validation/test and
  cross-validation index files correctly to output paths that contain spaces
  (and portably across platforms), replacing shell `cp`/`mv` calls with
  `shutil`. Thanks to @rayair250-droid.

## [3.23.4] – 15.07.2026
- MLatom is now released under the Apache License 2.0.
- Improved initial-conditions sampling: unified `random` and `Maxwell-Boltzmann`
  velocity generators, with correct angular-momentum removal and linearity-aware
  degrees of freedom for linear molecules and reproducible sampling.
- Retuned and hardened the state-gap loss used in OMNI-P2x fine-tuning and
  multi-state active learning.
- `pip install mlatom` now installs all required dependencies automatically — no
  manual dependency list needed.
- Fixed a crash when a method was queried while PySCF was not installed.
- The package can now be installed directly from a source clone (`pyproject.toml`).

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
[official release notes](http://mlatom.com/docs/releases.html).

[3.23.3]: https://pypi.org/project/mlatom/3.23.3/
[3.23.2]: https://pypi.org/project/mlatom/3.23.2/
[3.23.1]: https://pypi.org/project/mlatom/3.23.1/
[3.23.0]: https://pypi.org/project/mlatom/3.23.0/
[3.22.0]: https://pypi.org/project/mlatom/3.22.0/
[3.21.0]: https://pypi.org/project/mlatom/3.21.0/
[3.20.0]: https://pypi.org/project/mlatom/3.20.0/
