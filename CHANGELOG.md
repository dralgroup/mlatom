# Changelog

All notable changes to MLatom are summarized here. For the complete, detailed
release notes see the [official release notes](http://mlatom.com/docs/releases.html).

Dates are given as DD.MM.YYYY. Versions are available on
[PyPI](https://pypi.org/project/mlatom/) and
[GitHub](https://github.com/dralgroup/mlatom).

## [3.25.4] – 10.09.2026
- Fixed: the default KREG model did not load in 3.25.3 on Linux older than about 2021 —
  RHEL/CentOS 7 and 8, Rocky 8, Ubuntu 20.04, Debian 11. Its compiled kernel had been
  rebuilt for the Hessian fix on a newer system and needed glibc 2.34, where the 3.25.2
  kernel needed 2.14. It is rebuilt from the same sources on an older base and needs 2.14
  again; its energies, gradients and Hessians are bit-identical to 3.25.3's. By Pavlo O.
  Dral.
- Fixed: KREG Hessians were still wrong for anyone running MLatom under NumPy 2. The
  3.25.3 fix rebuilt only the kernel MLatom loads under NumPy 1; the one it loads under
  NumPy 2 predated the fix, and its Hessians were wrong - in our test by about three
  quarters of their size. It is rebuilt with the fix, gives results bit-identical to the
  NumPy 1 kernel, and also loads on older Linux. By Pavlo O. Dral.
- Fixed: nudged-elastic-band and dimer searches failed in 3.25.3 on ASE older than 3.23
  with `No module named 'ase.mep'`, because 3.25.3 imported them only from `ase.mep`,
  where ASE moved them in 3.23. MLatom now falls back to `ase.neb` and `ase.dimer`.
  (ASE 3.22's own nudged-elastic-band module also needs SciPy older than 1.14.) By
  Pavlo O. Dral.
- Fixed: `$mlatom input.inp` still failed with `Permission denied` after installing from
  source or from GitHub (`pip install .`, `pip install git+https://…`); 3.25.3 fixed it
  only for the package on PyPI. By Pavlo O. Dral.
- geomeTRIC must now be older than 1.1.1 (pip moves an installed 1.1.1 back to 1.1).
  Version 1.1.1 changed the rule that ends an intrinsic-reaction-coordinate branch, so its
  paths no longer match the ones MLatom is tested against. By Pavlo O. Dral.
- MDtrajNet needs e3nn 0.5.0, and the README and documentation now say so: with e3nn 0.5.1
  or newer, 0.6 included, the published MDtrajNet model produces a different trajectory, and
  0.4.4 does not load with PyTorch 2.6 or newer. By Pavlo O. Dral.
- Fixed: the version banner named a source commit 16 commits older than the release. By
  Pavlo O. Dral.
- Corrected four statements in the 3.25.3 notes below. They said nudged-elastic-band
  searches died on current ASE with `Atoms object has no calculator`: 3.25.2 hit that only
  when a band method other than `aseneb` was chosen, because ASE asks the end images for
  their energies with every method except `aseneb`, its default up to 3.27 (and 3.25.2's
  NEB does not load on ASE 3.27 or newer at all). They said NEB and dimer searches failed
  on any recent ASE with `No module named 'ase.neb'`: that holds from ASE 3.27 on. They
  said the move to `ase.mep` worked on old and new ASE alike: it required ASE 3.23 or
  newer. And they said a calculation could append to the trajectory of one run before it:
  only calculations running at the same time could share that state.

## [3.25.3] – 09.09.2026
- Fixed: transition-state optimizations and IRC runs that computed their own
  initial Hessian handed it to geomeTRIC in Hartree/Angstrom^2, where geomeTRIC
  expects Hartree/Bohr^2 — about 3.6 times too large. The optimizer therefore
  started from a badly scaled Hessian; paths and the steps taken to reach them
  change. By Yuxinxin Chen.
- Fixed: KREG Hessians were missing a term. The second derivative of the
  reduced-distance descriptors was not included, so every predicted Hessian was
  wrong. By Yifan Hou.
- Fixed: `density_fitting=True` with PySCF returned an energy that was not the
  energy — for methane at B3LYP/6-31G, -0.73 Hartree where the answer is -40.51,
  with nothing raised. `density_fit()` returns a new mean-field object and leaves
  the original un-run, so MLatom rebuilt the total by summing PySCF's
  `scf_summary`. That sum is the energy only while the dictionary holds exactly
  the additive components, and PySCF 2.14 also files `e2` in it — already counted
  as `coul + exc` — and `gap`, which is a HOMO-LUMO gap and not an energy. The
  energy now comes from the object that ran, and density fitting goes through the
  same convergence check as every other method. By Pavlo O. Dral.
- Fixed: a frequency calculation run through ASE (`optprog=ase`) died inside ASE
  with `Too few vibration modes (14) after selection` on any structure that has an
  imaginary frequency — every saddle point, and exactly the case MLatom prints a
  warning about one line earlier. Current ASE picks the true vibrations itself and
  insists on being handed 3N-6 of them, while MLatom had already removed the
  non-positive modes and passed only what was left. MLatom now tells ASE that these
  are the vibrations in exactly that case, and leaves ASE's own selection alone
  whenever enough modes are there — that selection is also quietly discarding the
  leftover translation and rotation modes that come out just above zero, which
  MLatom's own test keeps. No number changes: the four molecules of the AIQM1
  frequency test reproduce their saved reference to the last digit, and so do the
  thermochemistry, heat-of-formation and frequency tests that never crashed. By
  Pavlo O. Dral.
- Fixed: the test suite failed instead of skipping on a machine without Julia. The
  KRR-in-Julia test imported the PyJulia bridge at the top, and neither the bridge
  nor a Julia runtime is a dependency of MLatom — Julia is a language, not a wheel
  — so a fresh install could not run the suite clean. It also dropped the exit
  status of the run it started, as did four of the AIQM1 frequency tests, where a
  crashed calculation surfaced as an array-shape error against the reference
  instead of the traceback that caused it. The transition-state generation test
  did the same with EcTs, which is installed by hand from GitHub and is not a
  dependency either. All of these skip now, and say why. By Pavlo O. Dral.
- **The nudged elastic band now returns a band, and finds the saddle.** It builds seven
  middle images instead of three, interpolates with the image-dependent pair potential so a
  curved path does not start from a guess with atoms on top of each other, uses the improved
  tangent, and runs FIRE in two stages: relax the band, then switch on the climbing image so
  the highest image climbs onto the saddle point. What comes back is the relaxed band itself,
  reactant to product, one step per image with its energy, exposed as `geomopt.band` and
  plottable with `band.plot_energy_profile()`. The transition state is the highest image.
  Previously the return value was a log with one entry per model call, in whatever order the
  optimiser asked, and the "transition state" was whichever image it touched last. By Pavlo
  O. Dral.
- Every nudged-elastic-band image, the endpoints included, now gets a calculator: the
  improved-tangent method the new band uses asks the endpoints for their energies. By
  Pavlo O. Dral.
- Fixed: driving Gaussian for an MLatom method — geometry optimization, frequencies, IRC,
  QST2, IR — failed on a fresh install with only `Failed to open output file from external
  program` from Gaussian. MLatom writes that file through `fortranformat`, which it never
  declared as a dependency, and the import is inside the writing function, so the
  calculation ran and then died at the last step. By Pavlo O. Dral.
- Fixed: `$mlatom input.inp` did not run at all from a pip-installed MLatom. The package
  ships `shell_cmd.py` with a shebang and git marks it executable, but the build stripped
  that, so running it directly gave `Permission denied`. Every published wheel had this,
  including 3.25.2. The `mlatom` command itself was unaffected. By Pavlo O. Dral.
- Fixed: nudged-elastic-band and dimer transition-state searches failed on ASE 3.27
  and newer with `No module named 'ase.neb'`. ASE moved those into `ase.mep`, and MLatom still
  imported the old paths, so `pip install ase` — what the README tells you to do — gave a
  broken interface. The imports now use `ase.mep`, which ASE provides from 3.23 on
  (3.25.4 restores older ASE).
  By Pavlo O. Dral.
- Fixed: a MACE calculation left PyTorch in double precision for the rest of the
  session, so the next model built in the same Python session inherited it and
  crashed with `mat1 and mat2 must have the same dtype, but got Float and
  Double`. OMNI-P1 after MACE is the case that shows it. By Pavlo O. Dral.
- Fixed: three further places set PyTorch's default precision for the whole
  process and left it changed — one of them at import time, before anything was
  built. They now set it where the networks are built and put it back
  afterwards, so your own choice of precision survives. By Pavlo O. Dral.
- Fixed: the ASE calculator kept the molecule and the optimization trajectory in
  module-level variables, so two calculations running at the same time in one
  process shared them, and one could append to the other's trajectory.
  By Pavlo O. Dral.
- Dependencies now allow the versions MLatom is actually tested with: `torch` from
  2.1.2 up to but excluding 2.8, and `torchani` 2.2.x. The old `torch==2.1.2` /
  `torchani==2.2.3` were a snapshot of one environment, and nothing was tested
  against them; testing runs on torch 2.7.0 with torchani 2.2.4. These are ranges
  rather than exact pins because PyTorch publishes no build newer than 2.2.2 for
  Intel Macs, so an exact pin would make MLatom uninstallable there. torchani stays
  inside 2.2 because its model API changed in 2.8, and torch stops below 2.8 because
  2.13 removed an API our training loops use. By Pavlo O. Dral.
- AIMNet2 needs `pip install "aimnet==0.0.1"`, which the README now says. Newer
  `aimnet` releases are a rewrite that does not work with the models MLatom
  ships, and none of them could be installed next to the old `torch==2.1.2` pin
  at all. By Pavlo O. Dral.
- The READMEs now say plainly that MLatom is developed and tested on Linux and that
  this is the only platform it is verified on; macOS is often usable but untested, and
  Windows is not supported. By Pavlo O. Dral.
- Note for anyone running NumPy 2: MLatom ships a separate KREG binary for it,
  and that one has not been rebuilt since the Hessian fix above. The released
  package pins NumPy 1, where the fixed binary is the one used. By Pavlo O. Dral.

## [3.25.2] – 20.08.2026
- `AIQM3@DFT*` can now be requested: MLatom did not recognize it as a method. By Pavlo O. Dral.
- Fixed: reading Gaussian or ORCA output could fail with `No module named 'rmsd'`
  on a fresh install. `rmsd` is now installed together with MLatom. By Pavlo O. Dral.
- Fixed: UV/Vis spectra lost the peaks they had computed when IPython was not
  installed. By Pavlo O. Dral.
- Fixed: MLatomF could hang on long runs, and its error messages were not shown.
  By Pavlo O. Dral.
- A method that needs a program you do not have now names that program and gives
  the command that installs it. By Pavlo O. Dral.
- Molecular-orbital energies and occupations are available from the PySCF, xTB,
  Gaussian and ORCA interfaces. By Pavlo O. Dral.

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
