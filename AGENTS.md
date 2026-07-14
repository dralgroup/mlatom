# AGENTS.md

Guidance for AI coding agents (and humans) working with MLatom.

## What MLatom is

MLatom is an open-source package for **atomistic simulations with machine learning
and quantum-chemical methods** (DFT, wavefunction-based, and semi-empirical). It is
used as a Python library (`import mlatom as ml`), through input files, or from the
command line (`mlatom input.inp`). Full manuals and tutorials: <http://mlatom.com/docs>.

## This repository

This is the public MLatom package: the `mlatom/` Python package with its prebuilt
backends (MLatomF, model weights) plus the docs and citation pointers. There is
nothing to compile — to use MLatom, install it from PyPI (below) so you also get its
Python dependencies and the `mlatom` command.

## Install

```bash
python3 -m pip install -U mlatom
```

- Runs on **Linux**; requires **Python ≥ 3.9**.
- Optional third-party interfaces (Gaussian, ORCA, PySCF, MNDO, …) are set up
  separately — see the [installation guide](http://mlatom.com/docs/installation.html).
- **AIQM3** and other advanced Aitomistic methods are an add-on:
  `pip install aitomic-addons` (Linux, Python **3.9–3.11**). See the
  [add-ons guide](http://mlatom.com/docs/addons.html).

No installation needed for a quick try: MLatom runs online, registration-free, on the
[Aitomistic Hub](https://aitomistic.xyz).

## Run

```python
import mlatom as ml

mol = ml.data.molecule.from_xyz_file('init.xyz')
model = ml.models.methods(method='ANI-1ccx')      # a pre-trained universal ML model
ml.optimize_geometry(model=model, initial_molecule=mol)
```

More single-point, optimization, dynamics, and spectra examples are in the README
"Quick start" and throughout the [documentation](http://mlatom.com/docs).

## Task workflows — skills

Ready-made, tested recipes for common tasks — installing MLatom, running specific
methods, reproducing published workflows — live in a companion repository:

**<https://github.com/dralgroup/mlatom-skills>**

If a user is setting MLatom up or asks how to do a specific task, check there first
and follow the relevant `SKILL.md` (start with its install skill for setup). New
skills are welcome — contributions are open.

## When helping a user

- Prefer the documented interfaces (Python library, input files, CLI) over relying on
  internals.
- Check a method's requirements before installing — some need a specific Python
  version, OS, or add-on package (see the docs links above).
- To change MLatom itself, see [CONTRIBUTING.md](CONTRIBUTING.md) and the
  [Code of Conduct](CODE_OF_CONDUCT.md).

## Cite

If work uses MLatom, cite it — use **"Cite this repository"** (generated from
[`CITATION.cff`](CITATION.cff)); the full list with BibTeX is on the
[License and citations](http://mlatom.com/docs/license.html) page.
