![MLatom-banner](https://github.com/user-attachments/assets/9433ac9a-bcc9-4dd4-9128-d834c85603b0)

[![Downloads](https://static.pepy.tech/badge/mlatom)
](https://pepy.tech/project/mlatom)  [![Downloads](https://static.pepy.tech/badge/mlatom/month)](https://pepy.tech/project/mlatom)  [![Downloads](https://static.pepy.tech/badge/mlatom/week)](https://pepy.tech/project/mlatom)

# MLatom

MLatom is a package for atomistic simulations with machine learning and quantum chemical methods such as DFT, wavefunction based, and semi-empirical approximations. You can use it as a Python library, via input files or command line. It is an open-source software under the MIT license (modified to request proper citations).

Official website: [MLatom.com](http://mlatom.com) with manuals and tutorials.

MLatom is supported by [Aitomistic](https://aitomistic.com), which enables simulations with MLatom online on the [Aitomistic Hub](https://aitomistic.xyz) (registration free), with [AI assistant Aitomia](http://mlatom.com/aitomia/) helping with autonomous atomistic simulations. Aitomistic also distributes [add-ons](https://aitomistic.com/en/product/aitomic) to MLatom, which supercharge MLatom with the most advanced features and ML models, often before they are published.

The package was founded by `Pavlo O. Dral <http://dr-dral.com>`__ on 10 September 2013. Pavlo continues to develop and supervise the development of the package.

## Local installation

Quick installation:

`python3 -m pip install -U mlatom`

See detailed [installation instructions](http://mlatom.com/docs/installation.html) on recommended dependences and other tips.

# Brief overview of features

<p align="center"><img src="http://mlatom.com/docs/_images/image1.png"/></p>

# Releases
Full [release notes](http://mlatom.com/docs/releases.html).

- MLatom 3.18.0 - FSSH, KRR in Julia, MDtrajNet-1 - universal model for directly predicting MD trajectories, ECTS - a diffusion model for generating TSs.
- MLatom 3.17.1-3.17.2 (26.03-21.05.2025) - + QST2 and QST3, NEB, more options for RMSD calculations and UV/vis plots, code refactoring for higher efficiency, major bug fixes, etc.
- MLatom 3.16.2 (18.12.2024) - bug fixes, quality of life improvements (calculation of bond lengths, angles, RMSD, geometry optimization followed by frequency calculations in the same file, etc.).
- MLatom 3.16.1 (11.12.2024) - DFTB and TD-DFTB via interface to DFTB+.
- MLatom 3.16.0 (04.12.2024) - TDDFT and TDA calculations via PySCF interface, TDDFT via Gaussian, and parsing of Gaussian output files into MLatom data format ([overview](https://xacs.xmu.edu.cn/docs/mlatom/releases.html#mlatom-3-16-0))
- MLatom 3.15.0 (27.11.2024) - [fine-tuning](https://xacs.xmu.edu.cn/docs/mlatom/tutorial_tl.html#transfer-learning-from-the-universal-models) of the foundational ANI potentials ANI-1x, ANI-1ccx, ANI-1ccx-gelu, and ANI-2x.
- MLatom 3.14.0 (20.11.2024) - [UV/vis spectra](https://xacs.xmu.edu.cn/docs/mlatom/tutorial_uvvis.html) from single-point convolution and nuclear-ensemble approach. Updated interface to MACE to support its latest 0.3.8 version.
- MLatom 3.13.0 (06.11.2024) - [IR spectra](https://xacs.xmu.edu.cn/docs/mlatom/tutorial_ir.html) calculations with AIQM1, AIQM2, UAIQM with semi-empirical baseline, and a range of QM methods (DFT, semi-empirical, ab initio wavefunction), with empirical scaling for better accuracy, special spectra module with plotting routines in Python. 
- MLatom 3.12.0 (08.10.2024) - [AIQM2](https://doi.org/10.26434/chemrxiv-2024-j8pxp), [ANI-1ccx-gelu](https://doi.org/10.26434/chemrxiv-2024-c8s16).
- MLatom 3.11.0 (23.09.2024) - DENS24 functionals, simpler choice of methods, IR spectra, important bug fixes (particularly for active learning) ([overview](https://xacs.xmu.edu.cn/docs/mlatom/releases.html#mlatom-3-11-0)).
- MLatom 3.10.0-1 (21-22.08.2024) - active learning for surface hopping MD, multi-state ANI for excited states, gapMD for efficient exploration of the conical intersection regions, quality of life improvements such as viewing molecules, databases, and trajectories in Jupyter, easier load of molecules ([overview](https://xacs.xmu.edu.cn/docs/mlatom/releases.html#mlatom-3-10-0)).
- [A-MLatom/MLatom\@XACS](https://xacs.xmu.edu.cn/docs/mlatom/a-mlatom-xacs.html) update (24.07.2024) - [Raman spectra](https://xacs.xmu.edu.cn/docs/mlatom/tutorial_raman.html)
- MLatom 3.9.0 (23.07.2024) - [periodic boundary conditions](https://xacs.xmu.edu.cn/docs/mlatom/releases.html#mlatom-3-9-0)
- MLatom 3.8.0 (17.07.2024) - [directly learning dynamics](https://xacs.xmu.edu.cn/docs/mlatom/releases.html#mlatom-3-8-0)
- MLatom 3.7.0-1 (03-04.07.2024) - [active learning & batch parallelization of MD](https://xacs.xmu.edu.cn/docs/mlatom/releases.html#mlatom-3-7-0-3-7-1)
- [A-MLatom/MLatom\@XACS](https://xacs.xmu.edu.cn/docs/mlatom/a-mlatom-xacs.html) update (27.06.2024) - [universal and updatable AI-enhanced QM methods (UAIQM)](https://xacs.xmu.edu.cn/docs/mlatom/tutorial_uaiqm.html)
- [A-MLatom/MLatom\@XACS](https://xacs.xmu.edu.cn/docs/mlatom/a-mlatom-xacs.html) update (20.06.2024) - [IR spectra](https://xacs.xmu.edu.cn/docs/mlatom/tutorial_ir.html)
- MLatom 3.6.0 (15.05.2024) - [+ new universal ML models (ANI-1xnr, AIMnet2, DM21)](https://xacs.xmu.edu.cn/docs/mlatom/releases.html#mlatom-3-6-0)
- MLatom 3.5.0 (08.05.2024) - [quasi-classical trajectory/molecular dynamics](https://xacs.xmu.edu.cn/docs/mlatom/releases.html#mlatom-3-5-0)
- MLatom 3.4.0 (29.04.2024) - [usability improvements with focus on geometry optimizations](https://xacs.xmu.edu.cn/docs/mlatom/releases.html#mlatom-3-4-0)
- MLatom 3.3.0 (03.04.2024) - [surface-hopping dynamics](https://mlatom.com/docs/releases.html#mlatom-3-3-0)
- MLatom 3.2.0 (19.03.2024) - [diffusion Monte Carlo and energy-weighted training](http://mlatom.com/docs/releases.html#mlatom-3-2-0)

# Contributions and derivatives

We highly welcome the contributions to the MLatom project. You may also create your own private derivatives of the project by following the license requirements.

If you want to contribute to the main MLatom repository, the easiest way is to create a fork and then send a pull request. Alternatively, you can ask us to create a branch for you. After we receive a pull request, we will review the submitted modifications to the code and may clean up of the code and do other changes to it and eventually include your modifications in the main repository and the official release.
- MLatom 3.1.0 (12.29.2023) - [MACE interface](http://mlatom.com/releases/#Version_31)
- MLatom 3.0.0 (12.09.2023)
