# MLatom with MD code and scripts for "Energy-conserving molecular dynamics is not energy conserving"

## The algorithms

The algorithms are described in

* Lina Zhang†, Yi-Fan Hou†, Fuchun Ge, Pavlo O. Dral*. [Energy-conserving molecular dynamics is not energy conserving](https://doi.org/10.1039/D3CP03515H). *Phys. Chem. Chem. Phys.* **2023**, *Just Accepted*. DOI: 10.1039/D3CP03515H.

The MD was implemented by Yi-Fan Hou of [Pavlo O. Dral's group](http://dr-dral.com) in the development version of [MLatom](https://github.com/dralgroup/mlatom). This code snapshot can be used for obtaining the results reported in the above manuscript.

**Important note:** This branch will not be further updated, please check and use the future official releases of [MLatom](https://github.com/dralgroup/mlatom) with the latest implementations, manuals, and tutorials of the MD code. Future implementations may contain many changes.

## Data availability
MD data reported in the paper are available at [figshare](https://doi.org/10.6084/m9.figshare.22315147).

## About MLatom
MLatom is a package for atomistic simulations with machine learning. See the official website [MLatom.com](http://mlatom.com) for more information.

It is open-source software free for noncommercial uses such as academic research and education. MLatom is a part of [XACS](http://XACScloud.com/) (Xiamen Atomistic Computing Suite) since 2022 and you can use MLatom @ XACS cloud computing service for using the package online via a web browser.

The code and binaries currently can be obtained via pip:

`python3 -m pip install -U MLatom`

# How to use this code
The main program is `MLatom.py`.

To run it, please use Python which satisfies the basic requirements that can be found at http://mlatom.com/download/.

## N2O dynamics
After unzipping N2O_dynamics.zip, you will find several folders in it.

The equlibration of N2O was done in folder 'N2O_equilibration', where Python script 'N2O_equilibrate.py' was used to perform MD with NVT ensemble.

Then MD with NVE ensemble was performed using AIQM1, PBE/def2-SVP and GFN2-xTB in each folder named after the method. Then an input file 'IRSS.inp' was used to generate IR spectrum with MLatom ('$mlatom IRSS.inp > IRSS.out'. $mlatom is the path to bin/mlatom. Make sure that path to each program is specified in bin/mlatom). Trajectory with XYZ coordinates ('N2O_nve.xyz') was copied to the folder 'TBE', and TBE energies were calculated in this folder with python scripts 'generate_ccxd_inp_from_xyz.py' and 'calc_energy.py'. TBE energies were saved in, for example, 'aiqm1_tbe_energy.dat'.

Figures related to N2O dynamics were plotted in folder 'figure'. Experimental result ('10024-97-2-IR.jdx') was downloaded from NIST. The python script 'jdx2npy.py' was used to convert the spectrum into npy format. The python script 'N2O_NVE.py' was used to plot all the IR spectra, 'compare_tbe.py' was used to compare TBE total energies and 'etot.py' was used to plot the simulation total energies.
