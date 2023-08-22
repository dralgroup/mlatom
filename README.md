# MLatom with GICnet model

## GICnet model

The GICnet model is a realization of the concept of 4D-spacetime atomistic artificial intelligence models for molecules. The concept and models are described in 

* Fuchun Ge, Lina Zhang, Arif Ullah, Pavlo O. Dral*. [Four-dimensional spacetime atomistic artificial intelligence models](https://doi.org/10.1021/acs.jpclett.3c01592). J. Phys. Chem. Lett. 2023, in press. DOI: 10.1021/acs.jpclett.3c01592.
See also [preprint on ChemRxiv](https://doi.org/10.26434/chemrxiv-2022-qf75v).

The model is implemented by Fuchun Ge of [Pavlo O. Dral's group](http://dr-dral.com) in the development version of [MLatom](https://github.com/dralgroup/mlatom). This 4D-spacetime GICnet model implementation was used for obtaining the results reported in the above manuscript. The code snapshot of this implementation is available here.

**Important note:** This branch will not be further updated, please check and use the future official releases of [MLatom](https://github.com/dralgroup/mlatom) with the latest implementations, manuals, and tutorials of the GICnet model. Future implementations may contain many changes.

## Pre-trained models
Pre-trained GICnet models for all molecules are available in [figshare](https://doi.org/10.6084/m9.figshare.22723414).

## About MLatom
MLatom is a package for atomistic simulations with machine learning. See the official website [MLatom.com](http://mlatom.com) for more information.

It is open-source software free for noncommercial uses such as academic research and education. MLatom is a part of [XACS](http://XACScloud.com/) (Xiamen Atomistic Computing Suite) since 2022 and you can use MLatom @ XACS cloud computing service for using the package online via a web browser.

The code and binaries currently can be obtained via pip:

`python3 -m pip install -U MLatom`

# How to use this code
The main program is located in the `MLatom_GICnet/MLatom.py`.

To run it, please use Python which satisfies the basic requirements that can be found at http://mlatom.com/download/.
Tensorflow is also required to run the GICnet implementation. We recommend version 2.4, which is tested by us. Later versions might also work but with tons of warnings from Tensorflow...
In short, before running the above main program, it is recommended to install all dependencies for Python as:
    `$ python -m pip install numpy scipy tensoflow h5pu pyh5md rmsd tqdm`

After the zip file of the pre-trained models is unzipped, you can find a folder called Models. We provide the trained GICnets for tc=10 and 20, under subfolders named with corresponding molecules.

A sample input to use the GICnet model to propagate an MD trajectory is provided for ethanol, in `Models/ethanol/4DMD/MD.inp`. Switch to that folder and run `./runMD.sh` will do the job. The trajectory information will be stored in files whose names start with traj, e.g. `traj.xyz` for geometries, `traj.vxyz` for velocities.

For training a GICnet, another sample input for ethanol again is provided in `Models/ethanol/tc10/train.inp`. Note that all training trajectories in H5MD format should be listed in the file named `trajList`. We provide an example trajectory also in that folder, named `traj.h5`.

For the generation of the power spectrum, please use the command below:
    `$ python MLatom_GICnet/MLatom.py IRSS output=ps trajvxyzin=[path_to_your_traj.vxyz] dt=[correct_time_step]`
Then an image named `ps.png` will be generated.

