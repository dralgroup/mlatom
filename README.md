# MLatom with MD code and scripts for "Energy-conserving molecular dynamics is not energy conserving"

## The algorithms

The algorithms are described in

* Lina Zhang†, Yi-Fan Hou†, Fuchun Ge, Pavlo O. Dral*. [Energy-conserving molecular dynamics is not energy conserving](https://doi.org/10.1039/D3CP03515H). *Phys. Chem. Chem. Phys.* **2023**, *Just Accepted*. DOI: 10.1039/D3CP03515H.

The MD was implemented by Yi-Fan Hou of [Pavlo O. Dral's group](http://dr-dral.com) in the development version of [MLatom](https://github.com/dralgroup/mlatom). This code snapshot can be used for obtaining the results reported in the above manuscript.

**Important note:** This branch will not be further updated, please check and use the future official releases of [MLatom](https://github.com/dralgroup/mlatom) with the latest implementations, manuals, and tutorials of the MD code. Future implementations may contain many changes.

## Pre-trained models
MD data reported in the paper are available at [figshare](https://doi.org/10.6084/m9.figshare.22315147).

## About MLatom
MLatom is a package for atomistic simulations with machine learning. See the official website [MLatom.com](http://mlatom.com) for more information.

It is open-source software free for noncommercial uses such as academic research and education. MLatom is a part of [XACS](http://XACScloud.com/) (Xiamen Atomistic Computing Suite) since 2022 and you can use MLatom @ XACS cloud computing service for using the package online via a web browser.

The code and binaries currently can be obtained via pip:

`python3 -m pip install -U MLatom`

# How to use this code
The main program is `MLatom.py`.

To run it, please use Python which satisfies the basic requirements that can be found at http://mlatom.com/download/.

