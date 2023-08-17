#!/bin/bash
#
# Script that runs test jobs to check whether everything is 
# ok with a compilation of MLatom
#

mlatom='../MLatom.py' # You may need to adapt it to the path with MLatom.py (if you install from zipped file) or mlatom (if you install via pip)

rm -f corr_ML_451.dat E_D-ML_451.dat E_FCI_451-E_UHF_451.dat
$mlatom mlatom.inp > mlatom.out

diff mlatom.saved mlatom.out | grep MAE
