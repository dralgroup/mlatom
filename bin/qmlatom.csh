#!/bin/csh
#$ -S /bin/csh
#$ -o $JOB_ID.out
#$ -e $JOB_ID.err
#$ -m n
# 
# $1 is input file name

hostname

if (! $?COD_O_WORKDIR) then
   setenv COD_O_WORKDIR $SGE_O_WORKDIR
endif

limit

cd $COD_O_WORKDIR
#/ns80th/nas/users/dral/manuscripts/ML-spin-boson/programs/MLatom/bin/MLatom_20171130.py nthreads=8 $*
#/ns80th/nas/users/dral/manuscripts/ML-spin-boson/programs/MLatom/bin/MLatom_20171208.py nthreads=8 $*
/csnfs/users/dral/manuscripts/MLhierarchical/MLatom/bin/MLatom.py nthreads=20 $*
