import numpy as np 
import os

# module load gcc
# module load intel/19.1.114

os.system('rm -r KREG.cpython-39-x86_64-linux-gnu.so *.mod temp')
os.system("f2py --build-dir temp --f77exec=/share/intel/2019/compilers_and_libraries_2019.1.144/linux/bin/intel64/ifort --f90exec=/share/intel/2019/compilers_and_libraries_2019.1.144/linux/bin/intel64/ifort --f90flags='-O3 -mkl=parallel -qopenmp -static-intel' -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -c -m KREG stopper.f90 mathUtils.f90 KREG.f90")
os.system(f'/share/intel/2019/compilers_and_libraries_2019.1.144/linux/bin/intel64/ifort -o ./KREG.cpython-39-x86_64-linux-gnu.so -O3 -mkl=parallel -qopenmp -shared -shared -nofor_main temp/temp/src.linux-x86_64-3.9/KREGmodule.o temp/temp/src.linux-x86_64-3.9/temp/src.linux-x86_64-3.9/fortranobject.o temp/stopper.o temp/mathUtils.o temp/KREG.o temp/temp/src.linux-x86_64-3.9/KREG-f2pywrappers2.o -static-intel  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core  -liomp5 -lpthread -lm -ldl')
os.system('mv KREG.cpython-39-x86_64-linux-gnu.so KREG.so')