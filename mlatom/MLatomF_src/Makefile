# Makefile for MLatomF

# Select compiler, libraries and compiling options
include Makefile.intel

# Suffixes
.SUFFIXES:
.SUFFIXES: .f90 .o

.f90.o:
	$(FC) -c $(FFLAGS) $<

# All objects necessary to compile MLatomF
OBJ = MLatomF.o MLatomFInfo.o MLmodel.o constants.o molDescr.o \
	 D_CoulombMatrix.o D_rel2eq.o D_ID.o dataset.o mathUtils.o A_KRR.o A_KRR_kernel.o \
	 optionsModule.o precision.o sampling.o statistics.o stopper.o strings.o \
	 types.o timing.o analyze.o
	
all: ../MLatomF ../cs.so

# Compile it into MLatomF (or MLatomF.exe on cygwin) in bin folder
../MLatomF: $(OBJ)
	$(FC) -o $@ $(FFLAGS) $(OBJ) $(FLIBS)
	
# ML_NEA cross section shared object
../cs.so : cs.cpp
	$(CC) -fPIC cs.cpp $(CFLAGS) $(CLIBS) -o $@

# Dependencies
A_KRR.o : A_KRR.f90 A_KRR_kernel.o dataset.o mathUtils.o molDescr.o \
	optionsModule.o precision.o statistics.o stopper.o timing.o  
A_KRR_kernel.o : A_KRR_kernel.f90 constants.o dataset.o mathUtils.o molDescr.o optionsModule.o precision.o \
	 stopper.o strings.o timing.o 
analyze.o : analyze.f90 dataset.o optionsModule.o statistics.o stopper.o
constants.o : constants.f90 precision.o
D_CoulombMatrix.o : D_CoulombMatrix.f90 constants.o dataset.o mathUtils.o optionsModule.o precision.o \
	stopper.o
D_rel2eq.o : D_rel2eq.f90 dataset.o mathUtils.o optionsModule.o precision.o stopper.o
D_ID.o : D_ID.f90 mathUtils.o optionsModule.o precision.o stopper.o
dataset.o : dataset.f90 mathUtils.o optionsModule.o precision.o stopper.o strings.o timing.o types.o
mathUtils.o : mathUtils.f90 optionsModule.o precision.o stopper.o timing.o types.o
MLatomF.o : MLatomF.f90 A_KRR.o dataset.o MLatomFInfo.o MLmodel.o optionsModule.o sampling.o timing.o \
analyze.o 
MLatomFInfo.o : MLatomFInfo.f90 
MLmodel.o : MLmodel.f90 A_KRR.o dataset.o molDescr.o \
	optionsModule.o precision.o sampling.o statistics.o strings.o
molDescr.o : molDescr.f90 dataset.o D_CoulombMatrix.o D_rel2eq.o D_ID.o mathUtils.o optionsModule.o \
	 precision.o stopper.o timing.o types.o  
optionsModule.o : optionsModule.f90 MLatomFInfo.o precision.o stopper.o strings.o
precision.o : precision.f90
sampling.o : sampling.f90 dataset.o optionsModule.o precision.o stopper.o 
statistics.o : statistics.f90 precision.o timing.o stopper.o 
stopper.o : stopper.f90
strings.o : strings.f90 stopper.o
timing.o : timing.f90 precision.o
types.o : types.f90 precision.o

# Clean-up src folder:
clean:
	rm -f *.exe *.o *.mod ../MLatomF ../cs.so ../*.pyc
