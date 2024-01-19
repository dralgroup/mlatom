#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! sliceData: Scripts for working with sliced data                           ! 
  ! Implementations by: Pavlo O. Dral                                         ! 
  !---------------------------------------------------------------------------! 
'''

import sys, os, time, operator, re, copy
from . import stopper
from . import interface_MLatomF

class sliceDataCls(object):
    eqX  = [] # Equilibrium input vector
    xx   = [] # List of molecular descriptors
    nall = 0  # Number of molecular descriptors
    Ds   = {} # Dictionary {index i: distance to eqX}
    sorted_indices = [] # Indices of input vectors sorted by their distance to the equilibrium structure
    
    def __init__(self, argsSD = sys.argv[1:]):
        starttime = time.time()
        
        print(' %s ' % ('='*78))
        print(time.strftime(" sliceData started on %d.%m.%Y at %H:%M:%S", time.localtime()))
        for arg in argsSD:
            flagmatch = re.search('nthreads', arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if flagmatch:
                argsSD.remove(arg)
        args.parse(argsSD)
        print('        with the following options:')
        argsstr = '        '
        for arg in argsSD:
            argsstr += arg + ' '
        print(argsstr.rstrip())
        print(' %s ' % ('='*78))
        sys.stdout.flush()
        
        # Perform requested task
        if args.slicedata:
            self.getEqX(args.eqXfileIn)
            self.getX(args.XfileIn)
            self.sortByDistance()
            self.slicedata()
        if args.sampleFromSlices:
            self.sampleFromSlices()
        if args.mergeSlices:
            self.mergeSlices()
        
        endtime = time.time()
        wallclock = endtime - starttime
        print(' %s ' % ('='*78))
        print(' Wall-clock time: %.2f s (%.2f min, %.2f hours)\n' % (wallclock, wallclock / 60.0, wallclock / 3600.0))
        print(time.strftime(" sliceData terminated on %d.%m.%Y at %H:%M:%S", time.localtime()))
        print(' %s ' % ('='*78))
        sys.stdout.flush()
      
    def getEqX(self, xfilename):
        self.eqX = []
        with open(xfilename, 'r') as xf:
            for line in xf:
                self.eqX = [float(x) for x in line.split()]
                break
      
    def getX(self, xfilename):
        self.xx = []
        with open(xfilename, 'r') as xf:
            for line in xf:
                self.xx.append([float(x) for x in line.split()])
        self.nall = len(self.xx)
        
    def sortByDistance(self):
        for ii in range(1, self.nall+1):
            self.Ds[ii] = self.distance(self.eqX, self.xx[ii-1])
        self.sorted_indices = [ind for (ind, distance) in sorted(list(self.Ds.items()), key=operator.itemgetter(1))]
        
    @classmethod
    def distance(cls, xi, xj):
        # Returns distance between xi and xj descriptors (they must be of the same length)
        dd = 0.0
        for ii in range(len(xi)):
            dd += (xi[ii] - xj[ii]) ** 2
        dd = dd ** 0.5
        return dd

    def slicedata(self):
        self.print_ordered_set( nstart = 1, nend = self.nall,
                                xfilename = args.XfileIn,
                                xorderedname = 'xordered.dat',
                                indorderedname = 'indices_ordered.dat',
                                distordname = 'distances_ordered.dat')
        
        for ii in range(1, args.nslices + 1):
            ninslice = self.nall // args.nslices
            nstart = ninslice * (ii - 1) + 1
            nend = ninslice * ii
            if ii == args.nslices:
                nend = self.nall # The last slice takes the rest
            os.system('mkdir slice%d' % ii)
            self.print_ordered_set( nstart = nstart, nend = nend,
                                xfilename = args.XfileIn,
                                xorderedname = 'slice%d/x.dat' % ii,
                                indorderedname = 'slice%d/slice_indices.dat' % ii,
                                distordname = 'slice%d/slice_distances.dat' % ii)
    
    def print_ordered_set(self, nstart = 1, nend = 'all',
                                xfilename = 'x.dat',
                                xorderedname = 'xordered.dat',
                                indorderedname = 'indices_ordered.dat',
                                distordname = 'distances_ordered.dat'):
        xlines = open(xfilename, 'r').readlines()
        if nend == 'all':
            nend = self.nall
        
        with open(xorderedname, 'w') as xof, open(indorderedname, 'w') as indof:
            for nn in range(nstart, nend + 1):
                ii = self.sorted_indices[nn - 1]
                indof.writelines('%d\n' % ii)
                xof.writelines(xlines[ii-1])
        del xlines    
        
        with open(distordname, 'w') as dof:
            for nn in range(nstart, nend + 1):
                ii = self.sorted_indices[nn - 1]
                dof.writelines('%20.12f\n' % self.Ds[ii])
                
    def sampleFromSlices(self):
        for ii in range(1, args.nslices + 1):
            ninslice = args.Ntrain // args.nslices
            nstart = ninslice * (ii - 1) + 1
            nend = ninslice * ii
            if ii == args.nslices:
                ninslice += args.Ntrain - nend
                nend = args.Ntrain # The last slice takes the rest
            argsloc = ['sample'] + args.argsSD + ['XfileIn=x.dat'] + ['Ntrain=%d' % ninslice] + ['iTrainOut=itrain.dat']
            if True: # Local calculations
                interface_MLatomF.ifMLatomCls.run(argsloc, cwdpath='slice%d' % ii)
            else: # Submission to queue
                with open('slice%d/sample.inp' % ii, 'w') as finp:
                    for arg in argsloc:
                        finp.writelines('%s\n' % arg)
                #cd os.system('cd slice%d ; bsub < ../mlatom.bsub' % ii) # To submit the job with bsub
                
    def mergeSlices(self):
        Ntrain = args.Ntrain
        
        sampled_slices = []
        unsampled_slices = []
        for islice in range(1,args.nslices+1):
            sampled_slice = []
            unsampled_slice = []
            ordindices = [] # This indices refer to the lines in the original files x.dat and y.dat corresponding to the lines in the sorted files x.dat and y.dat before slicing
            with open('slice%d/slice_indices.dat' % islice, 'r') as ff:
                    for line in ff:
                        ordindices.append(int(line))
            trindices = [] # This indices refer to the lines in the sorted files x.dat and y.dat corresponding to the lines in the files x.dat and y.dat used for prediction
            with open('slice%d/itrain.dat' % islice, 'r') as ff:
                for line in ff:
                    trindices.append(int(line))
            for jj in trindices:
                sampled_slice.append(ordindices[jj - 1])
            for jj in ordindices:
                if not jj in sampled_slice:
                    unsampled_slice.append(jj)
            sampled_slices.append(sampled_slice)
            unsampled_slices.append(unsampled_slice)
                
        slicesSizes = [len(ss) for ss in sampled_slices]
        Nsmpl = sum(slicesSizes)
        if Nsmpl < Ntrain:
            stopper.stopMLatom('Sampled less points than requested for training set')
        unslicesSizes = [len(ss) for ss in unsampled_slices]
        Nunsmpl = sum(unslicesSizes)
        
        # Sample data points into subsets
        with open('itrain.dat', 'w') as fitrain, open('itest.dat', 'w') as fitest, open('isubtrain.dat', 'w') as fisubtrain, open('ivalidate.dat', 'w') as fivalidate:
            Nsubtrain = int(Ntrain * 0.80)
            counter = 0
            exitWhileLoop = False
            islc = [-1 for ss in range(args.nslices)]
            while counter < Nsmpl:
                for islice in range(args.nslices):
                    islc[islice] += 1
                    if islc[islice] < slicesSizes[islice]:
                        counter += 1
                        if counter <= Ntrain:
                            fitrain.writelines('%d\n' % sampled_slices[islice][islc[islice]])
                            if counter <= Nsubtrain:
                                fisubtrain.writelines('%d\n' % sampled_slices[islice][islc[islice]])
                            else:
                                fivalidate.writelines('%d\n' % sampled_slices[islice][islc[islice]])
                        elif counter <= Nsmpl:
                            fitest.writelines('%d\n' % sampled_slices[islice][islc[islice]])
                        else:
                            print('breaking the loop')
                            exitWhileLoop = True
                            break
                if exitWhileLoop:
                    break
            counter = 0
            exitWhileLoop = False
            islc = [-1 for ss in range(args.nslices)]
            while counter < Nunsmpl:
                for islice in range(args.nslices):
                    islc[islice] += 1
                    if islc[islice] < unslicesSizes[islice]:
                        counter += 1
                        if counter <= Nunsmpl:
                            fitest.writelines('%d\n' % unsampled_slices[islice][islc[islice]])
                        else:
                            print('breaking the loop')
                            exitWhileLoop = True
                            break
                if exitWhileLoop:
                    break
    
class args(object):
    # Default values:
    slicedata = False
    sampleFromSlices = False
    mergeSlices = False
    nslices   = 3
    XfileIn   = None
    eqXfileIn = None
    Ntrain    = None
    # Only this Python program options
    argspy          = ['slice',
                       'sampleFromSlices',
                       'mergeSlices',
                       'nslices=*',
                       'eqXfileIn=*',
                       'Ntrain=*']
    # Only options of other programs
    argsSD = []
    
    @classmethod
    def parse(cls, argsraw):
        if len(argsraw) == 0:
            printHelp()
            stopper.stopMLatom('At least one option should be provided')
        argslower = [arg.lower() for arg in argsraw]
        if ('help' in argslower
        or '-help' in argslower
        or '-h' in argslower
        or '--help' in argslower):
            printHelp()
            stopper.stopMLatom('')

        cls.argsSD = copy.deepcopy(argsraw)
        for arg in argsraw:
            if  (arg.lower() == 'help'
              or arg.lower() == '-help'
              or arg.lower() == '-h'
              or arg.lower() == '--help'):
                printHelp()
                stopper.stopMLatom('')
            elif arg.lower()                      == 'slice'.lower():
                cls.slicedata                      = True
            elif arg.lower()                      == 'sampleFromSlices'.lower():
                cls.sampleFromSlices               = True
            elif arg.lower()                      == 'mergeSlices'.lower():
                cls.mergeSlices                    = True
            elif arg.lower()[0:len('nslices=')]   == 'nslices='.lower():
                cls.nslices                        = int(arg[len('nslices='):])
            elif arg.lower()[0:len('XfileIn=')]   == 'XfileIn='.lower():
                cls.XfileIn                        = arg[len('XfileIn='):]
            elif arg.lower()[0:len('Ntrain=')]    == 'Ntrain='.lower():
                cls.Ntrain                         = int(arg[len('Ntrain='):])
            elif arg.lower()[0:len('eqXfileIn=')] == 'eqXfileIn='.lower():
                cls.eqXfileIn                      = arg[len('eqXfileIn='):]
            #else: # Other options are usually passed over to the interfaced programs
            #    printHelp()
            #    stopper.stopMLatom('Option "%s" is not recognized' % arg)
            for argpy in cls.argspy:
                flagmatch = re.search(argpy, arg.lower(), flags=re.UNICODE | re.MULTILINE | re.DOTALL | re.IGNORECASE)
                if flagmatch:
                    cls.argsSD.remove(arg)
                    break
        cls.checkArgs()
            
    @classmethod
    def checkArgs(cls):
        Ntasks = (cls.slicedata + cls.sampleFromSlices + cls.mergeSlices)
        if Ntasks == 0:
            printHelp()
            stopper.stopMLatom('At least one task should be requested')
        if cls.slicedata:
            if cls.XfileIn == None:
                printHelp()
                stopper.stopMLatom('Provide correct file name with XfileIn option')
            if cls.eqXfileIn == None:
                printHelp()
                stopper.stopMLatom('Provide correct file name with eqXfileIn option')
        if cls.sampleFromSlices:
            if cls.Ntrain == None:
                printHelp()
                stopper.stopMLatom('Provide the number of training points with Ntrain option')
        if cls.mergeSlices:
            if cls.Ntrain == None:
                printHelp()
                stopper.stopMLatom('Provide the number of training points with Ntrain option')
                
def printHelp():
    if __name__ == '__main__':
        print('''
  !---------------------------------------------------------------------------! 
  !                                                                           ! 
  !              sliceData: Scripts for working with sliced data              ! 
  !                                                                           ! 
  !---------------------------------------------------------------------------!
  
  Usage:
    sliceData.py [options]
    
  Options:
      help            print this help and exit
    
    Tasks for sliceData.py. At least one task should be requested.
''')
    helpText = '''      slice           slice data set
        nslices=N     number of slices [default = 3]
        XfileIn=S     file S with input vectors X
        eqXfileIn=S   file S with input vector for the equilibrium geometry
      sampleFromSlices   sample from each slice
        nslices=N     number N of slices [default = 3]
        Ntrain=N      total integer number N of training points from all slices
      mergeSlices     merges indices from slices [see sliceData help]
        nslices=N     number N of slices [default = 3]
        Ntrain=N      total integer number N of training points from all slices
'''
    print(helpText)

if __name__ == '__main__':
    print(__doc__)
    sliceDataCls()
