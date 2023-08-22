#!/usr/bin/env python3
'''

  !---------------------------------------------------------------------------!
  !                                                                           !
  !     MLatom: a Package for Atomistic Simulations with Machine Learning     !
  !                           MLatom@XACS version                             !
  !                           http://mlatom.com/                              !
  !                                                                           !
  !                  Copyright (c) 2013-2022 Pavlo O. Dral                    !
  !                           http://dr-dral.com/                             !
  !                                                                           !
  ! All rights reserved. This work is licensed under the                      !
  ! Attribution-NonCommercial-NoDerivatives 4.0 International                 ! 
  ! (http://creativecommons.org/licenses/by-nc-nd/4.0/) license.              !
  ! See LICENSE.CC-BY-NC-ND-4.0.                                              !
  !                                                                           !
  ! The above copyright notice and this permission notice shall be included   !
  ! in all copies or substantial portions of the Software.                    !
  !                                                                           !
  ! The software is provided "as is", without warranty of any kind, express   !
  ! or implied, including but not limited to the warranties of                !
  ! merchantability, fitness for a particular purpose and noninfringement. In !
  ! no event shall the authors or copyright holders be liable for any claim,  !
  ! damages or other liability, whether in an action of contract, tort or     !
  ! otherwise, arising from, out of or in connection with the software or the !
  ! use or other dealings in the software.                                    !
  !                                                                           !
  !                                Cite as:                                   !
  ! Pavlo O. Dral, J. Comput. Chem. 2019, 40, 2339-2347                       !
  ! Pavlo O. Dral, Fuchun Ge, Bao-Xin Xue, Yi-Fan Hou, Max Pinheiro Jr,       !
  ! Jianxing Huang, Mario Barbatti, Top. Curr. Chem. 2021, 379, 27            !
  !                                                                           !
  ! Pavlo O. Dral, Peikun Zheng, Bao-Xin Xue, Fuchun Ge, Yi-Fan Hou,          !
  ! Max Pinheiro Jr, Yuming Su, Yiheng Dai, Yangtao Chen,                     !
  ! MLatom: A Package for Atomistic Simulations with Machine Learning         !
  ! MLatom@XACS version, Xiamen University, Xiamen, China, 2013-2022          !
  !                                                                           !  
  !---------------------------------------------------------------------------!

'''

import os, sys, time
try:
    from . import header
    from . import MLtasks
    from . import sliceData
    from . import stopper
    from . import interface_MLatomF
    from .args_class import ArgsBase
    from .doc import Doc
except:
    import header
    import MLtasks
    import sliceData
    import stopper
    import interface_MLatomF
    from args_class import ArgsBase
    from doc import Doc

class run(object):
    def __init__(self, argv = []):
        starttime = time.time()
        
        print(__doc__)
        
        print(' %s ' % ('='*78))
        print(time.strftime(" MLatom started on %d.%m.%Y at %H:%M:%S", time.localtime()))
        print('        with the following options:')
        argsstr = '        '
        for arg in sys.argv:
            argsstr += arg + ' '
        print(argsstr.rstrip())
        if len(sys.argv[1:]) == 1:
            if os.path.exists(sys.argv[1]):
                print('\n        Input file content:')
                print(' %s ' % ('_'*78))
                with open(sys.argv[1], 'r') as finp:
                    for line in finp:
                        print(line.rstrip())
                print(' %s ' % ('_'*78))
        args = Args()
        if argv == []:
            args.parse(sys.argv[1:])
        else:
            args.parse(argv)
        print(' %s ' % ('='*78))
        header.printHeader(args)
        sys.stdout.flush()

        # Set the number of threads
        if args.nthreads:
            os.environ["OMP_NUM_THREADS"] = str(args.nthreads)
            os.environ["MKL_NUM_THREADS"] = str(args.nthreads)
            # os.environ["OMP_PROC_BIND"]   = 'true'
        try:
            f_cpu = open('/proc/cpuinfo')
            for line in f_cpu:
                if 'AMD'.lower() in line.lower():
                    os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
                    break
            f_cpu.close()
        except: pass
        
        # Perform requested task
        if args.XYZ2X or args.sample or args.analyze:
            interface_MLatomF.ifMLatomCls.run(args.args2pass)
            pass
        elif args.slice or args.sampleFromSlices or args.mergeSlices:
            sliceData.sliceDataCls(argsSD = args.args2pass)
        elif (args.useMLmodel    or args.createMLmodel or
              args.estAccMLmodel or args.learningCurve or
              args.reactionPath or
              args.callNXinterface or # DEVELOPMENT VERSION
              args.AIQM1 or args.AIQM1DFTstar or args.AIQM1DFT
              or args.ODM2 or args.ODM2star or args.CCSDTstarCBS or args.gfn2xtb or
              args.ani1x or args.ani2x or args.ani1ccx or args.ani1xd4 or args.ani2xd4 or
              args.geomopt or args.freq or args.ts or args.irc or
              args.MLNEA or args.MD or args.IRSS or args.Gaussian or 
              args.MLTPA or args.XYZ2SMI or args.mlqd
              or args.RMSD or args.AlignXYZ):
            MLtasks.MLtasksCls(argsMLtasks = args.args2pass)
        
        endtime = time.time()
        wallclock = endtime - starttime
        print(' %s ' % ('='*78))
        print(' Wall-clock time: %.2f s (%.2f min, %.2f hours)\n' % (wallclock, wallclock / 60.0, wallclock / 3600.0))
        print(time.strftime(" MLatom terminated on %d.%m.%Y at %H:%M:%S", time.localtime()))
        print(' %s ' % ('='*78))
        sys.stdout.flush()

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.task_list = [
            'XYZ2X', 'analyze', 'sample', 'sampleFromSlices', 'mergeSlices','slice',
            'useMLmodel', 'createMLmodel', 'estAccMLmodel', 'learningCurve', 
            'crossSection', # Interfaces
            'MD', 'IRSS','Gaussian', 'mlqd',
            'reactionPath','activeLearning',
            'callNXinterface', # DEVELOPMENT VERSION
            'AIQM1', 'AIQM1DFT', 'AIQM1DFTstar',
            'ODM2', 'ODM2star', 'CCSDTstarCBS', 'gfn2xtb',
            'geomopt', 'freq', 'ts', 'irc',
            'ani1x', 'ani2x', 'ani1ccx', 'ani1xd4', 'ani2xd4',
            'MLTPA',
            'XYZ2SMI',
            'RMSD', 'AlignXYZ',
        ]
        self.add_default_dict_args(self.task_list, bool)
        self.add_default_dict_args([
            'useMLmodel', 'createMLmodel', 'estAccMLmodel', 'learningCurve','deltaLearn','selfCorrect','MLmodelType',
            'geomopt', 'freq', 'ts', 'irc','IRSS','MD',
            ],
            bool
        )
        self.add_default_dict_args([
            'MLprog'
            ],
            ''
        )
        self.add_default_dict_args([
            'XYZfile', 'XfileIn',
            'Yfile', 'YestFile', 'Yb', 'Yt', 'YestT',
            'YgradFile', 'YgradEstFile', 'YgradB', 'YgradT', 'YgradEstT',
            'YgradXYZfile', 'YgradXYZestFile', 'YgradXYZb', 'YgradXYZt', 'YgradXYZestT',
            'hessianestfile',
            'absXfileIn', 'absXYZfile', 'absYfile', 'absYgradXYZfile',
            'Nuse', 'Ntrain', 'Nsubtrain', 'Nvalidate', 'Ntest', 'iTrainIn', 'iTestIn', 'iSubtrainIn', 'iValidateIn', 'sampling', 'MLmodelIn', 'MLmodelOut',
            'molDescriptor', 'kernel', 'mndokeywords'
            ],
            ''
        )
        self.add_dict_args({'MLmodelType': '', 'nthreads': None})
        self.set_keyword_alias('crossSection', ['ML-NEA', 'ML_NEA', 'crossSection', 'cross-section', 'cross_section','MLNEA'])
        self.set_keyword_alias('AIQM1DFTstar', ['AIQM1@DFT*'])
        self.set_keyword_alias('AIQM1DFT', ['AIQM1@DFT'])
        self.set_keyword_alias('ODM2star', ['ODM2*'])
        self.set_keyword_alias('CCSDTstarCBS', ['CCSD(T)*/CBS'])
        self.set_keyword_alias('ani1ccx', ['ANI-1ccx'])
        self.set_keyword_alias('ani1x', ['ANI-1x'])
        self.set_keyword_alias('ani2x', ['ANI-2x'])
        self.set_keyword_alias('ani1xd4', ['ANI-1x-D4'])
        self.set_keyword_alias('ani2xd4', ['ANI-2x-D4'])
        self.set_keyword_alias('gfn2xtb', ['GFN2-xTB'])
        self.args2pass = []

    def parse(self, argsraw):
        if len(argsraw) == 0:
            Doc.printDoc({})
            stopper.stopMLatom('At least one option should be provided')
        elif len(argsraw) == 1:
            if os.path.exists(argsraw[0]):
                self.parse_input_file(argsraw[0])
            else:
                self.parse_input_content(argsraw[0])
            self.args2pass = self.args_string_list(['', None])
        elif len(argsraw) >= 2:
            self.parse_input_content(argsraw)
            self.args2pass = self.args_string_list(['', None])
        self.checkArgs()

    def checkArgs(self):
        Ntasks = eval(' + '.join(map(lambda s: 'self.' + s, self.task_list)))
        if Ntasks == 0:
            Doc.printDoc({})
            stopper.stopMLatom('At least one task should be requested')
        elif self.callNXinterface and Ntasks > 1: # DEVELOPMENT VERSION
            Doc.printDoc({})
            stopper.stopMLatom('callNXinterface cannot be combined with any other task')

if __name__ == '__main__':
    run()
