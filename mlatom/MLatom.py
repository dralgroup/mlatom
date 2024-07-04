#!/usr/bin/env python3
'''

  !---------------------------------------------------------------------------!
  !                                                                           !
  !     MLatom: a Package for Atomistic Simulations with Machine Learning     !
  !                             MLatom 3.7.1                                  !
  !                                   @                                       !
  !                 Xiamen Atomistic Computing Suite (XACS)                   !
  !                                                                           !
  !                http://mlatom.com/ @ https://XACScloud.com                 !
  !                                                                           !
  !            MIT License (modified to request proper citations)             !
  !                   Copyright (c) 2013- Pavlo O. Dral                       !
  !                           http://dr-dral.com/                             !
  !                                                                           !
  ! Permission is hereby granted, free of charge, to any person obtaining a   !
  ! copy of this software and associated documentation files (the "Software"),!
  ! to deal in the Software without restriction, including without limitation !
  ! the rights to use, copy, modify, merge, publish, distribute, sublicense,  !
  ! and/or sell copies of the Software, and to permit persons to whom the     ! 
  ! Software is furnished to do so, subject to the following conditions:      !
  !                                                                           !
  ! The above copyright notice and this permission notice shall be included   !
  ! in all copies or substantial portions of the Software.                    !
  ! When this Software or its derivatives are used                            ! 
  ! in scientific publications, it shall be cited as:                         !
  !                                                                           !
  ! Pavlo O. Dral, Fuchun Ge, Yi-Fan Hou, Peikun Zheng, Yuxinxin Chen,        ! 
  ! Mario Barbatti, Olexandr Isayev, Cheng Wang, Bao-Xin Xue,                 !
  ! Max Pinheiro Jr, Yuming Su, Yiheng Dai, Yangtao Chen, Lina Zhang,         ! 
  ! Shuang Zhang, Arif Ullah, Quanhao Zhang, Yanchi Ou.                       !
  ! J. Chem. Theory Comput. 2024, 20, 1193-1213.                              !
  !                                                                           !
  ! Pavlo O. Dral, Fuchun Ge, Yi-Fan Hou, Peikun Zheng, Yuxinxin Chen, Bao-Xin!
  ! Xue, Max Pinheiro Jr, Yuming Su, Yiheng Dai, Yangtao Chen, Shuang Zhang,  !
  ! Lina Zhang, Arif Ullah, Quanhao Zhang, Sebastian V. Pios, Yanchi Ou,      !
  ! MLatom: A Package for Atomistic Simulations with Machine Learning,        !
  ! version 3.7.1, Xiamen University, Xiamen, China, 2013-2024.               !
  !                                                                           !
  ! The citations for MLatom's interfaces and features shall be eventually    !
  ! included too. See header.py, ref.json and http://mlatom.com.              !
  !                                                                           !
  ! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS   !
  ! OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF                !
  ! MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN !
  ! NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,  !
  ! DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR     !
  ! OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE !
  ! USE OR OTHER DEALINGS IN THE SOFTWARE.                                    !
  !                                                                           !  
  !---------------------------------------------------------------------------!

'''

import os, sys, time
from mlatom import header
from mlatom.MLtasks import CLItasks
from mlatom.args_class import mlatom_args

def run(argv = []):
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
    args = mlatom_args()
    if argv == []:
        args.parse(sys.argv[1:])
    else:
        args.parse(argv)
    print(' %s ' % ('='*78))
    header.printHeader(args)
    sys.stdout.flush()
    
    result = CLItasks(args).run()
    
    endtime = time.time()
    wallclock = endtime - starttime
    print(' %s ' % ('='*78))
    print(' Wall-clock time: %.2f s (%.2f min, %.2f hours)\n' % (wallclock, wallclock / 60.0, wallclock / 3600.0))
    print(time.strftime(" MLatom terminated on %d.%m.%Y at %H:%M:%S", time.localtime()))
    print(' %s ' % ('='*78), flush=True)
    
    return result

if __name__ == '__main__':
    run()
