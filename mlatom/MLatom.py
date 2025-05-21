#!/usr/bin/env python3
'''

  !---------------------------------------------------------------------------!
  !                                                                           !
  !     MLatom: a Package for Atomistic Simulations with Machine Learning     !
  !                             MLatom 3.17.3                                 !
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
  ! Pavlo O. Dral, Fuchun Ge, Yi-Fan Hou, Peikun Zheng, Yuxinxin Chen,        !
  ! Bao-Xin Xue, Mikolaj Martyka, Max Pinheiro Jr, Yuming Su, Yiheng Dai,     !
  ! Yangtao Chen, Shuang Zhang, Lina Zhang, Arif Ullah, Quanhao Zhang,        !
  ! Sebastian V. Pios, Yanchi Ou, Matheus O. Bispo, Vignesh B. Kumar,         !
  ! Xin-Yu Tong,                                                              !
  ! MLatom: A Package for Atomistic Simulations with Machine Learning,        !
  ! version 3.17.3, Xiamen University, Xiamen, China, 2013-2024.              !
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
# import header
# from mlatom.MLtasks import CLItasks
# from mlatom.args_class import mlatom_args

def run(argv = []):
    import importlib.util
    # ~POD, 2025.03.23
    # the complicated import below is required to load the same instance of mlatom,
    # where this script is located.

    dir_path = os.path.dirname(os.path.abspath(__file__))
    path2init = os.path.join(dir_path, '__init__.py')
    dirname = os.path.basename(dir_path)
    spec = importlib.util.spec_from_file_location(dirname, path2init)
    mlatom_with_this_file = importlib.util.module_from_spec(spec)
    sys.modules[dirname] = mlatom_with_this_file
    spec.loader.exec_module(mlatom_with_this_file)
    header = mlatom_with_this_file.header
    CLItasks = mlatom_with_this_file.MLtasks.CLItasks
    mlatom_args = mlatom_with_this_file.args_class.mlatom_args
    
    starttime = time.time()

    # add print mlatom version and list location of these versions
    for arg in sys.argv:
        if arg in ['-v', '--version']: 
            version = mlatom_with_this_file.__version__
            print(f'Current mlatom version: {version}')
            if 'dev' not in version: 
                # get latest version from pypi
                import requests 
                url = f"https://pypi.org/pypi/mlatom/json"
                response = requests.get(url)
                if response.status_code == 200:
                    latest_version = response.json()["info"]['version']
                    if latest_version != version:
                        print(f'The latest mlatom version is {latest_version}, please upgrade your mlatom with `pip install --upgrade mlatom`')
                else:
                    print('Fail to get the latest mlatom version from pypi')
            return
        elif arg in ['-l', '--list']:
            # get all installed packages in site packages
            print('Current mlatom installation:')
            print(f"  {'version':<15} location")
            print(f'* {mlatom_with_this_file.__version__:<15} {dir_path}')

            print('\nAvailable mlatom installation:')
            print(f"{'version':<15} location")

            def get_version_from_init(initpath, prefix):
                pp_version = None; f = open(initpath,'r').readlines()
                for ll in f:
                    if '__version__' in ll: 
                        import re 
                        pp_version = re.search(r'(["\'])(.*?)\1', ll).group(2)
                if not pp_version: pp_version = 'unknown'
                print(f'{pp_version:<15} {prefix}')


            # check pythonpath
            if 'PYTHONPATH' in os.environ: 
                pythonpaths = os.environ['PYTHONPATH'].split(os.pathsep)
                for pp in pythonpaths:
                    if os.path.exists(os.path.join(pp, 'mlatom')):
                        path2init = os.path.join(pp, 'mlatom/__init__.py')
                    elif os.path.exists(os.path.join(pp, 'aitomic')):
                        path2init = os.path.join(pp, 'aitomic/__init__.py')
                    else: continue
                    get_version_from_init(path2init, os.path.join(pp, 'mlatom'))

            # check python package
            import site
            sitepackage_paths = site.getsitepackages()
            for sp in sitepackage_paths:
                if os.path.exists(os.path.join(sp, 'mlatom')): 
                    path2init = os.path.join(sp, 'mlatom/__init__.py')
                    get_version_from_init(path2init, os.path.join(sp, 'mlatom'))
            return  
    
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
