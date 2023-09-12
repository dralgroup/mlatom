#!/usr/bin/env python3
'''
  !---------------------------------------------------------------------------! 
  ! stopper: Stopping MLatom and writing an error message                     ! 
  ! Implementations by: Pavlo O. Dral                                         ! 
  !---------------------------------------------------------------------------! 
'''

import sys

def stopMLatom(errorMsg):
    if errorMsg != '':
        print(' <!> %s <!>' % errorMsg)
    sys.exit()

if __name__ == '__main__':
    stopMLatom('Call the main program!')
