#!/usr/bin/python3
'''
  !---------------------------------------------------------------------------! 
  !                                                                           ! 
  !             Interface_FCHL: Interface between FCHL and MLatom               ! 
  !                                                                           ! 
  !---------------------------------------------------------------------------! 
  
'''
import numpy as np
import os, sys, subprocess, time, shutil, re, math, random
import stopper

filedir = os.path.dirname(__file__)
try:
    FCHLbin = os.environ['FCHL']
except:
    FCHLbin = ''

class FCHLCls(object):
    def __init__(self, argsFCHL = sys.argv[1:]):
        print(' ___________________________________________________________\n\n%s' % __doc__)
    
    @classmethod
    def convertdata(cls):
        pass

    @classmethod
    def createMLmodel(cls, argsFCHL):
        pass

    @classmethod
    def useMLmodel(cls, argsFCHL):

class args(object):
    # MLatom args
    xyzfile             = ""
    yfile               = ""
    ygradxyzfile            = ""
    virialfile          = ""
    ntrain              = 0
    ntest               = 0
    nsubtrain           = 0
    nvalidate           = 0
    sampling            = "random"
    itrainin            = ""
    itestin             = ""
    isubtrainin         = ""
    ivalidatein         = ""
    mlmodelout          = "GAPmodel.xml"
    mlmodelin           = ""
    yestfile            = "enest.dat"
    ygradxyzestfile     = "gradest.dat"

    # args for this interface
    natom               = 0
    atype               = []


    @classmethod
    def parse(cls,argsraw):
        if len(argsraw) == 0:
            printHelp()
            stopper.stopMLatom('At least one option should be provided')
        for arg in argsraw:
            if (arg.lower() == 'help'
              or arg.lower() == '-help'
              or arg.lower() == '-h'
              or arg.lower() == '--help'):
                printHelp()
                stopper.stopMLatom('')
            # elif arg.lower()[:7]=="gapfit.":
            #     if arg.lower()[:11]=="gapfit.gap.":
            #         exec('cls.gapdic[arg.lower().split(".")[2].split("=")[0]]=arg.lower().split("=",1)[1]')
            #         cls.gap = '{'+' '.join([k+'='+v for k,v in cls.gapdic.items()])[5:]+'}'
            #         cls.gapfitdic['gap']=cls.gap
            #     else:
            #         exec('cls.gapfitdic[arg.lower().split(".")[1].split("=")[0]]=arg.lower().split("=",1)[1]')   
            elif len(arg.lower()) == 1:                             # parse boolean args
                try:
                    exec('cls.'+arg.lower())
                    exec('cls.'+arg.lower()+'=True')
                except: pass
            else:                                               # parse other args
                try:
                    exec('cls.'+arg.split('=')[0].lower())
                    if type(eval('cls.'+arg.split('=')[0].lower())) == str :
                        exec('cls.'+arg.split('=')[0].lower()+'='+"arg.split('=')[1]")
                    else:
                        exec('cls.'+arg.split('=')[0].lower()+'='+arg.split('=')[1])
                except:
                    pass

        # calc. & fix something args that useful and you don't want users to change...
        with open(cls.xyzfile,'r') as f:
            cls.natom = int(f.readline())
            exec('f.readline()')
            cls.atype = [f.readline().split()[0] for i in range(cls.natom)]
        
        # for estAccMLmodel that MLmodelIn may not be provided
        if not cls.mlmodelin:
            cls.mlmodelin = cls.mlmodelout

            
def printHelp():
    helpText = '''
  !---------------------------------------------------------------------------! 
  !                                                                           ! 
  !            Interface_FCHL: Interface between FCHL and MLatom              ! 
  !                                                                           ! 
  !---------------------------------------------------------------------------! 

  To use Interface_FCHL, please define ......
  
  Options:
      help            print this help and exit
'''
    print(helpText)

if __name__ == '__main__':
    FCHLCls()