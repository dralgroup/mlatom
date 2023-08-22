#!/usr/bin/env python
# coding: utf-8


'''
  !---------------------------------------------------------------------------! 
  ! RMSD: align and calculate 2D-RMSD                                         ! 
  ! Implementations by: Fuchun Ge                                             ! 
  !---------------------------------------------------------------------------! 
'''

import rmsd
import numpy as np
import sys
import stopper
from args_class import ArgsBase 

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.args2pass = [] 
        self.add_default_dict_args([ 
            "RMSD","alignXYZ",
            ],
            False
        )

        self.add_dict_args({
            'xyzfiles':'',
            'RMSDout':'RMSD.txt',
            'alignedXYZ':'',
            'fmt':'%16.8g',
        })

    def parse(self,argsraw):
        self.parse_input_content(argsraw)
        self.args2pass = self.args_string_list(['',None])
        self.argProcess()

    def argProcess(self):
        if self.xyzfiles:
            self.xyzfiles=self.xyzfiles.split(',')
            if len(self.xyzfiles)<2:
                stopper.stopMLatom(' please provide two xyzfiles, e.g. xyzfiles=mol1.xyz,mol2.xyz')
            elif len(self.xyzfiles)>2:
                print(' only first two xyzfiles are to be used')
                self.xyzfiles=self.xyzfiles[:2]
        else:
            stopper.stopMLatom(' please provide two xyzfiles, e.g. xyzfiles=mol1.xyz,mol2.xyz')
        if not self.alignedXYZ:
            self.alignedXYZ='aligned_'+self.xyzfiles[1]

class RMSDcls(object):
    def __init__(self,
                argsMLTPA
                 ) -> None:

        args=Args()
        args.parse(argsMLTPA)
        mol1=loadXYZ(args.xyzfiles[0])[0]
        mol2, [sp,*_]=loadXYZ(args.xyzfiles[1])

        mol1=mol1-getCoM(mol1)
        mol2=mol2-getCoM(mol2)

        result=[]
        aligned=[]
        if args.RMSD:
            for i in range(len(mol1)):
                result_=[]
                for j in range(len(mol2)):
                    print(f'\r{i}-{j}',end='')
                    mol2_align=mol2[j].dot(align(mol2[j],mol1[i]))
                    if i==j:
                        aligned.append(mol2_align)
                    result_.append(np.sqrt(np.mean(np.square(mol2_align-mol1[i]))))
                result.append(result_)

            np.savetxt(args.rmsdout,np.array(result),fmt=args.fmt)
            saveXYZ(args.alignedXYZ,np.array(aligned),sp)
        else:
            for i in range(min(len(mol1),len(mol2))):
                print('\r'+i,end='')
                aligned.append(mol2[i].dot(align(mol2[i],mol1[i])))
        
            saveXYZ(args.alignedXYZ,np.array(aligned),sp)
        print('')

def align(mol,ref):
    Mr=rmsd.kabsch(mol,ref)
    return Mr

def getCoM(xyz,m=None):
    if m is None:
        m=np.ones(xyz.shape[-2])
    return np.sum(xyz*m[:,np.newaxis],axis=-2,keepdims=True)/np.sum(m)

def loadXYZ(fname,dtype=np.array,getsp=True):
    xyz=[]
    sp=[]
    with open(fname) as f:
        for line in f:
            xyz_=[]
            sp_=[]
            natom=int(line)
            f.readline()
            for _ in range(natom):
                if getsp: 
                    _sp_,*_xyz_=f.readline().split()
                    sp_.append(_sp_)
                else: _xyz_=f.readline().split()[-3:]
                xyz_.append(_xyz_)
            xyz.append(np.array(xyz_).astype(float))
            sp.append(np.array(sp_))
    return dtype(xyz),dtype(sp)

def saveXYZ(fname,xyzs,sp,mode='w',msgs=None):
    xyzs=xyzs.reshape(-1,sp.shape[0],3)
    with open(fname,mode)as f:
        for xyz in xyzs:
            f.write("%d\n"%sp.shape[0])
            if msgs: f.write(msgs.pop(0)+"\n")
            else: f.write('\n')
            for j in range(sp.shape[0]):
                f.write(sp[j]+' ')
                f.write('%12.8f %12.8f %12.8f\n'%tuple(xyz[j]))