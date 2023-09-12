#!/usr/bin/env python
# coding: utf-8
'''
  !---------------------------------------------------------------------------! 
  ! xyz: different operations on XYZ coordinates                              ! 
  ! Implementations by: Fuchun Ge                                             !    
  !---------------------------------------------------------------------------! 
'''

import numpy as np

def rmsd(xyz1, xyz2):
    xyz1 = xyz1 - get_center_of_mass(xyz1)
    xyz2 = xyz2 - get_center_of_mass(xyz2)

    result = 0.0

    aligned_xyz1 = align(xyz1, xyz2)
    result = np.sqrt(np.mean(np.square(aligned_xyz1-xyz2)))
    
    return result

def rotation_matrix(xyz, reference_xyz):
    import rmsd
    Mr = rmsd.kabsch(xyz, reference_xyz)
    return Mr

def align(xyz, reference_xyz):
    aligned_xyz = xyz.dot(rotation_matrix(xyz,reference_xyz))
    return aligned_xyz

def get_center_of_mass(xyz, nuclear_masses=None):
    # Can also take many xyzs and associated nuclear_masses (tensor operations)
    if nuclear_masses is None:
        nuclear_masses = np.ones(xyz.shape[-2])
    return np.sum(xyz*nuclear_masses[:,np.newaxis],axis=-2,keepdims=True)/np.sum(nuclear_masses)