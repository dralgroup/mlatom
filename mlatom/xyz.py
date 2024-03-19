#!/usr/bin/env python
# coding: utf-8
'''
  !---------------------------------------------------------------------------! 
  ! xyz: different operations on XYZ coordinates                              ! 
  ! Implementations by: Fuchun Ge                                             ! 
  ! To-do: implement permutation (Hungarian algorithm)                        !    
  !---------------------------------------------------------------------------! 
'''

import numpy as np

def rmsd(xyz1, xyz2):
    xyz1 = xyz1 - get_center_of_mass(xyz1)
    xyz2 = xyz2 - get_center_of_mass(xyz2)

    result = 0.0

    aligned_xyz1 = align(xyz1, xyz2)
    result = np.sqrt(3*np.mean(np.square(aligned_xyz1-xyz2)))
    
    return result

def rmsd_reorder(atoms1,atoms2,xyz1,xyz2,reorder='Hungarian'):
    xyz1 = xyz1 - get_center_of_mass(xyz1)
    xyz2 = xyz2 - get_center_of_mass(xyz2)
    import rmsd
    if reorder.casefold() == 'Hungarian'.casefold():
        order = rmsd.reorder_hungarian(atoms1,atoms2,xyz1,xyz2)
        xyz2 = xyz2[order]
        atoms2 = xyz2[order]
    result = rmsd.kabsch_rmsd(xyz1,xyz2)
    
    return result

def rmsd_reorder_check_reflection(atoms1,atoms2,xyz1,xyz2,reorder='Hungarian',keep_stereo=False):
    xyz1 = xyz1 - get_center_of_mass(xyz1)
    xyz2 = xyz2 - get_center_of_mass(xyz2)
    import rmsd
    if reorder.casefold() == 'Hungarian'.casefold():
        result,q_swap,q_reflection,order = rmsd.check_reflections(
            atoms1,
            atoms2,
            xyz1,
            xyz2,
            reorder_method=rmsd.reorder_hungarian,
            rmsd_method=rmsd.kabsch_rmsd,
            keep_stereo=keep_stereo,
        )    
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