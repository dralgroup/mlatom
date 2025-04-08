#!/usr/bin/env python
# coding: utf-8
'''
  !---------------------------------------------------------------------------! 
  ! xyz: different operations on XYZ coordinates                              ! 
  ! Implementations by: Fuchun Ge, Yi-Fan Hou, Pavlo O. Dral                  ! 
  !---------------------------------------------------------------------------! 
'''

import numpy as np

def rmsd(molecule1, molecule2, reorder=False, check_reflection=False, keep_stereo=False):
    '''
    Calculate RMSD (root-mean-squared deviation) between two structures.
    
    Arguments:
        molecule1 (required): either molecule class instance or numpy array with xyz coordinates
        molecule2 (required): either molecule class instance or numpy array with xyz coordinates
        reorder (bool or str, optional): whether to try to reorder atoms to get smaller RMSD (default: False; other supported options are True which is equivalent to Hungarian)
        check_reflection (bool, optional): check reflections (default: False).
        keep_stereo (bool, optional): (default: False).
        
    .. table::
        :align: center

        ========================  ===========================================================================
        reorder                    description
        ========================  ===========================================================================
        ``None`` (default)         no reorder
        ``'QML'``                  reorder using QML similarity and Hungarian method for assignment
        ``'Hungarian'``            reorder using Hungarian alogrithm
        ``'Inertia-Hungarian'``    align the principal inertia axis then reorder using Hungarian algorithm
        ``'Brute'``                reorder using all possible permutations
        ``'Distance'``             reorder by atom type and then by distance of each atom from the centroid
        ========================  ===========================================================================

        
    Example of the simple use:
        rmsd = mlatom.xyz.rmsd(molecule1.xyz_coordinates, molecule2.xyz_coordinates)
    Example of using Hungarian algorithm to check for homonuclear atom permutation and reflections:
        rmsd = mlatom.xyz.rmsd(molecule1, molecule2, reorder='Hungarian', check_reflection=True)
    '''
    if isinstance(molecule1, np.ndarray):
        xyz1 = molecule1
        xyz2 = molecule2
    else:
        xyz1 = molecule1.xyz_coordinates
        xyz2 = molecule2.xyz_coordinates
        
    result = 0.0
    
    if not reorder and not check_reflection:
    
        xyz1 = xyz1 - get_center_of_mass(xyz1)
        xyz2 = xyz2 - get_center_of_mass(xyz2)


        aligned_xyz1 = align(xyz1, xyz2)
        result = np.sqrt(3*np.mean(np.square(aligned_xyz1-xyz2)))
    
    elif reorder:
        if type(reorder) == bool: reorder = 'Hungarian'
        # atoms1 = molecule1.element_symbols
        # atoms2 = molecule2.element_symbols
        atoms1 = molecule1.atomic_numbers
        atoms2 = molecule2.atomic_numbers
        if not check_reflection:
            result = rmsd_reorder(atoms1,atoms2,xyz1,xyz2,reorder=reorder)
        else:
            result = rmsd_reorder_check_reflection(atoms1,atoms2,xyz1,xyz2,reorder=reorder,keep_stereo=keep_stereo)
        # if reorder.casefold() == 'Hungarian'.casefold() and not check_reflection:
        #     result = rmsd_reorder(atoms1,atoms2,xyz1,xyz2,reorder=reorder)
        # elif reorder.casefold() == 'Hungarian'.casefold() and check_reflection:
        #     result = rmsd_reorder_check_reflection(atoms1,atoms2,xyz1,xyz2,reorder=reorder,keep_stereo=keep_stereo)
        # else:
        #     raise ValueError('Unsupported type of RMSD calculations')
    else:
        raise ValueError('Unsupported type of RMSD calculations')
    
    return result

def rmsd_reorder(atoms1,atoms2,xyz1,xyz2,reorder='Hungarian'):
    xyz1 = xyz1 - get_center_of_mass(xyz1)
    xyz2 = xyz2 - get_center_of_mass(xyz2)
    import rmsd
    if reorder.casefold() == 'QML'.casefold():
        reorder_method = rmsd.reorder_similarity
    elif reorder.casefold() == 'Hungarian'.casefold():
        reorder_method = rmsd.reorder_hungarian
    elif reorder.casefold() == 'inertia-Hungarian'.casefold():
        reorder_method = rmsd.reorder_inertia_hungarian
    elif reorder.casefold() == 'brute'.casefold():
        reorder_method = rmsd.reorder_brute
    elif reorder.casefold() == 'distance'.casefold():
        reorder_method = rmsd.reorder_distance
    else:
        raise ValueError(f"Unsupported reorder method: {reorder}")
    # if reorder.casefold() == 'Hungarian'.casefold():
    order = reorder_method(atoms1,atoms2,xyz1,xyz2)
    xyz2 = xyz2[order]
    atoms2 = xyz2[order]
    result = rmsd.kabsch_rmsd(xyz1,xyz2)
    
    return result

def rmsd_reorder_check_reflection(atoms1,atoms2,xyz1,xyz2,reorder='Hungarian',keep_stereo=False):
    xyz1 = xyz1 - get_center_of_mass(xyz1)
    xyz2 = xyz2 - get_center_of_mass(xyz2)
    import rmsd
    if reorder.casefold() == 'QML'.casefold():
        reorder_method = rmsd.reorder_similarity
    elif reorder.casefold() == 'Hungarian'.casefold():
        reorder_method = rmsd.reorder_hungarian
    elif reorder.casefold() == 'inertia-Hungarian'.casefold():
        reorder_method = rmsd.reorder_inertia_hungarian
    elif reorder.casefold() == 'brute'.casefold():
        reorder_method = rmsd.reorder_brute
    elif reorder.casefold() == 'distance'.casefold():
        reorder_method = rmsd.reorder_distance
    else:
        raise ValueError(f"Unsupported reorder method: {reorder}")
    result,q_swap,q_reflection,order = rmsd.check_reflections(
        atoms1,
        atoms2,
        xyz1,
        xyz2,
        reorder_method=reorder_method,
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