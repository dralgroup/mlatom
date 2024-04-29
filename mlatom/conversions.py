import numpy as np

def smi2xyz(smi_strings):
    from openbabel import pybel as pb
    if isinstance(smi_strings, str):
        smi_strings =  smi_strings.strip().split('\n')
    xyz_strings = []
    for smi_string in smi_strings:
        mol = pb.readstring('smiles', smi_string)
        mol.make3D()
        xyz_strings.append(mol.write(format='xyz'))
    return ''.join(xyz_strings)

def xyz2smi(xyz_strings):
    from openbabel import pybel as pb
    if isinstance(xyz_strings, str):
        xyz_strings = split_xyz_string(xyz_strings)
    smi_strings = []
    for string in xyz_strings:
        mol = pb.readstring('xyz', string)
        smi_strings.append(mol.write(format='smiles'))
    return ''.join(smi_strings)

def split_xyz_string(string):
    lines = string.strip().split('\n')
    xyz_strings = []
    new_string =True
    for line in lines:
        if new_string:
            xyz_string = [line]
            lines_left = int(line) + 1
            new_string = False
        else:
            xyz_string.append(line)
            lines_left -= 1
            if not lines_left:
                xyz_strings.append('\n'.join(xyz_string))
                new_string = True
    return xyz_strings

def distance_matrix(x, y):
    x2 = (x**2).sum(axis=-1)
    y2 = (y**2).sum(axis=-1)
    xy = np.matmul(x, np.transpose(y, (0, 2, 1)))
    return np.sqrt(np.expand_dims(x2, -1) - 2*xy + np.expand_dims(y2, -2))

def xyz2re(xyz, eq):
    n=eq.shape[0]
    eq = np.expand_dims(eq, 0)
    return distance_matrix(eq, eq)[0,np.triu_indices(n,1)[0], np.triu_indices(n,1)[1]]/distance_matrix(xyz, xyz)[:, np.triu_indices(n,1)[0], np.triu_indices(n,1)[1]]

def re2xyz(re, eq):
    pass