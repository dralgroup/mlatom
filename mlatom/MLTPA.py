#!/usr/bin/env python
# coding: utf-8


'''
  !---------------------------------------------------------------------------! 
  ! mltpa: Machine learning prediction of the Two-photon absorption           ! 
  ! Implementations by: Yuming Su, Yiheng Dai, Yangtao Chen, Fuchun Ge        ! 
  !---------------------------------------------------------------------------! 
'''

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from typing import *
import joblib
import time
import os
import pandas as pd
import xgboost as xgb

from .args_class import ArgsBase 
mlatomdir=os.path.dirname(__file__)

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.args2pass = [] 
        self.add_default_dict_args([ 

            ],
            ""
        )

        self.add_dict_args({
            # 'mlmodelin':mlatomdir+'/mltpa/mltpa2022_2.pkl',
            'mlmodelin':'',
            'SMILESfile':'',
            'auxfile':'',
        })

    def parse(self,argsraw):
        self.parse_input_content(argsraw)
        if not self.mlmodelin:
            from sklearn import __version__ as Vsklear
            from xgboost import __version__ as Vxgboost
            if int(Vxgboost.split('.')[0])>1 or (int(Vxgboost.split('.')[0])==1 and int(Vxgboost.split('.')[1])>6):
                self.mlmodelin=mlatomdir+'/mltpa/mltpa2022_2_new_.pkl'
            else:
                self.mlmodelin=mlatomdir+'/mltpa/mltpa2022_2.pkl'
        self.args2pass = self.args_string_list(['',None])



class MLTPA(object):
    def __init__(self,
                argsMLTPA
                 ) -> None:

        args=Args()
        args.parse(argsMLTPA)
        smiles_list=args.SMILESfile
        model=args.mlmodelin
        auxfile=args.auxfile
        """
        Two Photon Absorption Cross-Section Predictor Class

        Parameters
        ----------
        smiles_list
            A file contains list with SMILES of one or many molecules
        model
            Model used to predict TPA cross-section. Default: '/home/mlatom/models/MLTPA/mltpa2022.pkl'
        auxfile 
            An auxiliary file including the infomation of wavelength and solvent(Et30). The format should be 
            'wavelength_lowbound,wavelength_upbound,Et30'. If the auxiliary file does not exist, then the default value
            of Et30 will be 33.9(toluene) and the whole spectra between 600-1100nm will be output.The length should be
            corresponding to the smiles_list.
        """
        self.smiles_list = []
        self.YestFolder = 'mltpa'+ time.strftime("%Y%m%d_%H%M%S", time.localtime())
        with open(smiles_list, 'r') as f:
            for line in f.readlines():
                self.smiles_list.append(line.strip('\n'))
        if auxfile:
            self.wavelength = []
            self.Et30 = []
            with open(auxfile, 'r') as f:
                for line in f.readlines():
                    info = list(map(float, line.strip('\n').split(',')))
                    self.wavelength.append(info[:2])
                    self.Et30.append(info[2])
            if len(self.wavelength)!=len(self.smiles_list):
                 raise ValueError("The length of smiles_list and the length of auxfile are not exactly the same.")
        else:
            self.wavelength = None
            self.Et30 = 33.9
        with open(model, 'rb') as m:
            self.model = joblib.load(m)
        self.length = len(self.smiles_list)

    @staticmethod
    def find_conj(smiles: str):
        """
        Find the conjugate part of the given molecule

        Parameters
        ----------
        smiles
            The SMILES of the molecule
        Returns
        -------
        ring_list:
            A list containing all ring
        f_a
            A list containing atom index of the conjugate part
        """
        mol = Chem.MolFromSmiles(smiles)
        a_m = Chem.rdmolops.GetAdjacencyMatrix(mol)
        patt_list_d = ['C=C', 'C#C', 'C#N', 'C=O', 'C=S', 'C=N', 'N=N', '[N+]([O-])=O']
        patt_list_m = ['N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P']
        ring_list = []
        f_a = []
        patt = Chem.MolFromSmarts('c')  # Define the aromatic carbon atom
        atomids = mol.GetSubstructMatches(patt)  # Find all the aromatic carbon atoms in the molecule
        atoms = mol.GetAtoms()  # get all the atoms in the molecule
        temp_list = []

        def find_ring(atom_id, found_atoms):
            nonlocal a_m, ring_list, f_a, atoms, temp_list
            flag = False
            c_list = np.argwhere(
                a_m[atom_id] == 1).flatten().tolist()  # Find the atom next to the atom with atom_id number
            for atom in c_list:
                if atom not in f_a:
                    a = atoms[atom]
                    if a.IsInRing() and str(a.GetHybridization()) != 'SP3':
                        found_atoms.append(atom)
                        f_a.append(atom)
                        find_ring(atom, found_atoms)
                        flag = True
            if not flag:
                temp_list.append(found_atoms)

        for atom in atomids:
            a = atom[0]
            if a not in f_a:
                find_ring(a, [])
            if len(temp_list) > 0:
                max_ring = temp_list[0]
                for l in temp_list:
                    if len(l) > len(max_ring):
                        max_ring = l
                ring_list.append(max_ring)
            temp_list = []
        for patt in patt_list_d:
            f = Chem.MolFromSmarts(patt)
            atomids = mol.GetSubstructMatches(f)
            if len(atomids) > 0:
                for pair in atomids:
                    n_l = []
                    flag_f = False
                    for a in pair:
                        if a in f_a:
                            flag_f = True
                            break
                        neighbors = atoms[a].GetNeighbors()
                        for na in neighbors:
                            n_l.append(na.GetIdx())
                    if flag_f:
                        continue
                    temp = []
                    temp_r_id = []
                    # So here we have the adjacent atoms of the double and triple bonds
                    for n in n_l:
                        if atoms[n].GetAtomicNum() in [6, 7, 8]:
                            for i in range(len(ring_list)):
                                ring = ring_list[i]
                                if n in ring:
                                    temp.append(ring)
                                    temp_r_id.append(i)
                    if len(temp) == 1:
                        ring_list[temp_r_id[0]].append(pair[0])
                        ring_list[temp_r_id[0]].append(pair[1])
                        f_a.append(pair[0])
                        f_a.append(pair[1])
                    else:
                        # Merging multiple lists
                        t_r = []
                        for r in temp:
                            t_r += r
                        # Plus double and triple bonds
                        t_r.append(pair[0])
                        t_r.append(pair[1])
                        # Get rid of the old ring
                        temp_r_id.sort()
                        temp_r_id = np.unique(temp_r_id)
                        for i in reversed(temp_r_id):
                            del ring_list[i]
                        # Plus a new ring
                        ring_list.append(t_r)
                        f_a.append(pair[0])
                        f_a.append(pair[1])
        for patt in patt_list_m:
            f = Chem.MolFromSmarts(patt)
            atomids = mol.GetSubstructMatches(f)
            if len(atomids) > 0:
                for atom in atomids:
                    a = atom[0]
                    if a not in f_a:
                        neighbors = atoms[a].GetNeighbors()
                        n_l = []
                        for na in neighbors:
                            n_l.append(na.GetIdx())
                        temp = []
                        temp_r_id = []
                        # So over here, we find the heteroatom's neighbor
                        for n in n_l:
                            for i in range(len(ring_list)):
                                ring = ring_list[i]
                                if (n in ring) and (i not in temp_r_id):
                                    temp.append(ring)
                                    temp_r_id.append(i)
                        if len(temp) == 1:
                            ring_list[temp_r_id[0]].append(a)
                            f_a.append(a)
                        else:
                            # Merging multiple lists
                            t_r = []
                            for r in temp:
                                t_r += r
                            # Plus heteroatoms
                            t_r.append(a)
                            # Get rid of the old ring
                            temp_r_id.sort()
                            for i in reversed(temp_r_id):
                                del ring_list[i]
                            # Plus a new ring
                            if len(t_r) > 1:
                                ring_list.append(t_r)
                                f_a.append(a)
        for i in range(len(atoms)):
            if i not in f_a:
                aa = atoms[i]
                if aa.GetSymbol() != 'C' or str(aa.GetHybridization()) != 'SP2':
                    continue
                aa_n = aa.GetNeighbors()
                flag = False
                for aaa in aa_n:
                    if aaa.GetIdx() in f_a:
                        flag = True
                        break
                if flag:
                    a = i
                    neighbors = atoms[a].GetNeighbors()
                    n_l = []
                    for na in neighbors:
                        n_l.append(na.GetIdx())
                    temp = []
                    temp_r_id = []
                    # So over here, we find the heteroatom's neighbor
                    for n in n_l:
                        for i in range(len(ring_list)):
                            ring = ring_list[i]
                            if (n in ring) and (i not in temp_r_id):
                                temp.append(ring)
                                temp_r_id.append(i)
                    if len(temp) == 1:
                        ring_list[temp_r_id[0]].append(a)
                        f_a.append(a)
                    else:
                        # Merging multiple lists
                        t_r = []
                        for r in temp:
                            t_r += r
                        # Plus heteroatoms
                        t_r.append(a)
                        # Get rid of the old ring
                        temp_r_id.sort()
                        for i in reversed(temp_r_id):
                            del ring_list[i]
                        # Plus a new ring
                        if len(t_r) > 1:
                            ring_list.append(t_r)
                            f_a.append(a)
        # Finally check whether the conjugate structure is connected
        if len(ring_list) > 1:
            temp_count = 0
            flag = True
            while flag:
                t_temp = int(len(ring_list) * (len(ring_list) - 1) / 2)
                temp = 0
                break_flag = False
                for i in range(len(ring_list) - 1):
                    for j in range(len(ring_list) - i - 1):
                        r_1 = ring_list[i]
                        r_2 = ring_list[i + j + 1]
                        if np.sum(a_m[r_1, :][:, r_2]) == 0:
                            temp += 1
                        else:
                            # need to merge
                            for k in r_2:
                                ring_list[i].append(k)
                            ring_list[i] = np.unique(ring_list[i])
                            del ring_list[i + j + 1]
                            break_flag = True
                            break
                    if break_flag:
                        break
                if temp == t_temp:
                    flag = False
        for i in range(len(ring_list)):
            ring_list[i] = np.unique(ring_list[i]).flatten().tolist()
        return ring_list, f_a

    def feature_generation(self, smiles, et30, wavelength) -> np.ndarray:
        """
        feature vector generation
        Parameters
        ----------
        smiles
            SMILES of the given molecule
        et30
            A descriptor to measure solvent polarity
        wavelength
            Wavelength in nanometer
        Returns
        -------
        feature_vector
            features of the molecule in the order of
            [Max Conju-Distance (MFF-Conju),
            Conju-PEOE-Charge-Maximum (MFF-Conju),
            Conju-Atomic-LogP-Minimum (MFF-Conju),
            Conju-Atomic-MR-Maximum (MFF-Conju),
            ET(30) (Solvent),
            Wavelength (Exp nm)
            ]
        """
        mff_title = []
        with open(mlatomdir+'/mltpa/mff.txt', 'r') as f:
            for line in f.readlines():
                mff_title.append(line.strip('\n'))
        feature_vector = np.zeros(6)
        mol = Chem.MolFromSmiles(smiles)
        d_m = Chem.rdmolops.GetDistanceMatrix(mol)
        ring_list, f_a = self.find_conj(smiles)
        lstmax = sorted(ring_list, key=len)[-1]

        # calculate max conju-distance
        conju_max_dis = []
        for r in ring_list:
            conju_max_dis.append(np.max(d_m[r, :][:, r]))
        feature_vector[0] = max(conju_max_dis)

        # calculate PEOE-charge
        AllChem.ComputeGasteigerCharges(mol, nIter=25)
        peoe_charge = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge') for i in range(mol.GetNumAtoms())]

        # calculate logp and mr
        contribs = rdMolDescriptors._CalcCrippenContribs(mol)
        logp = [contribs[i][0] for i in range(len(contribs))]
        mr = [contribs[i][1] for i in range(len(contribs))]
        atom_props = [peoe_charge, logp, mr]

        for _ in range(len(atom_props)):
            for r in [lstmax]:
                atom_props_list = atom_props[_]
                frag_x = []
                frag_atom_id = []
                for i in range(len(mff_title)):
                    patt = mff_title[i]
                    f = Chem.MolFromSmarts(patt)
                    atomids = mol.GetSubstructMatches(f)
                    if len(atomids) > 0:
                        for j in range(len(atomids)):
                            peoe_flag = True
                            for k in atomids[j]:
                                if k not in r:
                                    peoe_flag = False
                                    break
                            if peoe_flag:
                                frag_atom_id.append(atomids[j])
                                x_temp = 0
                                for k in atomids[j]:
                                    x_temp += atom_props_list[k]
                                frag_x.append(x_temp)
            if len(frag_x) == 0:
                frag_x = [0]
                frag_atom_id = [()]
            if _ == 0 or _ == 2:
                feature_vector[_ + 1] = (max(frag_x))
            else:
                feature_vector[_ + 1] = (min(frag_x))

        feature_vector[4] = et30
        feature_vector[5] = wavelength
        return feature_vector

    def predict(self):
        c_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_path = '{}'.format(self.YestFolder)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for i in range(self.length):
            feature_mat = []
            if self.wavelength is None:
                wavelength = np.array(np.arange(600, 1100, 10))
                feature_mat.append(self.feature_generation(self.smiles_list[i], self.Et30, wavelength[0]))
                feature_mat = np.array(feature_mat * len(wavelength))
                
                feature_mat[:, -1] = wavelength
                y_hat = self.model.predict(feature_mat)
                y_hat= np.exp(y_hat)
                dataframe = pd.DataFrame({'wavelength':wavelength,'predicted_sigma (GM)':y_hat})
            else:              
                wavelength = np.array(np.arange(self.wavelength[i][0],self.wavelength[i][1],10))
                if len(wavelength)>1:
                    feature_mat.append(self.feature_generation(self.smiles_list[i], self.Et30[i], wavelength[0]))
                    feature_mat = np.array(feature_mat * len(wavelength))
                    feature_mat[:, -1] = wavelength
                    y_hat = self.model.predict(xgb.DMatrix(feature_mat))
                    y_hat= np.exp(y_hat)
                    dataframe = pd.DataFrame({'wavelength':wavelength,'predicted_sigma (GM)':y_hat})
                else:
                    wavelength=self.wavelength[i][0]
                    feature_mat.append(self.feature_generation(self.smiles_list[i], self.Et30[i], wavelength))
                    feature_mat = np.array(feature_mat)
                    feature_mat[:, -1] = wavelength
                    y_hat = self.model.predict(xgb.DMatrix(feature_mat))
                    y_hat= np.exp(y_hat)
                    dataframe = pd.DataFrame({'wavelength':[wavelength],'predicted_sigma (GM)':y_hat})
            dataframe.to_csv(os.path.join(out_path, 'tpa{}.txt'.format(i+1)),sep=',',index=0)
            print('estimation saved in '+os.path.join(out_path, 'tpa{}.txt'.format(i+1)))

if __name__=='__main__':
    aaa = MLTPA('./Smiles_856-Copy1.csv', './mltpa2022_2.pkl', auxfile='aux.txt')
    aaa.predict()
    ccc = MLTPA('./Smiles_856-Copy1.csv', './mltpa2022.pkl')
    ccc.predict()





