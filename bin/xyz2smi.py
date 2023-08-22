#!/usr/bin/env python
# coding: utf-8

# #
# #'''
# #  !---------------------------------------------------------------------------! 
# #  ! xyz2smi: Machine learning preparation to change xyz format to smiles file ! 
# #  ! Implementations by: Yuming Su                                             !
# #  ! data:20221124                                                             !
# #  !---------------------------------------------------------------------------! 
# #'''

import os
from openbabel import openbabel

class Args(ArgsBase):
    def __init__(self):
        super().__init__()
        self.args2pass = [] 
        self.add_default_dict_args([ 

            ],
            ""
        )


    def parse(self,argsraw):
        self.parse_input_content(argsraw)
        self.args2pass = self.args_string_list(['',None])

class xyz2smi(object):
    def __init__(self,
                argsxyz2smi
                )-> None:   
        """
        file_PATH is the path of .xyz files
        output files named "Smiles.txt" will generated in the current path. 
        """
        args=Args()
        args.parse(argsxyz2smi)
        self.file_Path=args.file_Path
    def genxyz2smi(self):
        input_format='xyz'
        output_format='can'
        nowpath=os.getcwd ()
        tqdm=os.listdir(self.file_Path)
        for i in range(0,len(tqdm)):
            inputfile = os.path.join(self.file_Path,tqdm[i])
            conv=openbabel.OBConversion()
            conv.OpenInAndOutFiles(inputfile,inputfile+'_'+'.smi')
            conv.SetInAndOutFormats(input_format,output_format)
            conv.Convert()
            conv.CloseOutFile()
        datanames = os.listdir(self.file_Path)
        smilelist=[]
        for dataname in datanames:
            if os.path.splitext(dataname)[1] == '.smi':
                smilelist.append(dataname)
        f2=open(os.path.join(nowpath,'Smilesfile.txt'),'a', encoding='UTF-8')
        for smile in smilelist:
            smileinputfile = os.path.join(self.file_Path,smile)
            with open(smileinputfile, 'r') as f:
                for line in f.readlines():
                    info = line.split('xyz')[0]
                    print(info)
                    info=info+'\n'
                    f2.write(info)
        f2.close()    
        for root , dirs, files in os.walk(self.file_Path):
            for name in files:
              if name.endswith(".smi"):  
                os.remove(os.path.join(root, name))
                print ("Delete File: " + os.path.join(root, name))
