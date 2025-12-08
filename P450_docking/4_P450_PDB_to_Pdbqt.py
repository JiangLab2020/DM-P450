import os
import sys
import re
from rdkit import Chem

pdb_dir = sys.argv[1]

out_protein = open('protein.pdb', 'w')
out_ligand  = open('ligand.pdb', 'w')


flag = 0
protein_atom_id = {}

TarPDB = os.path.join(pdb_dir, os.listdir(pdb_dir)[0])

with open(TarPDB, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line[0:4] == "ATOM" and (line[17:20] != 'HEM' and line[17:20] != 'LIG') and flag <= 1:
            flag = 1
            protein_atom_id[line[6:11].strip()] = 1
            out_protein.write(line)
        if (line[0:4] == "ATOM" or line[0:6] == 'HETATM') and (line[17:20] == 'HEM' or line[17:20] == 'LIG'):
            flag = 2
            protein_atom_id[line[6:11].strip()] = 2
            out_protein.write(line)
            
        if (line[0:4] == "ATOM" or line[0:6] == 'HETATM') and (line[17:20] != 'HEM' and line[17:20] != 'LIG') and flag > 1:
            flag = 3
            protein_atom_id[line[6:11].strip()] = 3
            out_ligand.write(line)

        if line[0:6] == 'CONECT':
            a1 = line.strip('\r\n').split()
            flag = protein_atom_id[a1[1]]
            for i in range(2,len(a1)):
                if protein_atom_id[a1[i]] != flag:
                    flag = 0
            if flag == 1:
                out_protein.write(line)
            if flag == 2:
                out_protein.write(line)
            if flag == 3:
                out_ligand.write(line)


out_protein.close()
out_ligand.close()


os.system('python PDB_to_Pdbqt.py ligand.pdb')

os.system('rm -f ligand.pdb protein.pdb')
