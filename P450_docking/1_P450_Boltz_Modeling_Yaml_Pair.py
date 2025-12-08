#  python P450_AF3_Modeling_Json_Pair.py /hpcfs/fhome/cheng_j/Work/P450_Gene/P450_Function/FunP450_All_07_new.fasta /hpcfs/fhome/cheng_j/Work/P450_Gene/P450_Function/Family_Substrate_Class/SDF_All_Substrate

import re
import os
import sys
import json
from Bio import SeqIO
from rdkit import Chem

fasta_file = sys.argv[1]

ligand_dir = sys.argv[2]                 ###     要求  fasta序列id与sdf分子的id有一定的对应关系

msa_dir = fasta_file.split('/')[-1].split('.')[0] + '-msa_cleaned'
if len(sys.argv) > 3:
    msa_dir = sys.argv[3]

gene_ligand = {}

if os.path.isdir(ligand_dir):
    for file in os.listdir(ligand_dir):    
        if re.search('S\.sdf', file) or re.search('S1\.sdf', file):
            ligand_file = os.path.join(ligand_dir, file)
            mol = Chem.MolFromMolFile(ligand_file)
            ligand_file = Chem.MolToSmiles(mol)
            print(file)
            gene = file.split('_')[0]
            gene_ligand[gene] = ligand_file
else:
    ligand_file = sys.argv[2]
    mol = Chem.MolFromMolFile(ligand_file)
    ligand_file = Chem.MolToSmiles(mol)
    for seq in SeqIO.parse(fasta_file, 'fasta'):
        gene_name = seq.id
        gene_ligand[gene_name] = ligand_file


out_dir = fasta_file.split('/')[-1].split('.')[0] + '-Yaml'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)


for seq in SeqIO.parse(fasta_file, 'fasta'):
    gene_name = seq.id
    if os.path.isdir(ligand_dir):
        gene_name = seq.id.split('-')[0].split('_')[-1]

    my_seq = str(seq.seq).replace('*','')

    if gene_name in gene_ligand:

        ligand_file = gene_ligand[gene_name]
        
        output_json = os.path.join(out_dir, seq.id.replace('.','_') + '.yaml')
        with open(output_json, 'w') as json_file:
            json_file.write(f'version: 1\n')
            json_file.write(f'sequences:\n')
            json_file.write(f'    - protein:\n')
            json_file.write(f'        id: A\n')
            json_file.write(f'        sequence: {my_seq}\n')
            json_file.write(f'        msa: {msa_dir}/{seq.id}.a3m\n')
            json_file.write(f'\n')
            json_file.write(f'    - ligand:\n')
            json_file.write(f'        id: B\n')
            json_file.write(f'        smiles: \'C=CC1=C(C)C2C=c3c(C)c(CCC(=O)O)c4n3[Fe]35(O)N2C1=Cc1c(C)c(C=C)c(n13)C=C1C(C)=C(CCC(=O)O)C(C=4)N15\'\n')
            json_file.write(f'\n')
            json_file.write(f'    - ligand:\n')
            json_file.write(f'        id: C\n')
            json_file.write(f'        smiles: \'{ligand_file}\'\n')

