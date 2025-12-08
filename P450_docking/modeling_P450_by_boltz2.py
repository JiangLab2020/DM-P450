import os
import sys

input_fasta = sys.argv[1]

target_db = sys.argv[2]

ligand_sdf = sys.argv[3]

target = input_fasta.split('.')[0]

ligand = ligand_sdf.split('.')[0]

print('step1: generate msa file\n')
os.system(f'python 0_run_mmseqs2.py {input_fasta} {target_db}')

print('step2: generate yaml file\n')
os.system(f'python 1_P450_Boltz_Modeling_Yaml_Pair.py {input_fasta} {ligand_sdf}')

print('step3: model by Boltz2\n')
os.system(f'python 2_P450_Boltz_Modeling_Run.py {target}-Yaml')

print('step4: output pdb and ligand pdbqt file\n')
os.system(f'python 3_P450_Boltz_Modeling_Out.py {target}-Yaml-Out')
os.system(f'python 4_P450_PDB_to_Pdbqt.py {target}-Yaml-Out-Pdb')

os.system(f'mv ligand.pdbqt {ligand}.pdbqt')


