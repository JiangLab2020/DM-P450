import os
import re
import sys

TarPDB_Path = sys.argv[1]

TarSDF = sys.argv[2]

active_C = sys.argv[3]

pair_name = TarPDB_Path.split('/')[-1] + '_docking'         #   result docking dirctory

if not os.path.exists(pair_name):
    os.mkdir(pair_name)

out = open('no_docking_pdb.txt', 'w')

for pdb_file in os.listdir(TarPDB_Path):    
    TarPDB = os.path.join(TarPDB_Path, pdb_file)

    dock_path = os.getcwd()

    print(f'python docking_P450_4_Contain_Substrate.py {TarPDB} {TarSDF} {active_C}')
    os.system(f'python docking_P450_4_Contain_Substrate.py {TarPDB} {TarSDF} "{active_C}"')

    protein_name = TarPDB.split('/')[-1].split('.')[0]
    sdf_name = TarSDF.split('/')[-1].split('.')[0]

    out_path = protein_name + '_' + sdf_name + '_dock'

#   if not os.path.exists(out_path):
#       os.system(f'python docking_P450_4_Contain_Substrate_not_use.py {TarPDB} {TarSDF} "{active_C}" N7_trim.pdb')

    if os.path.exists(out_path):
        for file in os.listdir(out_path):
            if re.search('_docked_\d+_\d+_addH.pdb', file):
                os.system(f'cp {dock_path}/{out_path}/{file} {pair_name}/')
        os.system('rm -rf %s' % out_path)
    else:
        out.write(pdb_file + '\t' + 'no best score' + '\n')
    
#   break


out.close()
