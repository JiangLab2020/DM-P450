#  需要对接的蛋白中包含底物, 需要对接的底物是单独的pdbqt格式

import os
import sys
import re
import numpy as np
from rdkit import Chem

TarPDB = sys.argv[1]              #   target P450 pdb including HEM

TarSDF = sys.argv[2]              #   target substrate needing to dock in TarPDB

sdf_name = TarSDF.split('/')[-1].split('.')[0]

product_sdf = ''
if len(sys.argv) > 3:
    product_sdf = sys.argv[3]


#######################################################################################################   selecting the GPU
GPU_Num = -1
All_GPUs = 1
os.system('nvidia-smi >nvidia.txt')
nvidia_dict = {}
with open('nvidia.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        if re.match(r'\|   (\d+?) .*GeForce',line):
            All_GPUs = re.match(r'\|   (\d+?) .*GeForce', line).group(1)
        if re.match(r'\|    (\d)   N\/A  N\/A',line):
            num = re.match(r'\|    (\d)   N\/A  N\/A',line).group(1)
            nvidia_dict[num] = 1
for i in range(int(All_GPUs)+1):
    if str(i) not in nvidia_dict:
        GPU_Num = i
        break
print('there are totally ' + str(int(All_GPUs)+1) + ' GPUs')
if i > -1:
    print('selecting the GPU:' + str(GPU_Num))
    os.environ['CUDA_VISIBLE_DEVICES']=str(GPU_Num)
else:
    print('on free GPU !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
#######################################################################################################   selecting the GPU



def FetchXYZ(line):
    x = line[30:38]
    y = line[38:46]
    z = line[46:54]

    x = float(x.replace(' ',''))
    y = float(y.replace(' ',''))
    z = float(z.replace(' ',''))

    return [x,y,z]

protein_name = TarPDB.split('/')[-1].split('.')[0]

print(protein_name)

out_path_1 = protein_name + '_' + sdf_name + '_dock'
if os.path.exists(out_path_1):
    os.system('rm -rf ' + out_path_1)

#######################################################################################################   Remove ligand from AF3 model

tem_protein = protein_name + '_protein.pdb'
tem_ligand  = protein_name + '_ligand.pdb'

out_protein = open(tem_protein, 'w')
out_ligand = open(tem_ligand, 'w')


flag = 0
protein_atom_id = {}
OxyLoc = ''

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
            if re.search('F[eE]', line[5:19]):
                OxyLoc = FetchXYZ(line)
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

print("Fe loc", OxyLoc)

out_protein.close()
out_ligand.close()
#######################################################################################################   Remove ligand from AF3 model

#############################################################################################
#os.system('Confor_search.py ' + TarSDF)
#os.system('split_sdf.py ' + TarSDF.split('.')[0] + '_cluster.sdf')

os.system(f'cp {TarSDF} ' + TarSDF.split('.')[0] + '_cluster_' + str(1) + '.' + TarSDF.split('.')[-1])

#############################################################################################



def ScreenRightdockings(dock_pdb_file, no_ligand_pdb, product_sdf, OxyLoc):
    good_num = 0
    best_num = 0

    cnn_score = 0

    with open(dock_pdb_file, "r") as f:
        lines = f.readlines()

    model_blocks = []
    current_block = []
    model_index = 0

    for line in lines:
        if line.startswith("MODEL"):
            current_block = [line]
        elif line.startswith("ENDMDL"):
            current_block.append(line)
            model_blocks.append(current_block)
            model_index += 1
        elif current_block:
            current_block.append(line)

    # 处理每个 MODEL 分子块
    for i, block in enumerate(model_blocks):
        model_filename = dock_pdb_file.split('.')[0] + f"_{i+1}.pdb"
        with open(model_filename, "w") as fout:
            fout.writelines(block)

        # 查找 C13 原子
        for line in block:
            if re.search(r'REMARK CNNscore (0\.\d\d\d)', line):
                cnn_score = re.search(r'REMARK CNNscore (0\.\d\d\d)', line).group(1)

            if line.startswith("ATOM") or line.startswith("HETATM"):
                atom_name = line[12:16].strip()
                active_C_gene_list = re.split(r'[;,]', product_sdf)
                if atom_name in active_C_gene_list:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    print(f"Model {i+1} - {product_sdf} coordinates: x={x:.3f}, y={y:.3f}, z={z:.3f}")
                    CarbonLoc = [x,y,z]
                    distance_C_O = np.sqrt(np.sum((CarbonLoc - np.asarray(OxyLoc))**2))

                    if distance_C_O < 5.1:                        
                        
                        addH_file_pdb = model_filename.split('.')[0] + '_addH' + '.pdb'

                        mol = Chem.MolFromPDBFile(model_filename, removeHs=False)
                        mol_with_H = Chem.AddHs(mol)
                        Chem.MolToPDBFile(mol_with_H, addH_file_pdb)

                        substrate_name = addH_file_pdb.split('/')[-1].split('.')[0]

                        os.system('python Combine_Protein_Substrate.py %s %s' % (no_ligand_pdb, addH_file_pdb))

                        print(f'atom_name is: {atom_name}; {substrate_name} is good docking; cnn score is: {cnn_score}')

                        good_num+=1
                        if good_num == 1:
                            best_num = substrate_name.split('_')[1]
                        if good_num == 1:
                            break
        if good_num > 0:
            break

    return int(best_num),float(cnn_score)


#######################################################################################################   docking


good_num = 0
    
min_best_num = 100
min_sdf_num = 0
min_best_score = 0.0
min_best_pdb_name = ''

for i in range(1, 5):
    print(i)

    out_path = protein_name + '_' + sdf_name + '_dock' + '_' + str(i)

    no_ligand_pdb = os.path.join(out_path, protein_name + '_' + sdf_name + '.pdb')
    ligand_file   = os.path.join(out_path, 'Tem_AGI.pdb')

    if os.path.exists(out_path):
        os.system('rm -rf ' + out_path)
    os.mkdir(out_path)

    os.system(f'cp {tem_protein} {no_ligand_pdb}')
    os.system(f'cp {tem_ligand} {ligand_file}')

    dock_pdb_file = os.path.join(out_path,'docked.pdb')

    
    TarSDF_i = TarSDF.split('.')[0] + '_cluster_' + str(i) + '.' + TarSDF.split('.')[-1]

    if os.path.exists(TarSDF_i):

        os.system(f'gnina -r {no_ligand_pdb} -l {TarSDF_i} --autobox_ligand {ligand_file} -o {dock_pdb_file} --num_modes 60 --temperature 200 >{out_path}/1_Score.txt 2>/dev/null')

        best_num,cnn_score = ScreenRightdockings(dock_pdb_file, no_ligand_pdb, product_sdf, OxyLoc)

        if best_num > 0:
            print(TarSDF_i + ": best num = " + str(best_num))

            best_score = cnn_score
      
            if best_score > min_best_score:
                min_best_score = best_score
                min_best_num = best_num
                min_sdf_num = i
                min_best_pdb_name = no_ligand_pdb.split('.')[0] + '_docked' + '_' + str(best_num) + '_addH.pdb'
                min_best_pdb_score_name = no_ligand_pdb.split('.')[0] + '_docked' + '_' + str(best_num) + '_' + str(int(best_score*100)) + '_addH.pdb'
                os.system(f'mv {min_best_pdb_name} {min_best_pdb_score_name}')

            print("the best score is: " + str(best_score))
                


for i in range(1, 10):
    out_path = protein_name + '_' + sdf_name + '_dock' + '_' + str(i)
    
    if i == min_sdf_num:
        os.system('mv %s %s' % (out_path, out_path_1))
    else:
        os.system('rm -rf %s' % out_path)

os.system('rm -f %s_cluster*' % TarSDF.split('.')[0])

os.system(f'rm -f {tem_protein}')
os.system(f'rm -f {tem_ligand}')

print("the best conformation is: " + str(min_sdf_num) + '; ' + "the best docking is: " + str(min_best_num) + '; ' + "the best score is: " + str(min_best_score))

if min_sdf_num > 0:
    with open(f'{out_path_1}/1_Score_best.txt', 'w') as out:
        out.write('best score\t' + str(min_best_score) + '\t' + str(int(min_best_score*100)) + '\n')
