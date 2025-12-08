import sys
import os

out_dir = sys.argv[1]

pdb_dir = out_dir + '-Pdb'
if not os.path.exists(pdb_dir):
    os.mkdir(pdb_dir)

tem_dir = out_dir.replace('-Out', '')
predict_dir = f'{out_dir}/boltz_results_{tem_dir}/predictions'

for file in os.listdir(predict_dir):
    file_path = os.path.join(predict_dir, file)
    pdb_name_1 = file + '_model_0.pdb'
    pdb_name_2 = file + '_model.pdb'
    if os.path.isdir(file_path) and os.path.exists(f'{file_path}/{pdb_name_1}'):
        with open(f'{pdb_dir}/{pdb_name_2}', 'w') as out:
            with open(f'{file_path}/{pdb_name_1}', 'r') as f:
                for line in f:
                    if (line[0:4] == 'ATOM' or line[0:4] == 'HETA') and line[21] == 'B':
                        line = line.replace(' LIG ', ' HEM ')
                    if (line[0:4] == 'ATOM' or line[0:4] == 'HETA') and line[21] == 'C':
                        line = line.replace(' LIG ', ' UNK ')

                    out.write(line)
