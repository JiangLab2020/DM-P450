#!/home/hanxudong/anaconda3/envs/bio/bin/python

###   combining a pdb without ligand with a ligand pdb file

import re
import sys

protein_file = sys.argv[1]
substrate_file = sys.argv[2]

protein_name = protein_file.split('.')[0]                              ###   output to the path of pdb file without ligand
substrate_name = substrate_file.split('/')[-1].split('.')[0]

combine_file = protein_name + '_' + substrate_name + '.pdb'

def replace_pdb_line(line, line_type, new_value):
    beg_loc = 0
    end_loc = 0
    if line_type == 'serial':
        beg_loc = 6
        end_loc = 10
    elif line_type == 'res_seq':
        beg_loc = 22
        end_loc = 25
    elif line_type == 'chain_id':
        beg_loc = 21
        end_loc = 21
        
    tmp=list(line)
    j = 0
    for i in range(end_loc,beg_loc-1,-1):        
        if j < len(str(new_value)):
            tmp[i] = str(new_value)[len(str(new_value))-j-1]
            j+=1
        else:
            tmp[i] = ' '

    return ''.join(tmp)


with open(combine_file,'w') as out:
    res_seq = 0
    serial  = 0
    with open(protein_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if re.match(r'ATOM',line) or re.match(r'HETATM',line):
                serial  = int(line[6:11].replace(' ',''))
                res_seq = int(line[22:26].replace(' ',''))
                if re.match(r'HETATM \d+.*CYM A',line):
                    line = line.replace('HETATM ','ATOM   ')

                out.write(line)

            if re.match(r'CONECT',line):
                out.write(line)

    serial_step = serial

    serial+=1
    res_seq+=1
#    print(res_seq)

    space_5 = '     '
    
    with open(substrate_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if re.match(r'ATOM',line) or re.match(r'HETATM',line):
                line = replace_pdb_line(line, 'chain_id', 'A')
                line = replace_pdb_line(line, 'res_seq', res_seq)
                line = replace_pdb_line(line, 'serial', serial)
                serial+=1

                out.write(line)

            if re.match(r'CONECT',line):
                a1 = re.split(r'\s+',line.strip('\n'))
                new_line = 'CONECT'
                for i in range(1,len(a1)):
                    new_loc = int(a1[i]) + serial_step
                    new_line += space_5[0:(5-len(str(new_loc)))] + str(new_loc)

                out.write(new_line + '\n')

