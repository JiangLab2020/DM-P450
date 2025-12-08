import os
from cleanfasta import clean_fasta


def docking4infer(input_fa: str, substrate: str):
    os.chdir("P450_docking")
    os.system("rm -rf *-msa*")
    os.system("rm -rf *-Yaml*")
    os.system("sh clean.sh")
    clean_fasta(input_fasta=input_fa, output_fasta=input_fa)
    # step 1: modeling
    print(f"Running docking for {input_fa} with substrate {substrate}")
    os.system(
        f"python modeling_P450_by_boltz2.py {input_fa} P450_db/P450_db {substrate}"
    )
    # step 2: docking
    print("\n\n================= Attention ================")
    print(
        "please check the modeling results and input the reaction site residue number using Capital letter:(like C21)"
    )
    print(
        "you can find the residue number in the output pdb file(path ./P450_docking/*.pdbqt)"
    )
    print("================= Attention ================\n\n")
    residue_number = input("Input the reaction site residue number:")
    os.system(
        f"python docking_P450_4_Contain_Substrate_dir_one_Ligand.py {input_fa.split('.')[0]}-Yaml-Out-Pdb {substrate.split('.')[0]}.pdbqt {residue_number}"
    )
    # step 3: cleanup
    os.system("sh clean.sh")
    os.chdir("..")
