import sys
from rdkit import Chem
from meeko import MoleculePreparation

pdb_file = "ligand.pdb"
if len(sys.argv) > 1:
    pdb_file = sys.argv[1]

# Step 1: 从 PDB 读取分子
mol = Chem.MolFromPDBFile(pdb_file, removeHs=False)

# Step 2: 显式加氢（Meeko 强制要求）
mol = Chem.AddHs(mol, addCoords=True)

# Step 3: Meeko 处理
prep = MoleculePreparation()
prep.prepare(mol)

pdbqt_string = prep.write_pdbqt_string()
pdbqt_string = pdbqt_string.replace('UNL', 'UNK')

# Step 4: 输出结果
with open(pdb_file.split('.')[0] + ".pdbqt", "w") as f:
    f.write(pdbqt_string)
