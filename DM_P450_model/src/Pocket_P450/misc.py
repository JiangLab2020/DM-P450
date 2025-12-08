import os
import re
from collections import defaultdict
import numpy as np
import torch
import dgl
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import toml

cfg = toml.load("DM_P450_model/config/config.toml")

# --- 新增：蛋白质残基的物理化学性质 ---
# 包含疏水性、体积和极性等
RESIDUE_FEATURES = {
    "ALA": [1.8, 89.1, 0.25],  # Hydrophobicity, Volume, Polarity
    "ARG": [-4.5, 173.2, 0.81],
    "ASN": [-3.5, 114.1, 0.73],
    "ASP": [-3.5, 111.1, 0.77],
    "CYS": [2.5, 108.5, 0.24],
    "GLN": [-3.5, 143.2, 0.67],
    "GLU": [-3.5, 138.1, 0.74],
    "GLY": [-0.4, 60.1, 0.15],
    "HIS": [-3.2, 153.2, 0.5],
    "ILE": [4.5, 166.7, 0.13],
    "LEU": [3.8, 166.7, 0.13],
    "LYS": [-3.9, 168.2, 0.99],
    "MET": [1.9, 162.9, 0.22],
    "PHE": [2.8, 189.9, 0.12],
    "PRO": [-1.6, 112.7, 0.14],
    "SER": [-0.8, 89.0, 0.46],
    "THR": [-0.7, 116.1, 0.45],
    "TRP": [-0.9, 227.8, 0.25],
    "TYR": [-1.3, 193.6, 0.42],
    "VAL": [4.2, 140.0, 0.14],
    "HEM": [0.0, 0.0, 0.0],  # Placeholder for Heme
    "HEC": [0.0, 0.0, 0.0],  # Placeholder for Heme C
}


def get_ligand_atom_features(atom):
    """为单个配体原子生成丰富的化学特征向量。"""
    # 特征包括: 原子类型 (one-hot), 杂化类型 (one-hot), 度, 形式电荷, 是否在环中, 是否是芳香原子
    atomic_symbol = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H", "FE", "OTHER"]
    hybridization = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
    ]

    symbol_one_hot = [0] * len(atomic_symbol)
    try:
        symbol_one_hot[atomic_symbol.index(atom.GetSymbol())] = 1
    except ValueError:
        symbol_one_hot[-1] = 1  # 未知原子类型归为OTHER

    hybrid_one_hot = [0] * len(hybridization)
    try:
        hybrid_one_hot[hybridization.index(atom.GetHybridization())] = 1
    except ValueError:
        pass  # 如果杂化类型未知，则全为0

    features = symbol_one_hot + hybrid_one_hot
    features.append(atom.GetDegree())
    features.append(atom.GetFormalCharge())
    features.append(int(atom.IsInRing()))
    features.append(int(atom.GetIsAromatic()))
    return features


def parse_finger_file(filepath):
    """解析P450功能域文件。"""
    finger_map = defaultdict(list)
    try:
        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 4:
                    name, start, end, label = parts
                    protein_id = name.split("_")[0]
                    finger_map[protein_id].append(
                        {"start": int(start), "end": int(end), "label": label}
                    )
    except FileNotFoundError:
        print(f"错误: 功能域文件未找到 at {filepath}")
    return finger_map


def get_residue_label(res_id, protein_id, finger_map):
    """根据残基ID和功能域映射，获取其功能域标签。"""
    if protein_id in finger_map:
        for region in finger_map[protein_id]:
            if region["start"] <= res_id <= region["end"]:
                return region["label"]
    return "other"


def parse_pdb_and_create_ligand_mol(filepath):
    """
    解析PDB文件，分离蛋白质和配体，并使用RDKit为配体创建一个Mol对象。
    这一步对于后续的特征提取至关重要。
    """
    protein_residues = defaultdict(list)
    ligand_lines = []
    conect_records = []

    protein_id_match = re.search(r"(CYP\d+[A-Z]*\d*)", os.path.basename(filepath))
    protein_id = protein_id_match.group(0) if protein_id_match else None

    WATER_RES = {"HOH", "WAT"}
    LIGAND_RES = {"UNK"}  # 将UNK残基明确视为配体

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                record_type = line[0:6].strip()
                res_name = line[17:20].strip()

                if res_name in WATER_RES:
                    continue  # 忽略水分子

                try:
                    coords = np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    )
                    if np.all(coords == 0.0):
                        continue  # 忽略坐标全为0的原子

                    atom_info = {
                        "atom_serial": int(line[6:11]),
                        "atom_name": line[12:16].strip(),
                        "coords": coords,
                        "element": line[76:78].strip() or line[12:14].strip()[0],
                    }
                except (ValueError, IndexError):
                    continue

                is_ligand = False
                if res_name in LIGAND_RES:
                    is_ligand = True
                elif record_type == "ATOM":  # 蛋白质原子
                    res_seq = int(line[22:26])
                    atom_info["res_name"] = res_name
                    protein_residues[res_seq].append(atom_info)
                elif record_type == "HETATM":  # 其他HETATM
                    # 将血红素辅因子视为蛋白质环境的一部分
                    if "HEM" in res_name or "HEC" in res_name:
                        res_seq = int(line[22:26])
                        atom_info["res_name"] = res_name
                        protein_residues[res_seq].append(atom_info)
                    else:  # 其他所有HETATM都视为配体
                        is_ligand = True

                if is_ligand:
                    ligand_lines.append(line)

            elif line.startswith("CONECT"):
                conect_records.append(line)

    # 将配体相关的行（HETATM和CONECT）构建成PDB格式的字符串块
    ligand_pdb_block = ""
    if ligand_lines:
        ligand_pdb_block = "".join(ligand_lines) + "".join(conect_records) + "END"

    # 使用RDKit从PDB块创建分子对象
    mol = None
    if ligand_pdb_block:
        mol = Chem.MolFromPDBBlock(ligand_pdb_block, sanitize=True, removeHs=False)

    return protein_residues, mol, protein_id


def create_graph(
    protein_residues,
    ligand_mol,
    interacting_res_ids,
    protein_id,
    finger_map,
    vocab_maps,
):
    """根据给定的数据创建一个DGL异构图，包含丰富的节点和边特征。"""
    res_vocab, finger_label_vocab = vocab_maps["res"], vocab_maps["finger"]
    res_to_idx, finger_to_idx = vocab_maps["res_map"], vocab_maps["finger_map"]

    # 确保配体和相互作用残基都存在
    if not ligand_mol or ligand_mol.GetNumAtoms() == 0 or not interacting_res_ids:
        return None

    num_ligand_atoms = ligand_mol.GetNumAtoms()
    num_protein_res = len(interacting_res_ids)

    # --- 1. 创建图结构 ---
    # 从RDKit分子对象获取邻接矩阵，以确定化学键
    ligand_adj = GetAdjacencyMatrix(ligand_mol, useBO=False)
    ligand_bonds = np.array(np.nonzero(ligand_adj)).T

    g = dgl.heterograph(
        {
            ("ligand", "bonded_to", "ligand"): (ligand_bonds[:, 0], ligand_bonds[:, 1]),
            ("ligand", "interacts_with", "protein"): [],
            ("protein", "interacts_with", "ligand"): [],
        },
        num_nodes_dict={"ligand": num_ligand_atoms, "protein": num_protein_res},
    )

    # --- 2. 添加相互作用边和边特征 ---
    interaction_edges_src, interaction_edges_dst = [], []
    interaction_distances = []
    res_id_to_node_idx = {res_id: i for i, res_id in enumerate(interacting_res_ids)}
    ligand_coords = ligand_mol.GetConformer(0).GetPositions()

    for ligand_idx, l_coords in enumerate(ligand_coords):
        for res_id in interacting_res_ids:
            if res_id not in protein_residues:
                continue
            res_node_idx = res_id_to_node_idx[res_id]
            for protein_atom in protein_residues[res_id]:
                dist = np.linalg.norm(protein_atom["coords"] - l_coords)
                if dist < cfg["train_config"]["INTERACTION_DISTANCE"]:
                    interaction_edges_src.append(ligand_idx)
                    interaction_edges_dst.append(res_node_idx)
                    interaction_distances.append(dist)
                    break  # 每个配体原子与残基只添加一次相互作用

    if not interaction_edges_src:
        return None  # 如果没有相互作用，则不创建图

    g.add_edges(
        interaction_edges_src,
        interaction_edges_dst,
        etype=("ligand", "interacts_with", "protein"),
    )
    g.add_edges(
        interaction_edges_dst,
        interaction_edges_src,
        etype=("protein", "interacts_with", "ligand"),
    )

    # 将距离作为边特征添加到图中
    g.edges[("ligand", "interacts_with", "protein")].data["distance"] = torch.tensor(
        interaction_distances, dtype=torch.float32
    ).unsqueeze(1)
    g.edges[("protein", "interacts_with", "ligand")].data["distance"] = torch.tensor(
        interaction_distances, dtype=torch.float32
    ).unsqueeze(1)

    # --- 3. 添加节点特征 ---
    # 配体节点特征
    ligand_feats = [get_ligand_atom_features(atom) for atom in ligand_mol.GetAtoms()]
    g.nodes["ligand"].data["h"] = torch.tensor(ligand_feats, dtype=torch.float32)

    # 蛋白质节点特征
    protein_feats = []
    res_feat_dim = len(res_vocab)
    finger_feat_dim = len(finger_label_vocab)

    for res_id in interacting_res_ids:
        if res_id not in protein_residues:
            continue
        res_info = protein_residues[res_id][0]
        res_name = res_info["res_name"]

        # 氨基酸类型的One-hot编码
        res_one_hot = torch.zeros(res_feat_dim)
        if res_name in res_to_idx:
            res_one_hot[res_to_idx[res_name]] = 1

        # 功能域标签的One-hot编码
        finger_label = get_residue_label(res_id, protein_id, finger_map)
        finger_one_hot = torch.zeros(finger_feat_dim)
        if finger_label in finger_to_idx:
            finger_one_hot[finger_to_idx[finger_label]] = 1

        # 理化性质特征
        physchem_feats = torch.tensor(
            RESIDUE_FEATURES.get(res_name, [0, 0, 0]), dtype=torch.float32
        )

        # 拼接所有特征
        protein_feats.append(torch.cat([res_one_hot, finger_one_hot, physchem_feats]))

    g.nodes["protein"].data["h"] = torch.stack(protein_feats)

    return g
