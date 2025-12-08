import os
import re
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import dgl
import toml
from unimol_tools import UniMolRepr
from rdkit import Chem
from pathlib import Path
from DM_P450_model.src.Pocket_P450.misc import (
    parse_finger_file,
    parse_pdb_and_create_ligand_mol,
    create_graph,
    RESIDUE_FEATURES,
)
from Bio import SeqIO
import os
# ============================ 路径与配置 ============================

# 读取配置
cfg_path = Path("DM_P450_model") / "config" / "config.toml"
cfg = toml.load(cfg_path)


# ============================ 主预处理函数 ============================


def preprocess_seq(input_fasta_path: str, sort_list: list):
    os.system(
        f"python DM_P450_model/src/tools/extract.py esm2_t33_650M_UR50D P450_docking/{input_fasta_path} {cfg['data_path']['Cache_PATH']} --repr_layers 33 --include mean"
    )


def get_seq_id(fasta_file_path: str) -> list:
    seq_ids = []
    # 读取 FASTA 文件并提取每个记录的 ID
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        seq_ids.append(record.id)
    return seq_ids


def preprocess_substrate(input_substrate_path: str):
    # 初始化结果字典
    sdf_smiles_dict = {}

    # 遍历所有 .sdf 文件
    input_substrate_path = f"P450_docking/{input_substrate_path}"
    suppl = Chem.SDMolSupplier(input_substrate_path)
    mol = suppl[0] if suppl and suppl[0] is not None else None
    if mol is None:
        raise ValueError(
            f"无法从 {input_substrate_path} 读取分子。请检查文件格式是否正确。"
        )

    smiles = Chem.MolToSmiles(mol)
    sdf_smiles_dict[Path(input_substrate_path).stem] = smiles
    # 初始化 UniMolRepr
    clf = UniMolRepr(
        data_type="molecule",
        remove_hs=True,
    )

    # 获取所有 key 和 smiles
    mol_ids = list(sdf_smiles_dict.keys())
    smiles_list = list(sdf_smiles_dict.values())
    # 批量编码
    unimol_batch_output = clf.get_repr(smiles_list, return_atomic_reprs=True)
    # 创建保存路径（如果不存在则创建）
    save_dir = Path(cfg["data_path"]["Seq_P450_PATH"]["infer_substrate_save_path"])
    save_dir.mkdir(parents=True, exist_ok=True)

    cls_vector = unimol_batch_output["cls_repr"][0]
    # cls_vector = unimol_batch_output['atomic_reprs'][i]
    save_path = save_dir / f"{mol_ids[0]}.pt"
    torch.save(torch.tensor(cls_vector), save_path)


def preprocess_graph(sort_list: list, fasta_file_path: str):
    all_graphs, all_labels, all_scores = [], [], []
    finger_map = parse_finger_file(
        Path(cfg["data_path"]["Pocket_P450_PATH"]["finger_file"])
    )
    # 定义用于编码的词汇表
    vocab_maps = {
        "res": list(RESIDUE_FEATURES.keys()),
        "finger": ["finger3", "finger4", "finger5", "palm1", "palm2", "other"],
    }
    vocab_maps["res_map"] = {name: i for i, name in enumerate(vocab_maps["res"])}
    vocab_maps["finger_map"] = {name: i for i, name in enumerate(vocab_maps["finger"])}

    for protein_id, ligand_id, scores, file in sort_list:
        protein_residues, ligand_mol, protein_id = parse_pdb_and_create_ligand_mol(
            f"P450_docking/{fasta_file_path.split('.')[0]}-Yaml-Out-Pdb_docking/{file}"
        )
        if not ligand_mol or not protein_residues:
            continue
        # 确定相互作用的残基
        ligand_coords = ligand_mol.GetConformer(0).GetPositions()
        interacting_res_ids_set = {
            res_id
            for res_id, atoms in protein_residues.items()
            for p_atom in atoms
            for l_coords in ligand_coords
            if np.linalg.norm(p_atom["coords"] - l_coords)
            < cfg["train_config"]["INTERACTION_DISTANCE"]
        }
        if not interacting_res_ids_set:
            continue
        interacting_res_ids = sorted(list(interacting_res_ids_set))
        graph = create_graph(
            protein_residues,
            ligand_mol,
            interacting_res_ids,
            protein_id,
            finger_map,
            vocab_maps,
        )
        all_graphs.append(graph)
        # fake label
        all_labels.append(torch.tensor([float(0)]))
        all_scores.append(float(scores) * 0.01)
    return all_graphs, all_labels, all_scores


def get_sort_list(fasta_file_path: str, sdf_file_path: str, way: str) -> list:
    sort_list = []
    # 获取path的所有文件
    fasta_name = os.path.basename(fasta_file_path).split(".")[0]
    files = os.listdir(f"P450_docking/{fasta_name}-Yaml-Out-Pdb_docking")
    for file in files:
        if file.endswith(".pdb"):
            # 提取文件名前的ID部分
            pattern = r"^(.*?)_model_(.*?)_docked_\d+_(\d+)_addH"
            match = re.match(pattern, file)
            protein_id = match.group(1)
            ligand_id = match.group(2)
            scores = match.group(3)
            sort_list.append([protein_id, ligand_id, scores, file])
    return sort_list


def preprocess_dataset(
    sort_list: list, way: str, fasta_file_path: str, sdf_file_path: str
):
    if way == "Seq-Only":
        sort_list = []
        # 获取seq id和substrate id
        all_substrates = sdf_file_path.split(".")[0]
        all_seqs = get_seq_id(fasta_file_path=f"P450_docking/{fasta_file_path}")
        for protein_id in all_seqs:
            ligand_id = all_substrates
            scores = 0  # fake score
            file = ""  # no file
            sort_list.append([protein_id, ligand_id, scores, file])
    # info
    all_info = []
    for protein_id, ligand_id, scores, file in sort_list:
        all_info.append(
            {
                "enzyme_id": protein_id,
                "substrate_id": ligand_id,
            }
        )
    # other tensor
    if way in ["Pocket-Only", "DM-P450"]:
        all_graphs, all_labels, all_scores = preprocess_graph(
            sort_list=sort_list, fasta_file_path=fasta_file_path
        )
        dgl.save_graphs(
            str(cfg["data_path"]["Pocket_P450_PATH"]["graph_save_file"]),
            all_graphs,
            {"labels": torch.stack(all_labels)},
        )
        # 保存特征维度等元信息，供训练和推理时使用
        sample_graph = all_graphs[0]
        info_to_save = {
            "ligand_feature_dim": sample_graph.nodes["ligand"].data["h"].shape[1],
            "protein_feature_dim": sample_graph.nodes["protein"].data["h"].shape[1],
        }
        with open(cfg["data_path"]["Pocket_P450_PATH"]["info_save_file"], "wb") as f:
            pickle.dump(info_to_save, f)
        # 保存分数
        torch.save(
            torch.tensor(all_scores).unsqueeze(-1),
            cfg["data_path"]["Pocket_P450_PATH"]["score_save_file"],
        )
        torch.save(all_info, cfg["data_path"]["psname_save_file"])

    if way in ["Seq-Only", "DM-P450"]:
        all_seqs, all_substrates = [], []
        preprocess_seq(input_fasta_path=fasta_file_path, sort_list=sort_list)
        preprocess_substrate(input_substrate_path=sdf_file_path)
        for protein_id, ligand_id, scores, file in sort_list:
            all_seqs.append(
                torch.load(Path(cfg["data_path"]["Cache_PATH"]) / f"{protein_id}.pt")[
                    "mean_representations"
                ][33]
            )
            all_substrates.append(
                torch.load(Path(cfg["data_path"]["Cache_PATH"]) / f"{ligand_id}.pt")
            )
        # 保存序列和底物数据
        torch.save(
            all_seqs,
            cfg["data_path"]["Seq_P450_PATH"]["infer_seq_save_file"],
        )
        torch.save(
            all_substrates,
            cfg["data_path"]["Seq_P450_PATH"]["infer_substrate_save_file"],
        )
        torch.save(all_info, cfg["data_path"]["psname_save_file"])


def preprocess(sdf_file_path: str, fasta_file_path: str, way: str):
    assert way in ["Seq-Only", "Pocket-Only", "DM-P450"], (
        f"way 必须是 Seq-Only / Pocket-Only / DM-P450 之一，目前为: {way}"
    )
    if way in ["Seq-Only"]:
        preprocess_dataset(
            sort_list=None,
            way=way,
            fasta_file_path=fasta_file_path,
            sdf_file_path=sdf_file_path,
        )
    elif way in ["Pocket-Only", "DM-P450"]:
        sort_list = get_sort_list(
            fasta_file_path=fasta_file_path, sdf_file_path=sdf_file_path, way=way
        )
        preprocess_dataset(
            sort_list=sort_list,
            way=way,
            fasta_file_path=fasta_file_path,
            sdf_file_path=sdf_file_path,
        )
    elif way in ["DM-P450"]:
        sort_list = get_sort_list(
            fasta_file_path=fasta_file_path, sdf_file_path=sdf_file_path, way=way
        )
        preprocess_dataset(
            sort_list=sort_list,
            way=way,
            fasta_file_path=fasta_file_path,
            sdf_file_path=sdf_file_path,
        )
    else:
        raise ValueError(f"不支持的预处理方式: {way}")


if __name__ == "__main__":
    preprocess(
        sdf_file_path="AGI.sdf",
        fasta_file_path="test.fasta",
        way="DM-P450",
    )
