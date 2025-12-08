from DM_P450_model.src.datasets import (
    InferSeqDataset,
    InferPocketDataset,
    InferDMP450Dataset,
    collate_fn,
    collate_fn_pocket,
    collate_fn_seq,
)
from torch.utils.data import DataLoader
import torch
import toml
from DM_P450_model.src.model import DMP450, SeqP450, PocketP450
from pathlib import Path

cfg = toml.load("DM_P450_model/config/config.toml")


def infer_wrapper(model_type):
    print(f"Running inference for model type: {model_type}")
    # 1. ------------加载数据集-----------------
    # 读取配置
    if model_type == "Seq-Only":
        dataset = InferSeqDataset(
            sequence_path=cfg["data_path"]["Seq_P450_PATH"]["infer_seq_save_file"],
            substrate_path=cfg["data_path"]["Seq_P450_PATH"][
                "infer_substrate_save_file"
            ],
            name_path=cfg["data_path"]["psname_save_file"],
        )
    elif model_type == "Pocket-Only":
        dataset = InferPocketDataset(
            graph_path=cfg["data_path"]["Pocket_P450_PATH"]["graph_save_file"],
            info_path=cfg["data_path"]["Pocket_P450_PATH"]["info_save_file"],
            score_path=cfg["data_path"]["Pocket_P450_PATH"]["score_save_file"],
            name_path=cfg["data_path"]["psname_save_file"],
        )
    elif model_type == "DM-P450":
        dataset = InferDMP450Dataset(
            graph_path=cfg["data_path"]["Pocket_P450_PATH"]["graph_save_file"],
            info_path=cfg["data_path"]["Pocket_P450_PATH"]["info_save_file"],
            score_path=cfg["data_path"]["Pocket_P450_PATH"]["score_save_file"],
            sequence_path=cfg["data_path"]["Seq_P450_PATH"]["infer_seq_save_file"],
            substrate_path=cfg["data_path"]["Seq_P450_PATH"][
                "infer_substrate_save_file"
            ],
            name_path=cfg["data_path"]["psname_save_file"],
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 2. 加载模型
    if model_type == "Seq-Only":
        model = SeqP450()
        state = torch.load(cfg["model_path"]["Seq_Only_path"], map_location="cpu")
        model.load_state_dict(state, strict=False)
    elif model_type == "Pocket-Only":
        model = PocketP450()
        state = torch.load(cfg["model_path"]["Pocket_Only_path"], map_location="cpu")
        model.load_state_dict(state, strict=False)
    elif model_type == "DM-P450":
        model = DMP450()
        model.load_state_dict(
            torch.load(
                cfg["model_path"]["DM_P450_path"],
                map_location="cpu",
                weights_only=False,
            )
        )
    # 3. 创建DataLoader
    if model_type == "Seq-Only":
        infer_loader = DataLoader(
            dataset,
            batch_size=cfg["train_config"]["BATCH_SIZE"],
            shuffle=False,
            collate_fn=collate_fn_seq,
        )
    elif model_type == "Pocket-Only":
        infer_loader = DataLoader(
            dataset,
            batch_size=cfg["train_config"]["BATCH_SIZE"],
            shuffle=False,
            collate_fn=collate_fn_pocket,
        )
    elif model_type == "DM-P450":
        infer_loader = DataLoader(
            dataset,
            batch_size=cfg["train_config"]["BATCH_SIZE"],
            shuffle=False,
            collate_fn=collate_fn,
        )
    # 4. 进行评估
    model.eval()
    all_probs = []
    all_enzyme_ids = []
    all_substrate_ids = []

    with torch.no_grad():
        if model_type == "Seq-Only":
            for (
                batched_seqs,
                batched_substrates,
                batched_psnames,
            ) in infer_loader:
                batched_seqs = batched_seqs.to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                batched_substrates = batched_substrates.to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                predictions = model((batched_seqs, batched_substrates))
                probs = torch.sigmoid(predictions.squeeze()).detach().cpu().numpy()
                all_enzyme_ids.extend(
                    [psname["enzyme_id"] for psname in batched_psnames]
                )
                all_substrate_ids.extend(
                    [psname["substrate_id"] for psname in batched_psnames]
                )
                all_probs.extend(probs)
        elif model_type == "Pocket-Only":
            for (
                batched_graphs,
                batched_labels,
                batched_scores,
                batched_psnames,
            ) in infer_loader:
                batched_graphs = batched_graphs.to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                batched_labels = batched_labels.to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                batched_scores = batched_scores.to(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                node_features = {
                    "ligand": batched_graphs.nodes["ligand"].data["h"],
                    "protein": batched_graphs.nodes["protein"].data["h"],
                }
                predictions = model(
                    (batched_graphs, node_features, batched_scores),
                )
                probs = torch.sigmoid(predictions.squeeze()).detach().cpu().numpy()
                all_enzyme_ids.extend(
                    [psname["enzyme_id"] for psname in batched_psnames]
                )
                all_substrate_ids.extend(
                    [psname["substrate_id"] for psname in batched_psnames]
                )
                all_probs.extend(probs)
        elif model_type == "DM-P450":
            for (
                batched_graphs,
                batched_labels,
                batched_scores,
                batched_seqs,
                batched_substrates,
                batched_psnames,
            ) in infer_loader:
                (
                    batched_graphs,
                    batched_labels,
                    batched_scores,
                    batched_seqs,
                    batched_substrates,
                ) = (
                    batched_graphs.to("cuda" if torch.cuda.is_available() else "cpu"),
                    batched_labels.to("cuda" if torch.cuda.is_available() else "cpu"),
                    batched_scores.to("cuda" if torch.cuda.is_available() else "cpu"),
                    batched_seqs.to("cuda" if torch.cuda.is_available() else "cpu"),
                    batched_substrates.to(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    ),
                )
                node_features = {
                    "ligand": batched_graphs.nodes["ligand"].data["h"],
                    "protein": batched_graphs.nodes["protein"].data["h"],
                }
                predictions = model(
                    (batched_graphs, node_features, batched_scores),
                    (batched_seqs, batched_substrates),
                )
                probs = torch.sigmoid(predictions.squeeze()).detach().cpu().numpy()
                all_enzyme_ids.extend(
                    [psname["enzyme_id"] for psname in batched_psnames]
                )
                all_substrate_ids.extend(
                    [psname["substrate_id"] for psname in batched_psnames]
                )
                all_probs.extend(probs)
    # 5. 保存结果
    output_path = (
        Path(cfg["data_path"]["inference_output_path"])
        / f"inference_results_{model_type}.csv"
    )
    with open(output_path, "w") as f:
        f.write("Enzyme_ID,Substrate_ID,Predicted_Probability\n")
        for (
            eid,
            sid,
            prob,
        ) in zip(
            all_enzyme_ids,
            all_substrate_ids,
            all_probs,
        ):
            f.write(f"{eid},{sid},{prob}\n")
    print(f"Inference results saved to {output_path}")


if __name__ == "__main__":
    # infer_wrapper(model_type="DM-P450")
    infer_wrapper(model_type="Pocket-Only")
    # infer_wrapper(model_type="Seq-Only")
