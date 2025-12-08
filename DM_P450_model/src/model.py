from DM_P450_model.src.Pocket_P450.model import GATBinaryGNN
from DM_P450_model.src.Seq_P450.model import SeqP450Model
import toml
from DM_P450_model.src.Pocket_P450.datasets import PocketP450Dataset
import torch
from torch import nn
import torch.nn.functional as F

# 读取配置
cfg = toml.load("DM_P450_model/config/config.toml")


class DMP450(torch.nn.Module):
    """
    DMP450模型
    inputs:
        - pocket_data: 口袋数据，包含图结构和节点特征
        - seq_data: 序列数据，包含蛋白质序列和底物序列
        - label: 标签数据，用于监督学习
    outputs:
        - combined_output: 预测结果
    description:
        DMP450模型结合了图神经网络和序列模型用于预测P450酶的底物结合能力。
    """

    def __init__(self):
        # 初始化模型参数
        super(DMP450, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 仅用于初始化info 所以不需要graph_path和score_path
        self.pocket_init = PocketP450Dataset(
            graph_path=None,
            info_path=cfg["data_path"]["Pocket_P450_PATH"]["info_save_file"],
            score_path=None,
        )
        self.dataset_info = self.pocket_init.info
        self.pocket_model = GATBinaryGNN(
            in_feats_ligand=self.dataset_info["ligand_feature_dim"],
            in_feats_protein=self.dataset_info["protein_feature_dim"],
            hidden_dim=cfg["train_config"]["HIDDEN_DIM"],
            num_heads=cfg["train_config"]["NUM_HEADS"],
        )
        self.seq_model = SeqP450Model()
        self.proj_scores = nn.Linear(1, cfg["train_config"]["PROJ_DIM"])
        self.mlp = nn.Sequential(
            nn.Linear(cfg["train_config"]["PROJ_DIM"], 1),
            nn.Dropout(0.1),
        )
        self.gating = nn.Linear(cfg["train_config"]["PROJ_DIM"] * 3, 3)
        self.to(self.device)

    def forward(self, pocket_data, seq_data):
        # 定义前向传播逻辑
        g, inputs, scores = pocket_data
        seqs, substrates = seq_data
        pocket_output = self.pocket_model(g, inputs)
        seq_output = self.seq_model(seqs, substrates)
        scores_output = self.proj_scores(scores)
        all_feat = torch.cat([pocket_output, seq_output, scores_output], dim=1)
        gate_logits = self.gating(all_feat)
        # NOTE temperature
        gate_weights = F.softmax(gate_logits / 1.5, dim=1)
        fused = (
            gate_weights[:, 0:1] * pocket_output
            + gate_weights[:, 1:2] * seq_output
            + gate_weights[:, 2:3] * scores_output
        )
        # 最终输出
        out = self.mlp(fused)
        return out


class SeqP450(torch.nn.Module):
    def __init__(self):
        # 初始化模型参数
        super(SeqP450, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seq_model = SeqP450Model()
        self.proj_scores = nn.Linear(1, cfg["train_config"]["PROJ_DIM"])
        self.mlp = nn.Sequential(
            nn.Linear(cfg["train_config"]["PROJ_DIM"], 1),
            nn.Dropout(0.1),
        )
        self.to(self.device)

    def forward(self, seq_data):
        # 定义前向传播逻辑
        seqs, substrates = seq_data
        seq_output = self.seq_model(seqs, substrates)
        # 最终输出
        out = self.mlp(seq_output)
        return out


class PocketP450(torch.nn.Module):
    def __init__(self):
        # 初始化模型参数
        super(PocketP450, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 仅用于初始化info 所以不需要graph_path和score_path
        self.pocket_init = PocketP450Dataset(
            graph_path=None,
            info_path=cfg["data_path"]["Pocket_P450_PATH"]["info_save_file"],
            score_path=None,
        )
        self.dataset_info = self.pocket_init.info
        self.pocket_model = GATBinaryGNN(
            in_feats_ligand=self.dataset_info["ligand_feature_dim"],
            in_feats_protein=self.dataset_info["protein_feature_dim"],
            hidden_dim=cfg["train_config"]["HIDDEN_DIM"],
            num_heads=cfg["train_config"]["NUM_HEADS"],
        )
        self.mlp = nn.Sequential(
            nn.Linear(cfg["train_config"]["PROJ_DIM"] + 1, 1),
            nn.Dropout(0.1),
        )
        self.to(self.device)

    def forward(self, pocket_data):
        # 定义前向传播逻辑
        g, inputs, scores = pocket_data
        pocket_output = self.pocket_model(g, inputs)
        all_feat = torch.cat([pocket_output, scores], dim=1)
        # 最终输出
        out = self.mlp(all_feat)
        return out
