import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GATConv, HeteroGraphConv
import toml


class GATBinaryGNN(nn.Module):
    """
    使用图注意力网络 (GAT) 的异构图神经网络，用于P450口袋分类。
    这个模型能够处理包含多种节点和边类型的图结构。
    """

    def __init__(self, in_feats_ligand, in_feats_protein, hidden_dim, num_heads):
        """
        初始化模型层。
        Args:
            in_feats_ligand (int): 配体节点输入特征的维度。
            in_feats_protein (int): 蛋白质节点输入特征的维度。
            hidden_dim (int): GNN隐藏层的维度。
            num_heads (int): GAT中的多头注意力头数。
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # GAT层不允许输入特征维度为0，进行检查
        if in_feats_ligand == 0 or in_feats_protein == 0:
            raise ValueError("输入特征维度不能为0")

        # 第一层 GAT 卷积
        # HeteroGraphConv 允许为图中不同类型的关系定义不同的消息传递函数
        self.conv1 = HeteroGraphConv(
            {
                # 关系一: 配体原子通过化学键连接
                ("ligand", "bonded_to", "ligand"): GATConv(
                    in_feats_ligand, hidden_dim, num_heads, allow_zero_in_degree=True
                ),
                # 关系二: 配体原子与蛋白质残基相互作用
                ("ligand", "interacts_with", "protein"): GATConv(
                    (in_feats_ligand, in_feats_protein),
                    hidden_dim,
                    num_heads,
                    allow_zero_in_degree=True,
                ),
                # 关系三: 蛋白质残基与配体原子相互作用 (反向)
                ("protein", "interacts_with", "ligand"): GATConv(
                    (in_feats_protein, in_feats_ligand),
                    hidden_dim,
                    num_heads,
                    allow_zero_in_degree=True,
                ),
            },
            aggregate="sum",
        )  # 聚合来自不同关系类型的结果
        # 第二层 GAT 卷积
        # 输入维度是上一层的输出维度 (hidden_dim * num_heads)
        self.conv2 = HeteroGraphConv(
            {
                ("ligand", "bonded_to", "ligand"): GATConv(
                    hidden_dim * num_heads,
                    hidden_dim,
                    num_heads,
                    allow_zero_in_degree=True,
                ),
                ("ligand", "interacts_with", "protein"): GATConv(
                    (hidden_dim * num_heads, hidden_dim * num_heads),
                    hidden_dim,
                    num_heads,
                    allow_zero_in_degree=True,
                ),
                ("protein", "interacts_with", "ligand"): GATConv(
                    (hidden_dim * num_heads, hidden_dim * num_heads),
                    hidden_dim,
                    num_heads,
                    allow_zero_in_degree=True,
                ),
            },
            aggregate="sum",
        )
        self.config = toml.load("DM_P450_model/config/config.toml")
        self.mlp = nn.Sequential(
            nn.Linear(512, self.config["train_config"]["PROJ_DIM"]),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.relu = nn.ReLU()

    def forward(self, g, inputs):
        """
        定义模型的前向传播过程。
        Args:
            g (DGLGraph): 输入的异构图批次。
            inputs (dict): 包含节点类型及其对应特征张量的字典。
        Returns:
            torch.Tensor: 维度变化后的向量。
        """
        # 第一层卷积
        h = self.conv1(g, inputs)
        # GATConv的输出是 (N, H, D_out)，其中H是头数。需要reshape以合并多头输出
        h = {k: v.view(v.shape[0], -1) for k, v in h.items()}
        h = {k: self.relu(v) for k, v in h.items()}

        # 第二层卷积
        h = self.conv2(g, h)
        h = {k: v.view(v.shape[0], -1) for k, v in h.items()}
        h = {k: self.relu(v) for k, v in h.items()}

        # 读出 (Readout) 和分类
        with g.local_scope():  # 使用local_scope确保对图的操作不会影响外部
            g.nodes["ligand"].data["h_readout"] = h["ligand"]
            g.nodes["protein"].data["h_readout"] = h["protein"]

            # 使用平均池化从节点特征得到图级别的全局特征
            ligand_global_feat = dgl.mean_nodes(g, "h_readout", ntype="ligand")
            protein_global_feat = dgl.mean_nodes(g, "h_readout", ntype="protein")
            # 将配体和蛋白质的全局特征拼接起来
            global_feat = torch.cat([ligand_global_feat, protein_global_feat], dim=1)
            global_feat = self.mlp(global_feat)
            return global_feat
