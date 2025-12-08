from DM_P450_model.src.Pocket_P450.datasets import PocketP450Dataset
from DM_P450_model.src.Seq_P450.datasets import SeqP450Dataset
from torch.utils.data import Dataset
import dgl
import torch


class DMP450Dataset(Dataset):
    """
    DMP450数据集，包含图数据和序列数据
    inputs:
        graph_path: 图数据文件路径
        info_path: 图数据的元信息文件路径
        score_path: 图数据的打分文件路径
        sequence_path: 序列数据文件路径
        substrate_path: 底物数据文件路径
        name_path : 原始酶底物文件路径
    outputs:
        pocket_dataset: PocketP450Dataset实例
        seq_dataset: SeqP450Dataset实例
        labels: 标签列表，来自图数据集
    """

    def __init__(
        self,
        graph_path,
        info_path,
        score_path,
        sequence_path,
        substrate_path,
        name_path,
    ):
        self.pocket_dataset = PocketP450Dataset(graph_path, info_path, score_path)
        self.seq_dataset = SeqP450Dataset(sequence_path, substrate_path)
        self.psname_dataset = torch.load(
            name_path, map_location="cpu", weights_only=True
        )
        assert len(self.pocket_dataset) == len(self.seq_dataset), (
            "图数据集、序列数据集长度不一致！"
        )

    def __len__(self):
        return len(self.pocket_dataset)

    def __getitem__(self, idx):
        pocket_datas = self.pocket_dataset[idx]
        seq_datas = self.seq_dataset[idx]
        psnames = self.psname_dataset[idx]
        return pocket_datas, seq_datas, psnames


def collate_fn(batch):
    # batch 是一个 list，每个元素是 (pocket_data, seq_data)
    pocket_data, seq_data, psnames = zip(*batch)  # 分开
    graphs, labels, scores = zip(*pocket_data)
    (
        seqs,
        substrates,
    ) = zip(*seq_data)
    graph_batch = dgl.batch(graphs)
    score_batch = torch.stack(scores)
    seq_batch = torch.stack(seqs)
    substrate_batch = torch.stack(substrates)
    label_batch = torch.tensor(labels)
    return graph_batch, label_batch, score_batch, seq_batch, substrate_batch, psnames


def collate_fn_seq(batch):
    seq_data, psnames = zip(*batch)  # 分开
    (
        seqs,
        substrates,
    ) = zip(*seq_data)
    seq_batch = torch.stack(seqs)
    substrate_batch = torch.stack(substrates)
    return seq_batch, substrate_batch, psnames


def collate_fn_pocket(batch):
    # batch 是一个 list，每个元素是 (pocket_data, seq_data)
    pocket_data, psnames = zip(*batch)  # 分开
    graphs, labels, scores = zip(*pocket_data)

    graph_batch = dgl.batch(graphs)
    score_batch = torch.stack(scores)
    label_batch = torch.tensor(labels)
    return graph_batch, label_batch, score_batch, psnames


class InferSeqDataset(Dataset):
    def __init__(
        self,
        sequence_path,
        substrate_path,
        name_path,
    ):
        self.seq_dataset = SeqP450Dataset(sequence_path, substrate_path)
        self.psname_dataset = torch.load(
            name_path, map_location="cpu", weights_only=True
        )

    def __len__(self):
        return len(self.seq_dataset)

    def __getitem__(self, idx):
        seq_datas = self.seq_dataset[idx]
        psnames = self.psname_dataset[idx]
        return seq_datas, psnames


class InferPocketDataset(Dataset):
    def __init__(
        self,
        graph_path,
        info_path,
        score_path,
        name_path,
    ):
        self.pocket_dataset = PocketP450Dataset(graph_path, info_path, score_path)
        self.psname_dataset = torch.load(
            name_path, map_location="cpu", weights_only=True
        )

    def __len__(self):
        return len(self.pocket_dataset)

    def __getitem__(self, idx):
        pocket_datas = self.pocket_dataset[idx]
        psnames = self.psname_dataset[idx]
        return pocket_datas, psnames


class InferDMP450Dataset(Dataset):
    def __init__(
        self,
        graph_path,
        info_path,
        score_path,
        sequence_path,
        substrate_path,
        name_path,
    ):
        self.pocket_dataset = PocketP450Dataset(graph_path, info_path, score_path)
        self.seq_dataset = SeqP450Dataset(sequence_path, substrate_path)
        self.psname_dataset = torch.load(
            name_path, map_location="cpu", weights_only=True
        )
        assert len(self.pocket_dataset) == len(self.seq_dataset), (
            "图数据集、序列数据集长度不一致！"
        )

    def __len__(self):
        return len(self.pocket_dataset)

    def __getitem__(self, idx):
        pocket_datas = self.pocket_dataset[idx]
        seq_datas = self.seq_dataset[idx]
        psnames = self.psname_dataset[idx]
        return pocket_datas, seq_datas, psnames
