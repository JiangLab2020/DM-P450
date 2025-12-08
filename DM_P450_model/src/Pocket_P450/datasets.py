import pickle
import dgl
import torch


class PocketP450Dataset(dgl.data.DGLDataset):
    """自定义数据集类，用于加载由preprocess.py预处理并保存的图数据。"""

    def __init__(self, graph_path, info_path, score_path):
        self.graph_path = graph_path
        self.info_path = info_path
        self.score_path = score_path
        if graph_path is not None:
            self._load_data()
        self._load_info()
        super().__init__(name="p450_binary_gat")

    def _load_data(self):
        """从磁盘加载图、标签和元数据。"""
        self.graphs, label_dict = dgl.load_graphs(self.graph_path)
        self.labels = label_dict["labels"]
        self.scores = torch.load(self.score_path, weights_only=True)

    def _load_info(self):
        with open(self.info_path, "rb") as f:
            self.info = pickle.load(f)

    def __getitem__(self, idx):
        # DGLDataset需要实现这个方法
        return self.graphs[idx], self.labels[idx], self.scores[idx]

    def __len__(self):
        # DGLDataset需要实现这个方法
        return len(self.graphs)
