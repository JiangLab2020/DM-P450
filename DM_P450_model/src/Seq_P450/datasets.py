from torch.utils.data import Dataset
import torch
import pathlib

torch.serialization.add_safe_globals([pathlib.PosixPath])


class SeqP450Dataset(Dataset):
    """
    用于加载已经编码好的 protein 和 substrate tensor 对。
    每一行表示一个样本。
    """

    def __init__(self, protein_path, substrate_path):
        self.load_tensor = lambda path: torch.load(
            path, map_location="cpu", weights_only=True
        )
        self.proteins = self.load_tensor(protein_path)
        self.substrates = self.load_tensor(substrate_path)

        assert len(self.proteins) == len(self.substrates), "两个tensor长度必须一致"
        print(f"SeqP450Dataset 初始化完成，包含 {len(self.proteins)} 个样本。")

    def __getitem__(self, idx):
        return (
            self.proteins[idx],
            self.substrates[idx],
        )

    def __len__(self):
        return len(self.proteins)
