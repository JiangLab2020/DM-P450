import torch.nn as nn
import torch
import toml


class SeqP450Model(nn.Module):
    def __init__(self):
        super(SeqP450Model, self).__init__()
        self.cfg = toml.load("DM_P450_model/config/config.toml")
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(1280 + 512, self.cfg["train_config"]["PROJ_DIM"]),
            torch.nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, seqs, substrates):
        x = torch.cat([seqs, substrates], dim=-1)
        return self.mlp(x)
