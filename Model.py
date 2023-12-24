import torch
from torch import nn
from Encoder import Encoder_layer

class model(nn.Module):
    def __init__(self, class_num, embed_dim, head_num, encoder_num, dropout, Seq, device):
        super().__init__()

        # 编码
        self.Embed = nn.Sequential(
            nn.Conv1d(1, embed_dim, 11, 1, 5),
            nn.BatchNorm1d(embed_dim),
            nn.AvgPool1d(1, 1, 0)
        )

        # Encoder(embed_dim, head_num, hidden_dim, dropout) hidden_dim:embed_dim*4
        self.encoder = nn.ModuleList(
            [Encoder_layer(embed_dim, head_num, dropout=dropout, Seq=Seq, device=device) for _ in range(encoder_num)]
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(Seq, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, class_num)
        )

    def forward(self, input):
        x = input.reshape((input.shape[0], 1, input.shape[1]))  # [batch, seq] -> [batch, dim, Seq]
        x = self.Embed(x)       # [batch, dim, Seq]

        for enconder_layer in self.encoder:
            # [batch, dim, Seq]
            x = enconder_layer(x)

        x = torch.mean(x, dim=1)        # 全局平均池化层   尝试换一个维度

        x = self.fc(x)      # 全连接

        return x

