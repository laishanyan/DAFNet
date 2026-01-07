import torch
import torch.nn as nn
import torch.nn.functional as F


class DamageLevelClassifier(nn.Module):
    """
    灾害受损程度分类器
    输入:  h_fuse [B, D]
    输出:  logits [B, num_classes] (默认 3 类: 无/一般/严重)
    """

    def __init__(
        self,
        dim_in: int = 256,      # 与 DCRF 输出维度一致
        hidden_dim: int = 256,  # 中间层维度, 可与 dim_in 相同
        num_classes: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim_in)

        self.mlp = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, h_fuse: torch.Tensor):
        """
        h_fuse: [B, D]
        """
        # 先做一层 LayerNorm，稳定训练
        x = self.layer_norm(h_fuse)   # [B, D]
        logits = self.mlp(x)          # [B, num_classes]
        return logits
