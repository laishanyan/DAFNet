import torch
import torch.nn as nn
import torch.nn.functional as F


class DisasterConsensusReliabilityFusion(nn.Module):
    """
    DCRF: Disaster Consensus and Reliability Fusion Module

    输入:
        z_text: [B, D]  文本灾害语义向量 (z_dis)
        v_img:  [B, D]  图像灾害语义向量 (v_dis)

    输出:
        h_fuse: [B, D_out]  融合后的统一灾害表示
        aux: dict:
            - h_t: [B, D']   文本投影
            - h_v: [B, D']   图像投影
            - g_t: [B, 1]    文本 gate
            - g_v: [B, 1]    图像 gate
            - align_loss:    标量，对齐损失 L_align
    """

    def __init__(
        self,
        dim_in: int = 256,      # 输入的特征维度 D
        dim_proj: int = 256,    # 统一语义空间维度 D'
        dim_mlp: int = 256,     # 门控 MLP 隐层维度
        lambda_align: float = 1.0
    ):
        super().__init__()

        self.dim_in = dim_in
        self.dim_proj = dim_proj
        self.dim_mlp = dim_mlp
        self.lambda_align = lambda_align

        # 文本 / 图像 投影到统一空间
        self.proj_text = nn.Linear(dim_in, dim_proj)
        self.proj_img  = nn.Linear(dim_in, dim_proj)

        # 门控网络: 先把 m 投影，再为 text/img 各出一个打分
        # m = [h_t; h_v; |h_t - h_v|; h_t * h_v] -> 4 * dim_proj
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * dim_proj, dim_mlp),
            nn.ReLU(),
            # nn.Linear(dim_mlp, dim_mlp),
            # nn.ReLU()
        )
        self.gate_text = nn.Linear(dim_mlp, 1)
        self.gate_img  = nn.Linear(dim_mlp, 1)

    def forward(self, z_text: torch.Tensor, v_img: torch.Tensor):
        """
        z_text: [B, D]
        v_img:  [B, D]
        """
        # -------- 1. 投影到统一空间 --------
        # [B, D] -> [B, D']
        h_t = self.proj_text(z_text)
        h_v = self.proj_img(v_img)

        # -------- 2. 构造跨模态交互特征 m --------
        # abs_diff, prod: [B, D']
        # abs_diff = torch.abs(h_t - h_v)
        prod = h_t * h_v
        # m: [B, 4*D']
        #　m = torch.cat([h_t, h_v, abs_diff, prod], dim=-1)
        m = torch.cat([h_t, h_v, prod], dim=-1)

        # -------- 3. 门控 MLP 得到模态可靠性打分 --------
        # gate_hidden: [B, dim_mlp]
        gate_hidden = self.gate_mlp(m)  # 共享前两层

        # 原始分数 (未归一化): [B, 1]
        r_t = torch.sigmoid(self.gate_text(gate_hidden))
        r_v = torch.sigmoid(self.gate_img(gate_hidden))

        # 归一化成门控权重
        eps = 1e-8
        denom = r_t + r_v + eps
        g_t = r_t / denom  # [B, 1]
        g_v = r_v / denom  # [B, 1]

        # -------- 4. 加权融合 --------
        # [B, D'] * [B, 1] -> [B, D']
        h_fuse = g_t * h_t + g_v * h_v  # [B, D']

        # -------- 5. 对齐损失 L_align --------
        # L_align = mean ||h_t - h_v||^2
        # align_loss = F.mse_loss(h_t, h_v) * self.lambda_align

        return h_fuse
