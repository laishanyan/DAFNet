import torch
import torch.nn as nn
import torch.nn.functional as F


class DisasterTextEncoderV2(nn.Module):
    """
    Disaster-Aware Text Encoder (with external disaster proto):

    输入:
        hidden_states: [B, L, H]  BERT 输出
        attention_mask: [B, L]    1=有效, 0=padding
        disaster_proto: [1, H] 或 [B, H] 灾害词典向量 (和 BERT hidden_size 一致)

    输出:
        z_dis: [B, D]   灾害语义句向量
        aux:   dict:
            - s: [B, L, D]  语义子空间
            - e: [B, L, D]  情感子空间
            - alpha: [B, L] 注意力权重
            - r: [B, L]     灾害先验得分
            - orth_loss: 标量张量
    """

    def __init__(
        self,
        hidden_size: int,          # BERT hidden dim H (e.g., 768)
        subspace_size: int = 256, # 语义/情感子空间维度 D
        attn_hidden_size: int = 128,
        beta: float = 1.0          # 灾害先验在注意力中的权重
    ):
        super().__init__()

        if subspace_size is None:
            subspace_size = hidden_size

        self.hidden_size = hidden_size
        self.subspace_size = subspace_size
        self.attn_hidden_size = attn_hidden_size
        self.beta = beta

        # h -> s (语义通道)
        self.W_s = nn.Linear(hidden_size, subspace_size)
        # h -> e (情感通道)
        self.W_e = nn.Linear(hidden_size, subspace_size)

        # 注意力能量: s -> u
        self.W_a = nn.Linear(subspace_size, attn_hidden_size)
        self.u = nn.Linear(attn_hidden_size, 1, bias=False)

        # 灾害原型从 H 维投影到子空间 D 维
        # 因为 disaster_proto 是 [*, H]，而 s_i 是 [*, D]
        self.q_proj = nn.Linear(hidden_size, subspace_size, bias=False)

        self.dropout = nn.Dropout(0.2)

    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, L, H]
        attention_mask: torch.Tensor,     # [B, L] 1=valid, 0=pad
        disaster_proto: torch.Tensor      # [1, H] 或 [B, H]
    ):
        B, L, H = hidden_states.size()
        device = hidden_states.device

        # -------- 0. 处理 attention_mask --------
        if attention_mask is None:
            attention_mask = torch.ones(B, L, dtype=torch.long, device=device)
        mask = attention_mask.float()  # [B, L]
        # -------- 1. 子空间投影: h -> s, e --------
        # s, e: [B, L, D]
        s = self.W_s(hidden_states)
        e = self.W_e(hidden_states)

        # -------- 2. 正交损失 L_orth --------
        # token 级相似度: [B, L]
        sim = (s * e).sum(dim=-1)
        valid_len = mask.sum(dim=-1) + 1e-8  # [B]
        gamma = (sim * mask).sum(dim=-1) / valid_len  # [B]
        orth_loss = torch.mean(gamma ** 2)

        # 将灾害词典向量投影到子空间 D: [B, H] -> [B, D]
        q_dis = self.q_proj(disaster_proto)  # [B, D]

        # -------- 4. 计算灾害先验 r_i --------
        # s: [B, L, D], q_dis: [B, D]
        # 先扩展 q_dis: [B, 1, D] 以便与 [B, L, D] 做点积
        q_dis_expanded = q_dis.unsqueeze(1)  # [B, 1, D]
        # 点积: [B, L, D] * [B, 1, D] -> [B, L]
        r = torch.sigmoid((s * q_dis_expanded).sum(dim=-1))  # [B, L]

        # -------- 5. 自注意力 + 灾害先验 --------
        # s -> energy
        h_attn = torch.tanh(self.W_a(s))          # [B, L, A]
        u_energy = self.u(h_attn).squeeze(-1)     # [B, L]

        # 加入灾害先验偏置
        u_tilde = u_energy + self.beta * r        # [B, L]

        # 对 padding 位置 mask 掉
        minus_inf = -1e4
        u_tilde = u_tilde.masked_fill(attention_mask == 0, minus_inf)

        # 注意力权重
        alpha = F.softmax(u_tilde, dim=-1)        # [B, L]

        # -------- 6. 聚合得到句级灾害语义向量 z_dis --------
        alpha_expanded = alpha.unsqueeze(-1)      # [B, L, 1]
        alpha_expanded = self.dropout(alpha_expanded)     # [B, L, 1]
        z_dis = torch.sum(alpha_expanded * s, dim=1)  # [B, D]

        return z_dis, orth_loss
