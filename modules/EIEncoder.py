import torch
import torch.nn as nn
import torch.nn.functional as F


class EditedImageEncoder(nn.Module):
    """
    Edited-Image Robust Damage Encoder

    输入:
        patch_tokens: [B, P, H]
            - ViT 输出的 patch token 表征 (通常去掉 CLS 之后的特征)
        patch_mask: [B, P] (可选)
            - 1 表示有效 patch，0 表示 padding（如果不做 padding，可传 None）

    输出:
        v_dis: [B, D] 图像灾害语义向量
        aux: dict, 包含
            - s: [B, P, D]  损伤语义通道
            - a: [B, P, D]  编辑伪迹通道
            - e: [B, P]     伪迹概率
            - alpha: [B, P] 注意力权重
            - orth_loss: 标量张量 L_orth^(v)
    """

    def __init__(
        self,
        hidden_size: int,          # ViT hidden dim H
        subspace_size: int = None, # 损伤/伪迹子空间维度 D (默认 = H)
        attn_hidden_size: int = 768,
        lambda_art: float = 1.0    # 伪迹惩罚项系数 λ_art
    ):
        super().__init__()

        if subspace_size is None:
            subspace_size = hidden_size

        self.hidden_size = hidden_size
        self.subspace_size = subspace_size
        self.attn_hidden_size = attn_hidden_size
        self.lambda_art = lambda_art

        # h -> s (损伤语义通道)
        self.proj_damage = nn.Linear(hidden_size, subspace_size)
        # h -> a (编辑伪迹通道)
        self.proj_artifact = nn.Linear(hidden_size, subspace_size)

        # s -> 注意力能量 u_p
        self.attn_fc = nn.Linear(subspace_size, attn_hidden_size)
        self.attn_u = nn.Linear(attn_hidden_size, 1, bias=False)

        # a -> 伪迹概率 e_p
        self.artifact_score = nn.Linear(subspace_size, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(
        self,
        patch_tokens: torch.Tensor,    # [B, P, H]
        patch_mask: torch.Tensor = None  # [B, P] 或 None
    ):
        """
        返回:
            v_dis: [B, D]
            aux: dict
        """
        B, P, H = patch_tokens.size()
        device = patch_tokens.device

        if patch_mask is None:
            patch_mask = torch.ones(B, P, dtype=torch.long, device=device)

        # -------- 1. 双子空间投影 h -> s, a --------
        # s, a: [B, P, D]
        s = self.proj_damage(patch_tokens)   # 损伤语义通道
        a = self.proj_artifact(patch_tokens) # 编辑伪迹通道

        # -------- 2. 正交损失 L_orth^(v) --------
        # 每个 patch 的内积 sim_p: [B, P]
        sim = (s * a).sum(dim=-1)

        # 只对有效 patch 计算平均内积
        mask = patch_mask.float()          # [B, P]
        valid_len = mask.sum(dim=-1) + 1e-8  # [B]
        # 每个样本的 gamma_b = ∑ sim_p / 有效长度
        gamma = (sim * mask).sum(dim=-1) / valid_len  # [B]
        # L_orth = mean_b (gamma_b^2)
        orth_loss = torch.mean(gamma ** 2)

        # -------- 3. 伪迹概率 e_p (基于伪迹通道 a) --------
        # artifact_score: [B, P, D] -> [B, P, 1] -> squeeze -> [B, P]
        e = torch.sigmoid(self.artifact_score(a).squeeze(-1))  # [B, P]

        # -------- 4. 损伤语义注意力 (基础能量 u_p) --------
        # s: [B, P, D] -> [B, P, A] -> [B, P, 1] -> [B, P]
        h_attn = torch.tanh(self.attn_fc(s))         # [B, P, A]
        u_energy = self.attn_u(h_attn).squeeze(-1)   # [B, P]

        # -------- 5. 伪迹惩罚: u_tilde = u - λ_art * e --------
        u_tilde = u_energy - self.lambda_art * e     # [B, P]

        # 对 padding 位置 mask 掉
        minus_inf = -1e4
        u_tilde = u_tilde.masked_fill(patch_mask == 0, minus_inf)

        # -------- 6. softmax 得到注意力权重 alpha_p --------
        alpha = F.softmax(u_tilde, dim=-1)           # [B, P]

        # -------- 7. 聚合得到图像级灾害语义向量 v_dis --------
        # [B, P, D] * [B, P, 1] -> [B, P, D] -> sum_P -> [B, D]
        alpha_expanded = alpha.unsqueeze(-1)         # [B, P, 1]
        alpha_expanded = self.dropout(alpha_expanded)
        v_dis = torch.sum(alpha_expanded * s, dim=1) # [B, D]

        return v_dis, orth_loss
