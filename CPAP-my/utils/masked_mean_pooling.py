import torch
import torch.nn as nn
import torch.nn.functional as F

#########################################
# 辅助函数：mask-aware 平均池化
#########################################
def masked_mean_pooling(x, mask):
    """
    x: [T, B, D]   序列表示
    mask: [B, T]   1=PAD, 0=有效
    return: [B, D] 句级表示
    """
    if mask is None:
        return x.mean(dim=0)  # [B,D]

    # mask=1 表示 PAD，所以先转成有效位
    valid_mask = 1 - mask.float()  # [B,T]，PAD=0，有效=1

    # 广播到 [T,B,1]
    weights = valid_mask.transpose(0, 1).unsqueeze(-1)  # [T,B,1]

    # 求和并归一化
    summed = (x * weights).sum(dim=0)  # [B,D]
    denom = weights.sum(dim=0).clamp(min=1e-6)  # [B,1]
    return summed / denom