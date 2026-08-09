# -*- coding: utf-8 -*-
"""
从「真实未来段」推导辅助任务标签，用于联合损失中的隐式正则化。
当前仅保留二分类「是否有发运」，避免多任务过难导致隐空间表征混乱。
"""
import torch
from typing import Dict

# 二分类：0=无发运（未来段全为零），1=有发运（存在非零）
HAS_SHIPMENT_NO, HAS_SHIPMENT_YES = 0, 1


def compute_derived_auxiliary_labels(
    batch_y_future: torch.Tensor,
    enc_in: int,
    zero_eps: float = 1e-6,
    point_to_point: bool = False,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    从未来段 (B, pred_len, C) 推导「是否有发运」二分类标签，与主任务共享骨干、联合训练时使用。

    重要：若 DataLoader 使用 StandardScaler（scale=True），必须传入原始量纲的 batch_y_future，
    Recorded zeros may become negative after standardization, so event labels
    must be derived from values in the source scale.
    训练时 run_main 会先对 batch_y_future 做 inverse_transform 再传入本函数。

    Args:
        batch_y_future: (B, pred_len, enc_in)，真实未来段（原始量纲时才有正确 0/1）
        enc_in: 通道数（此处未用于二分类逻辑，保留接口兼容）
        zero_eps: 视为 0 的数值阈值（原始量纲下，如 1e-6）
        point_to_point: 若 True，返回 (B, pred_len, N) 逐点 0/1，用于点对点辅助头 BCE

    Returns:
        dict: 'has_shipment' (B,) long 或 (B, pred_len, N) float，0=无发运 / 1=有发运
    """
    if point_to_point:
        # 点对点：(B, pred_len, N)，该步该通道 target > 0 则为 1
        has_shipment = (batch_y_future.abs() > zero_eps).float()
        return {'has_shipment': has_shipment}
    # 整窗：(B,) 该窗内任一步任一道有发运则为 1
    has_any = (batch_y_future.abs() > zero_eps).any(dim=(1, 2))  # (B,)
    has_shipment = has_any.long()  # 0 或 1
    return {'has_shipment': has_shipment}
