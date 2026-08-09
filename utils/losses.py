"""Loss functions used by the DLR magnitude and event streams."""

import torch as t
import torch.nn as nn
import torch.nn.functional as F


class ZeroInflatedLoss(nn.Module):
    def __init__(self, zero_weight=2.0):
        super().__init__()
        self.zero_weight = zero_weight # 增加 0 值的惩罚权重

    def forward(self, forecast, target):
        mse = (forecast - target) ** 2
        # 当目标是 0 时，加大惩罚，强迫模型“学会”预测 0
        weight = t.where(target == 0, t.tensor(self.zero_weight).to(target.device), t.tensor(1.0).to(target.device))
        return t.mean(mse * weight)


class MaskedMSE(nn.Module):
    """
    仅数值头的 Masked MSE：与 JointMaskedAux 中数值头部分完全一致。
    有货日（raw_targets_num > 0）才计入 MSE，没货日不扣分。用于「砍掉辅助头」消融时保持数值头训练信号不变。
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds_num, targets_num, raw_targets_num=None):
        preds_num = preds_num.float()
        targets_num = targets_num.float()
        if preds_num.dim() == 3 and targets_num.dim() == 3 and preds_num.shape != targets_num.shape:
            B, pred_len, N = targets_num.shape
            if preds_num.numel() == B * N * pred_len:
                preds_num = preds_num.view(B, N, pred_len, -1).permute(0, 2, 1, 3).squeeze(-1)
        if raw_targets_num is not None:
            raw_targets_num = raw_targets_num.float().to(device=preds_num.device)
            if raw_targets_num.shape != preds_num.shape and preds_num.dim() == 3 and targets_num.dim() == 3:
                B, pred_len, N = targets_num.shape
                if raw_targets_num.numel() == B * N * pred_len:
                    raw_targets_num = raw_targets_num.view(B, N, pred_len, -1).permute(0, 2, 1, 3).squeeze(-1)
            mask = (raw_targets_num > 0).float()
        else:
            mask = (targets_num > 0).float()
        squared_error = (preds_num - targets_num) ** 2
        masked_error = squared_error * mask
        return masked_error.sum() / (mask.sum() + self.eps)


class MaskedMAE(nn.Module):
    """
    仅数值头的 Masked MAE：与 JointMaskedAuxMAE 中数值头部分一致，无辅助头。
    用于「去掉 AIN 辅助头」消融：两阶段法主实验仅去掉辅助头与评估截断，回归头仍用 Masked MAE。
    支持 use_raw_mask=False（ablate_no_rmgm）时 mask 全 1。
    """
    def __init__(self, eps=1e-8, use_raw_mask=True):
        super().__init__()
        self.eps = eps
        self.use_raw_mask = use_raw_mask

    def forward(self, preds_num, targets_num, raw_targets_num=None):
        preds_num = preds_num.float()
        targets_num = targets_num.float()
        if preds_num.dim() == 3 and targets_num.dim() == 3 and preds_num.shape != targets_num.shape:
            B, pred_len, N = targets_num.shape
            if preds_num.numel() == B * N * pred_len:
                preds_num = preds_num.view(B, N, pred_len, -1).permute(0, 2, 1, 3).squeeze(-1)
        if not self.use_raw_mask:
            mask = t.ones_like(targets_num, device=preds_num.device, dtype=t.float32)
        elif raw_targets_num is not None:
            raw_targets_num = raw_targets_num.float().to(device=preds_num.device)
            if raw_targets_num.shape != preds_num.shape and preds_num.dim() == 3 and targets_num.dim() == 3:
                B, pred_len, N = targets_num.shape
                if raw_targets_num.numel() == B * N * pred_len:
                    raw_targets_num = raw_targets_num.view(B, N, pred_len, -1).permute(0, 2, 1, 3).squeeze(-1)
            mask = (raw_targets_num > 0).float()
        else:
            mask = (targets_num > 0).float()
        abs_error = (preds_num - targets_num).abs()
        masked_error = abs_error * mask
        return masked_error.sum() / (mask.sum() + self.eps)


class JointMaskedMSEAuxBCE(nn.Module):
    """
    联合损失：数值头纯 Masked MSE + 辅助头 BCE。训练时两路分道扬镳，绝不合并：
    - output_num uses Masked MSE on positive records; recorded zeros do not enter the magnitude loss.
    - output_aux → 只进 BCE：用 0/1 标签逼置信度在没货日压向 0、有货日推向 1。
    支持点对点辅助头：preds_aux (B*N, pred_len, 2)、targets_aux (B, pred_len, N)，逐点 BCE。
    推理时用 C_aux 裁决：C_aux < 阈值则该点置 0（点对点门控）。
    软掩码：mask_zero_weight>0 时，0 值日权重为该值（如 0.1），非 0 日权重 1.0，避免 0 值日完全不更新回归头。
    num_loss_scale：回归损失放大系数，total = num_loss_scale * L_reg + lambda * L_aux；>1 时逼模型更关注「测准非零峰值」。
    """
    def __init__(
        self,
        lambda_weight=1.0,
        eps=1e-8,
        use_raw_mask=True,
        use_mae_for_num=False,
        mask_zero_weight=0.0,
        num_loss_scale=1.0,
        aux_loss_type='bce',
        aux_channel_balance=False,
        aux_train_channel_index=-1,
        aux_pos_weight=1.0,
        aux_neg_weight=1.0,
        aux_focal_gamma=2.0,
        aux_focal_gamma_neg=2.0,
        final_loss_weight=0.0,
        final_loss_type='mae',
        final_huber_delta=1.0,
        zero_magnitude_weight=0.0,
        zero_magnitude_loss_type='mae',
        zero_huber_delta=1.0,
        aux_rank_loss_weight=0.0,
        aux_rank_margin=0.0,
        aux_rank_max_pairs=4096,
        expected_loss_weight=0.0,
        expected_loss_type='huber',
        expected_loss_huber_delta=1.0,
        expected_loss_decision_weight=0.0,
        expected_loss_decision_margin=0.0,
        expected_loss_decision_pos_weight=1.0,
        expected_loss_gap_weight=1.0,
        aux_teacher_distill_weight=0.0,
        aux_teacher_distill_type='bce',
        aux_teacher_huber_delta=1.0,
        window_aux_loss_weight=0.0,
        window_horizons='7,14,30,48',
        window_aux_pos_weight=1.0,
        window_aux_neg_weight=1.0,
    ):
        super().__init__()
        self.lambda_weight = lambda_weight
        self.eps = eps
        self.bce = nn.BCELoss()
        # use_raw_mask=False：消融 w/o RMGM，强制 mask 全 1，全部算 MSE，证明原始量纲掩码保护数值头
        self.use_raw_mask = use_raw_mask
        # use_mae_for_num=True：数值头用 Masked MAE 替代 Masked MSE，减轻「回归到均值」、缓解非零预测过于平稳
        self.use_mae_for_num = use_mae_for_num
        # mask_zero_weight：0=硬掩码（0值日权重0）；>0=软掩码，0值日权重为该值（如0.1），非0日权重1.0，保持梯度流动
        self.mask_zero_weight = float(mask_zero_weight)
        # num_loss_scale：回归损失倍数，total = num_loss_scale * L_num + lambda * L_aux，>1 放大回归、减轻分类主导
        self.num_loss_scale = float(num_loss_scale)
        self.aux_loss_type = str(aux_loss_type or 'bce').lower()
        if self.aux_loss_type not in ('bce', 'weighted_bce', 'focal', 'asym_focal'):
            raise ValueError("aux_loss_type must be one of: bce, weighted_bce, focal, asym_focal")
        self.aux_channel_balance = bool(aux_channel_balance)
        self.aux_train_channel_index = int(aux_train_channel_index)
        self.aux_pos_weight = float(aux_pos_weight)
        self.aux_neg_weight = float(aux_neg_weight)
        self.aux_focal_gamma = float(aux_focal_gamma)
        self.aux_focal_gamma_neg = float(aux_focal_gamma_neg)
        self.final_loss_weight = float(final_loss_weight)
        self.final_loss_type = str(final_loss_type or 'mae').lower()
        self.final_huber_delta = float(final_huber_delta)
        self.zero_magnitude_weight = float(zero_magnitude_weight)
        self.zero_magnitude_loss_type = str(zero_magnitude_loss_type or 'mae').lower()
        self.zero_huber_delta = float(zero_huber_delta)
        self.aux_rank_loss_weight = float(aux_rank_loss_weight)
        self.aux_rank_margin = float(aux_rank_margin)
        self.aux_rank_max_pairs = int(aux_rank_max_pairs)
        self.expected_loss_weight = float(expected_loss_weight)
        self.expected_loss_type = str(expected_loss_type or 'huber').lower()
        self.expected_loss_huber_delta = float(expected_loss_huber_delta)
        self.expected_loss_decision_weight = float(expected_loss_decision_weight)
        self.expected_loss_decision_margin = float(expected_loss_decision_margin)
        self.expected_loss_decision_pos_weight = float(expected_loss_decision_pos_weight)
        self.expected_loss_gap_weight = float(expected_loss_gap_weight)
        self.aux_teacher_distill_weight = float(aux_teacher_distill_weight)
        self.aux_teacher_distill_type = str(aux_teacher_distill_type or 'bce').lower()
        self.aux_teacher_huber_delta = float(aux_teacher_huber_delta)
        self.window_aux_loss_weight = float(window_aux_loss_weight)
        if isinstance(window_horizons, str):
            self.window_horizons = [int(x) for x in window_horizons.split(',') if x.strip()]
        else:
            self.window_horizons = [int(x) for x in window_horizons]
        self.window_aux_pos_weight = float(window_aux_pos_weight)
        self.window_aux_neg_weight = float(window_aux_neg_weight)
        for name, value in (
            ('final_loss_type', self.final_loss_type),
            ('zero_magnitude_loss_type', self.zero_magnitude_loss_type),
            ('expected_loss_type', self.expected_loss_type),
        ):
            if value not in ('mae', 'mse', 'huber'):
                raise ValueError("{} must be one of: mae, mse, huber".format(name))
        if self.aux_teacher_distill_type not in ('bce', 'mse', 'huber'):
            raise ValueError("aux_teacher_distill_type must be one of: bce, mse, huber")

    def _align_aux_predictions(self, preds_aux, targets_aux):
        """Return active logits, active probabilities, and labels in the same shape."""
        targets_aux_float = targets_aux.float()
        if preds_aux.dim() >= 1 and preds_aux.shape[-1] == 2:
            active_logits = preds_aux[..., 1] - preds_aux[..., 0]
            probs = preds_aux.softmax(dim=-1)[..., 1]
        else:
            probs = preds_aux.float().clamp(1e-6, 1.0 - 1e-6)
            active_logits = t.log(probs / (1.0 - probs))

        if targets_aux.dim() == 3:
            B, pred_len, N = targets_aux.shape
            if probs.numel() == B * N * pred_len:
                probs = probs.view(B, N, pred_len).permute(0, 2, 1)
                active_logits = active_logits.view(B, N, pred_len).permute(0, 2, 1)
            elif probs.numel() == B * pred_len:
                probs = probs.unsqueeze(-1).expand(-1, -1, N)
                active_logits = active_logits.unsqueeze(-1).expand(-1, -1, N)
            else:
                raise RuntimeError(
                    "JointMaskedMSEAuxBCE: preds_aux shape {} does not match targets_aux (B={}, pred_len={}, N={})".format(
                        preds_aux.shape, B, pred_len, N))
            return active_logits, probs, targets_aux_float

        probs = probs.view(-1)
        active_logits = active_logits.view(-1)
        targets_aux_float = targets_aux_float.view(-1)
        if probs.numel() != targets_aux_float.numel() and targets_aux_float.numel() > 0 and probs.numel() % targets_aux_float.numel() == 0:
            targets_aux_float = targets_aux_float.repeat_interleave(probs.numel() // targets_aux_float.numel())
        elif probs.numel() != targets_aux_float.numel():
            min_len = min(probs.numel(), targets_aux_float.numel())
            probs = probs[:min_len]
            active_logits = active_logits[:min_len]
            targets_aux_float = targets_aux_float[:min_len]
        return active_logits, probs, targets_aux_float

    def _channel_balanced_reduce(self, element_loss, element_weight, channel_ids=None):
        if not self.aux_channel_balance:
            return (element_loss * element_weight).sum() / (element_weight.sum() + self.eps)

        if element_loss.dim() != 3:
            return (element_loss * element_weight).sum() / (element_weight.sum() + self.eps)

        B, _, N = element_loss.shape
        weighted_loss = element_loss * element_weight
        channel_losses = []

        if N > 1:
            for c in range(N):
                w = element_weight[:, :, c]
                if w.sum() > 0:
                    channel_losses.append(weighted_loss[:, :, c].sum() / (w.sum() + self.eps))
        elif channel_ids is not None:
            ids = channel_ids.to(device=element_loss.device).long().view(-1)
            if ids.numel() == B:
                for c in t.unique(ids):
                    mask = (ids == c).float().view(B, 1, 1)
                    w = element_weight * mask
                    if w.sum() > 0:
                        channel_losses.append((weighted_loss * mask).sum() / (w.sum() + self.eps))

        if not channel_losses:
            return (element_loss * element_weight).sum() / (element_weight.sum() + self.eps)
        return t.stack(channel_losses).mean()

    def _selected_channel_mask(self, ref, channel_ids=None):
        """Build the exact commodity mask used by an independent event head."""
        if self.aux_train_channel_index < 0:
            return t.ones_like(ref)
        if ref.dim() != 3:
            raise RuntimeError(
                'aux_train_channel_index requires point-wise auxiliary tensors with shape (B, horizon, channels)'
            )

        batch_size, _, n_channels = ref.shape
        if n_channels > 1:
            if self.aux_train_channel_index >= n_channels:
                raise ValueError(
                    'aux_train_channel_index={} is outside the available channel range [0, {})'.format(
                        self.aux_train_channel_index, n_channels
                    )
                )
            channel_mask = t.zeros_like(ref)
            channel_mask[:, :, self.aux_train_channel_index] = 1.0
            return channel_mask

        if channel_ids is None:
            raise RuntimeError(
                'aux_train_channel_index requires channel_ids when channels are unfolded into single-channel samples'
            )
        ids = channel_ids.to(device=ref.device).long().view(-1)
        if ids.numel() != batch_size:
            raise RuntimeError(
                'channel_ids has {} entries but the auxiliary batch has {}'.format(ids.numel(), batch_size)
            )
        return (ids == self.aux_train_channel_index).to(ref.dtype).view(batch_size, 1, 1).expand_as(ref)

    def _aux_loss(self, active_logits, probs, targets_aux_float, channel_ids=None):
        element_bce = F.binary_cross_entropy_with_logits(active_logits, targets_aux_float, reduction='none')
        if self.aux_loss_type == 'bce':
            class_weight = t.ones_like(targets_aux_float)
        else:
            class_weight = t.where(
                targets_aux_float > 0.5,
                t.full_like(targets_aux_float, self.aux_pos_weight),
                t.full_like(targets_aux_float, self.aux_neg_weight),
            )
        if self.aux_loss_type == 'focal':
            p = t.sigmoid(active_logits)
            pt = t.where(targets_aux_float > 0.5, p, 1.0 - p)
            element_bce = element_bce * (1.0 - pt).clamp(min=0.0).pow(self.aux_focal_gamma)
        elif self.aux_loss_type == 'asym_focal':
            p = t.sigmoid(active_logits)
            pos_factor = (1.0 - p).clamp(min=0.0).pow(self.aux_focal_gamma)
            neg_factor = p.clamp(min=0.0).pow(self.aux_focal_gamma_neg)
            focal_factor = t.where(targets_aux_float > 0.5, pos_factor, neg_factor)
            element_bce = element_bce * focal_factor

        if self.aux_train_channel_index >= 0:
            class_weight = class_weight * self._selected_channel_mask(class_weight, channel_ids)

        return self._channel_balanced_reduce(element_bce, class_weight, channel_ids=channel_ids)

    def _aux_rank_loss(self, active_logits, targets_aux_float):
        if self.aux_rank_loss_weight <= 0:
            return active_logits.new_tensor(0.0)

        logits = active_logits.reshape(-1)
        labels = targets_aux_float.reshape(-1)
        pos = logits[labels > 0.5]
        neg = logits[labels <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return active_logits.new_tensor(0.0)

        max_pairs = max(1, self.aux_rank_max_pairs)
        n_pairs = pos.numel() * neg.numel()
        if n_pairs <= max_pairs:
            diff = pos[:, None] - neg[None, :]
        else:
            pos_idx = t.randint(pos.numel(), (max_pairs,), device=logits.device)
            neg_idx = t.randint(neg.numel(), (max_pairs,), device=logits.device)
            diff = pos[pos_idx] - neg[neg_idx]
        return F.softplus(self.aux_rank_margin - diff).mean()

    def _aux_teacher_distill_loss(self, active_logits, probs, teacher_score, channel_ids=None):
        if self.aux_teacher_distill_weight <= 0 or teacher_score is None:
            return active_logits.new_tensor(0.0)
        teacher_score = self._align_optional_like(teacher_score, probs).clamp(1e-4, 1.0 - 1e-4)
        if self.aux_teacher_distill_type == 'bce':
            element_loss = F.binary_cross_entropy_with_logits(active_logits, teacher_score, reduction='none')
        elif self.aux_teacher_distill_type == 'mse':
            element_loss = (probs - teacher_score).pow(2)
        else:
            element_loss = F.smooth_l1_loss(
                probs,
                teacher_score,
                reduction='none',
                beta=self.aux_teacher_huber_delta,
            )
        if self.aux_train_channel_index < 0:
            return element_loss.mean()
        channel_mask = self._selected_channel_mask(element_loss, channel_ids)
        return (element_loss * channel_mask).sum() / (channel_mask.sum() + self.eps)

    def _align_optional_like(self, values, ref):
        if values is None:
            return None
        values = values.float().to(device=ref.device)
        if values.shape == ref.shape:
            return values
        if values.numel() == ref.numel():
            return values.reshape(ref.shape)
        if ref.dim() == 3:
            B, pred_len, N = ref.shape
            if values.numel() == B * N * pred_len:
                return values.view(B, N, pred_len, -1).permute(0, 2, 1, 3).squeeze(-1)
            if values.numel() == N:
                return values.view(1, 1, N).expand(B, pred_len, N)
            if values.numel() == B:
                return values.view(B, 1, 1).expand(B, pred_len, N)
        raise RuntimeError("cannot align optional tensor of shape {} to {}".format(values.shape, ref.shape))

    def _align_expected_losses(self, values, ref):
        values = values.float().to(device=ref.device)
        if values.dim() == ref.dim() + 1 and values.shape[:-1] == ref.shape and values.shape[-1] == 2:
            return values
        if ref.dim() == 3:
            B, pred_len, N = ref.shape
            if values.numel() == B * N * pred_len * 2:
                return values.view(B, N, pred_len, 2).permute(0, 2, 1, 3)
            if values.numel() == B * pred_len * 2:
                return values.view(B, pred_len, 1, 2).expand(B, pred_len, N, 2)
        raise RuntimeError("cannot align expected-loss tensor of shape {} to {}".format(values.shape, ref.shape))

    def _window_occurrence_labels(self, raw_targets_num, ref_logits):
        """Build window-level any-active labels from daily raw targets.

        raw_targets_num is usually (B, pred_len, N). The output is aligned to
        ref_logits, whose active-logit view is typically (B*N, H) for the
        independent-channel Iron pipeline.
        """
        y = raw_targets_num.float().to(device=ref_logits.device)
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        if y.dim() != 3:
            raise RuntimeError("window labels expect raw_targets_num with dim 2/3, got {}".format(y.shape))
        B, pred_len, N = y.shape
        labels = []
        for horizon in self.window_horizons:
            h = max(1, min(int(horizon), pred_len))
            labels.append((y[:, :h, :] > 0).any(dim=1).float())
        labels = t.stack(labels, dim=-1)  # (B, N, H)
        if ref_logits.shape == (B, len(self.window_horizons)):
            if N != 1:
                labels = labels.mean(dim=1)
            else:
                labels = labels[:, 0, :]
        elif ref_logits.numel() == labels.numel():
            labels = labels.reshape(ref_logits.shape)
        elif ref_logits.dim() == 2 and ref_logits.shape[0] == B * N and ref_logits.shape[1] == len(self.window_horizons):
            labels = labels.reshape(B * N, len(self.window_horizons))
        else:
            raise RuntimeError("cannot align window labels {} to logits {}".format(labels.shape, ref_logits.shape))
        return labels

    def _window_aux_loss(self, window_logits, raw_targets_num):
        if self.window_aux_loss_weight <= 0 or window_logits is None or raw_targets_num is None:
            if window_logits is not None:
                return window_logits.float().new_tensor(0.0)
            return None
        if window_logits.dim() >= 1 and window_logits.shape[-1] == 2:
            active_logits = window_logits[..., 1] - window_logits[..., 0]
        else:
            probs = window_logits.float().clamp(1e-6, 1.0 - 1e-6)
            active_logits = t.log(probs / (1.0 - probs))
        labels = self._window_occurrence_labels(raw_targets_num, active_logits)
        element = F.binary_cross_entropy_with_logits(active_logits, labels, reduction='none')
        weights = t.where(
            labels > 0.5,
            t.full_like(labels, self.window_aux_pos_weight),
            t.full_like(labels, self.window_aux_neg_weight),
        )
        return (element * weights).sum() / (weights.sum() + self.eps)

    def _elementwise_loss(self, pred, target, loss_type, huber_delta):
        if loss_type == 'mae':
            return (pred - target).abs()
        if loss_type == 'mse':
            return (pred - target) ** 2
        return F.smooth_l1_loss(pred, target, reduction='none', beta=huber_delta)

    def forward(self, preds_num, preds_aux, targets_num, targets_aux, raw_targets_num=None, zero_targets_num=None):
        # preds_num / targets_num: 归一化空间（模型输入输出一致）；raw_targets_num 若提供则为原始量纲，用于 mask 与 aux 标签
        # Standardized values cannot identify recorded zeros; use source-scale
        # targets to construct the event mask.
        # preds_aux: (B*N, pred_len, 2) 点对点 logits，或 (B*N, 2) 旧版整窗
        # targets_aux: (B, pred_len, N) 点对点 0/1（必须由原始量纲生成），或 (B,) 整窗
        preds_num = preds_num.float()
        targets_num = targets_num.float()
        expected_loss_pred = None
        teacher_score = None
        window_occurrence_pred = None
        channel_ids = None
        if isinstance(preds_aux, dict):
            expected_loss_pred = preds_aux.get('expected_loss')
            teacher_score = preds_aux.get('chronos_teacher_score')
            window_occurrence_pred = preds_aux.get('window_occurrence')
            channel_ids = preds_aux.get('channel_ids')
            preds_aux = preds_aux.get('has_shipment')
            if preds_aux is None:
                raise RuntimeError("JointMaskedMSEAuxBCE: preds_aux dict must contain 'has_shipment'")
        preds_aux = preds_aux.float()
        if preds_num.dim() == 3 and targets_num.dim() == 3 and preds_num.shape != targets_num.shape:
            B, pred_len, N = targets_num.shape
            if preds_num.numel() == B * N * pred_len:
                preds_num = preds_num.view(B, N, pred_len, -1).permute(0, 2, 1, 3).squeeze(-1)
        zero_targets_num = self._align_optional_like(zero_targets_num, targets_num)
        # Mask：有货日权重 1，没货日：硬掩码时 0，软掩码时 mask_zero_weight（如 0.1）；use_raw_mask=False 时全 1
        if not self.use_raw_mask:
            mask = t.ones_like(targets_num, device=preds_num.device, dtype=t.float32)
        elif raw_targets_num is not None:
            raw_targets_num = self._align_optional_like(raw_targets_num, targets_num)
            if self.mask_zero_weight > 0:
                mask = t.where(raw_targets_num > 0, t.full_like(raw_targets_num, 1.0), t.full_like(raw_targets_num, self.mask_zero_weight))
            else:
                mask = (raw_targets_num > 0).float()
        else:
            if self.mask_zero_weight > 0:
                mask = t.where(targets_num > 0, t.full_like(targets_num, 1.0), t.full_like(targets_num, self.mask_zero_weight))
            else:
                mask = (targets_num > 0).float()
        # 1) 数值头：加权 MSE/MAE，分母为权重和，使 0 值日也有小梯度（软掩码时）
        if self.use_mae_for_num:
            abs_error = (preds_num - targets_num).abs()
            masked_error = abs_error * mask
            loss_num = masked_error.sum() / (mask.sum() + self.eps)
        else:
            squared_error = (preds_num - targets_num) ** 2
            masked_error = squared_error * mask
            loss_num = masked_error.sum() / (mask.sum() + self.eps)

        # 2) 辅助头：默认保持原 BCE；可切换为 cost-sensitive weighted BCE 或 focal loss。
        active_logits, probs, targets_aux_float = self._align_aux_predictions(preds_aux, targets_aux)
        loss_aux = self._aux_loss(active_logits, probs, targets_aux_float, channel_ids=channel_ids)
        loss_rank = self._aux_rank_loss(active_logits, targets_aux_float)
        loss_teacher = self._aux_teacher_distill_loss(
            active_logits,
            probs,
            teacher_score,
            channel_ids=channel_ids,
        )
        total = self.num_loss_scale * loss_num + self.lambda_weight * loss_aux
        if self.aux_rank_loss_weight > 0:
            total = total + self.aux_rank_loss_weight * loss_rank
        if self.aux_teacher_distill_weight > 0:
            total = total + self.aux_teacher_distill_weight * loss_teacher

        # 3) Window-level occurrence auxiliary loss:
        # This supervises whether any active shipment appears within coarser
        # horizons (e.g. 7/14/30/48 days). It is an auxiliary risk signal, not a
        # replacement for the daily gate.
        if self.window_aux_loss_weight > 0 and window_occurrence_pred is not None and raw_targets_num is not None:
            loss_window = self._window_aux_loss(window_occurrence_pred.float(), raw_targets_num)
            if loss_window is not None:
                total = total + self.window_aux_loss_weight * loss_window

        # 4) Explicit expected-loss head:
        # The head learns two decision costs for each point:
        #   open_loss  = |magnitude_pred - y|
        #   close_loss = |zero_target - y|
        # At inference, open_loss < close_loss is the cost-sensitive gate rule.
        # Targets detach the magnitude prediction so this branch trains the gate
        # cost estimator instead of back-propagating a second regression loss.
        if (self.expected_loss_weight > 0 or self.expected_loss_decision_weight > 0) and expected_loss_pred is not None:
            if zero_targets_num is None:
                zero_targets_num = t.zeros_like(targets_num)
            expected_loss_pred_raw = self._align_expected_losses(expected_loss_pred, targets_num)
            expected_loss_pred = F.softplus(expected_loss_pred_raw)
            open_cost_target = (preds_num.detach() - targets_num).abs()
            close_cost_target = (zero_targets_num.detach() - targets_num).abs()
            if self.expected_loss_weight > 0:
                expected_loss_target = t.stack([open_cost_target, close_cost_target], dim=-1)
                loss_expected = self._elementwise_loss(
                    expected_loss_pred,
                    expected_loss_target.detach(),
                    self.expected_loss_type,
                    self.expected_loss_huber_delta,
                ).mean()
                total = total + self.expected_loss_weight * loss_expected
            if self.expected_loss_decision_weight > 0:
                open_better = (open_cost_target < close_cost_target).float()
                score = expected_loss_pred[..., 1] - expected_loss_pred[..., 0]
                if self.expected_loss_decision_margin > 0:
                    signed_score = t.where(open_better > 0.5, score, -score)
                    decision_element = F.softplus(self.expected_loss_decision_margin - signed_score)
                else:
                    pos_weight = t.tensor(
                        self.expected_loss_decision_pos_weight,
                        dtype=score.dtype,
                        device=score.device,
                    )
                    decision_element = F.binary_cross_entropy_with_logits(
                        score,
                        open_better,
                        reduction='none',
                        pos_weight=pos_weight,
                    )
                if self.expected_loss_gap_weight > 0:
                    gap = (close_cost_target - open_cost_target).abs().detach()
                    gap = gap / (gap.mean() + self.eps)
                    gap = 1.0 + self.expected_loss_gap_weight * gap
                    decision_element = decision_element * gap
                loss_decision = decision_element.mean()
                total = total + self.expected_loss_decision_weight * loss_decision

        # 5) Decision-aware soft final loss:
        # 推理时 hard gate 会把预测置为原始量纲 0。训练在标准化空间中进行，因此关门基线不能用 0，
        # Use the standardized value that corresponds to a recorded zero.
        if self.final_loss_weight > 0:
            if zero_targets_num is None:
                zero_targets_num = t.zeros_like(targets_num)
            if probs.shape != preds_num.shape and probs.numel() == preds_num.numel():
                probs = probs.reshape(preds_num.shape)
            soft_final = probs * preds_num + (1.0 - probs) * zero_targets_num
            final_loss = self._elementwise_loss(
                soft_final,
                targets_num,
                self.final_loss_type,
                self.final_huber_delta,
            ).mean()
            total = total + self.final_loss_weight * final_loss

        # 6) Zero-position magnitude regularization:
        # MAE 分解显示 wpos15/wpos2 的 MAE 恶化主要来自 true-zero 上 false open。
        # 该项直接约束 true-zero 位置的 magnitude head 靠近 raw 0 对应的标准化值。
        if self.zero_magnitude_weight > 0:
            if zero_targets_num is None:
                zero_targets_num = t.zeros_like(targets_num)
            if raw_targets_num is not None:
                zero_mask = (raw_targets_num <= 0).float()
            else:
                if targets_aux_float.shape != preds_num.shape and targets_aux_float.numel() == preds_num.numel():
                    targets_aux_float = targets_aux_float.reshape(preds_num.shape)
                zero_mask = (targets_aux_float <= 0.5).float()
            zero_element = self._elementwise_loss(
                preds_num,
                zero_targets_num,
                self.zero_magnitude_loss_type,
                self.zero_huber_delta,
            ) * zero_mask
            zero_loss = zero_element.sum() / (zero_mask.sum() + self.eps)
            total = total + self.zero_magnitude_weight * zero_loss
        return total
