"""Load a trained checkpoint and evaluate one chronological data split.

Architecture and data arguments must match the training run so that the model
and checkpoint shapes are identical.
"""
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import platform

from models import Autoformer, DLinear, TimeLLM, TimeLLM_TransformerOnly, iTransformer
from models.GatedBackbones import AutoformerGate, DLinearGate, iTransformerGate
from data_provider.data_factory import data_provider
from utils.tools import load_content


def get_parser():
    p = argparse.ArgumentParser(description='Eval: load checkpoint and compute MAE/RMSE on test set')
    p.add_argument('--task_name', type=str, default='long_term_forecast')
    p.add_argument('--eval_split', type=str, default='test', choices=['train', 'val', 'test'], help='which split to evaluate; default keeps historical behavior')
    p.add_argument('--model_id', type=str, default='Iron_96_48')
    p.add_argument('--model_comment', type=str, default='iron')
    p.add_argument('--model', type=str, default='TimeLLM')
    p.add_argument('--data', type=str, default='custom')
    p.add_argument('--root_path', type=str, default='./dataset/')
    p.add_argument('--data_path', type=str, default='2023_2025_Iron_data.csv')
    p.add_argument('--target', type=str, default='OT')
    p.add_argument('--freq', type=str, default='d')
    p.add_argument('--features', type=str, default='M')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--label_len', type=int, default=48)
    p.add_argument('--pred_len', type=int, default=48)
    p.add_argument('--checkpoints', type=str, default='./checkpoints/')
    p.add_argument('--load_ckpt_dir', type=str, default='',
                   help='optional checkpoint directory to load at eval time; useful for cross-channel transfer diagnostics')
    p.add_argument(
        '--strict_checkpoint_load',
        action='store_true',
        help=(
            'require the checkpoint to match the instantiated model exactly; '
            'paper result reconstruction should enable this'
        ),
    )
    p.add_argument('--enc_in', type=int, default=4)
    p.add_argument('--dec_in', type=int, default=4)
    p.add_argument('--c_out', type=int, default=4)
    p.add_argument('--d_model', type=int, default=32)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--d_layers', type=int, default=1)
    p.add_argument('--d_ff', type=int, default=128)
    p.add_argument('--factor', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--activation', type=str, default='gelu', help='activation (for Autoformer etc.)')
    p.add_argument('--embed', type=str, default='timeF')
    p.add_argument('--des', type=str, default='Iron_Ore_Transport_Exp')
    p.add_argument('--patch_len', type=int, default=16)
    p.add_argument('--stride', type=int, default=8)
    p.add_argument('--use_multiscale_patch', action='store_true', help='match training: multi-scale patch (ablation)')
    p.add_argument('--no_revin', action='store_true', help='match training: ablation without RevIN')
    p.add_argument('--llm_model', type=str, default='LLAMA')
    p.add_argument('--llm_model_id', type=str, default='',
                   help='optional HuggingFace model id/path overriding the built-in llm_model default')
    p.add_argument('--llm_dim', type=int, default=4096)
    p.add_argument('--llm_layers', type=int, default=32)
    p.add_argument('--num_tokens', type=int, default=1000,
                   help='number of text prototype tokens used by the mapping_layer; must match training')
    p.add_argument('--prototype_mode', type=str, default='vocab_mapping', choices=['vocab_mapping', 'direct'],
                   help='text prototype parameterization; must match training')
    p.add_argument('--reprogramming_d_keys', type=int, default=0,
                   help='inner key/value dimension per head for ReprogrammingLayer. 0 keeps historical behavior: use d_ff')
    p.add_argument('--history_gate_fusion', action='store_true',
                   help='match training: auxiliary gate uses compact history features')
    p.add_argument('--history_gate_dim', type=int, default=32,
                   help='match training: hidden dimension of the history-fusion gate branch')
    p.add_argument('--state_event_gate', action='store_true',
                   help='match training: auxiliary gate uses per-channel event-state history features')
    p.add_argument('--state_event_gate_dim', type=int, default=32,
                   help='match training: hidden dimension of the state-event gate branch')
    p.add_argument('--state_event_feature_set', type=str, default='v1', choices=['v1', 'v2'],
                   help='match training: v1 compact state-event features or v2 enhanced event-state features')
    p.add_argument('--expected_loss_head', action='store_true',
                   help='match training: model has expected open/close loss head')
    p.add_argument('--expected_loss_head_dim', type=int, default=64,
                   help='match training: hidden dimension of the expected-loss head')
    p.add_argument('--window_occurrence_head', action='store_true',
                   help='match training: model has window occurrence auxiliary head')
    p.add_argument('--window_occurrence_head_dim', type=int, default=64,
                   help='match training: hidden dimension of the window occurrence auxiliary head')
    p.add_argument('--window_horizons', type=str, default='7,14,30,48',
                   help='match training: comma-separated horizons for window occurrence head')
    p.add_argument('--gate_decision_mode', type=str, default='prob', choices=['prob', 'expected_loss'],
                   help='prob=use P(active)>tau; expected_loss=use sigmoid(close_loss-open_loss)>tau, where tau=0.5 means open_loss<close_loss')
    p.add_argument('--prompt_domain', type=int, default=0)
    p.add_argument('--prompt_bank_name', type=str, default='',
                   help='explicit prompt-bank basename, independent of model_comment')
    p.add_argument('--prompt_type', type=str, default='full', choices=['full', 'short'], help='match training: full or short (ablation)')
    p.add_argument('--ablate_reprogramming', action='store_true', help='match training: ablation without reprogramming layer (linear projection only)')
    p.add_argument('--ablate_prompt', action='store_true', help='match training: ablation without prompt (reprogrammed patches only as LLM input)')
    p.add_argument('--ablate_prompt_description', action='store_true', help='match training: ablation without dataset description in prompt')
    p.add_argument('--ablate_prompt_task', action='store_true', help='match training: ablation without task instruction in prompt')
    p.add_argument('--ablate_prompt_stats', action='store_true', help='match training: ablation without input statistics in prompt')
    p.add_argument('--regression_head_mlp', action='store_true', help='match training: use MLP regression head (Linear->GELU->Dropout->Linear); e.g. Iron_anti_flat')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seasonal_patterns', type=str, default='Monthly')
    p.add_argument('--percent', type=int, default=100)
    p.add_argument('--itr', type=int, default=1, help='experiment count used in training; use 1 for the first run')
    p.add_argument('--device', type=str, default='', help='device: cuda, cpu, or empty for auto (cuda if available)')
    p.add_argument('--eval_batch_size', type=int, default=8, help='batch size for eval, smaller to avoid OOM')
    p.add_argument('--moving_avg', type=int, default=25, help='DLinear moving average window')
    p.add_argument('--output_attention', action='store_true', help='whether to output attention (for Autoformer etc.)')
    p.add_argument('--multivariate', action='store_true', help='custom data: return (seq_len, enc_in) per sample; must match training when eval Iron_multivariate checkpoint')
    p.add_argument('--channel_mixing', action='store_true', help='model has channel mixing layer; must match training when eval Iron_multivariate checkpoint')
    p.add_argument('--save_pred_true', action='store_true', help='save pred and true to checkpoint dir for confusion matrix script')
    p.add_argument('--save_plot', action='store_true', help='save the first forecast window and its plotting data')
    p.add_argument('--save_diagnostics', action='store_true', help='save true_y, raw magnitude pred, gate probabilities, final pred for gate/magnitude/postprocess diagnostics')
    p.add_argument(
        '--record_gate_prob_only',
        action='store_true',
        help=(
            'record auxiliary gate probabilities in diagnostics without '
            'applying a gate decision; requires --aux_confidence_threshold 0'
        ),
    )
    p.add_argument('--weight_decay', type=float, default=0.0, help='match training: if trained with weight_decay>0, pass same value to resolve checkpoint path')
    p.add_argument('--use_aux_head', action='store_true', help='instantiate and export the auxiliary event head; the checkpoint must contain that head')
    p.add_argument('--aux_confidence_threshold', type=float, default=0.5, help='assign zero when the auxiliary event probability is below this value; 0 disables the rule')
    p.add_argument('--aux_smooth_kernel', type=int, default=0, help='if >0, smooth aux_probs with 1D avg_pool (kernel_size=this, stride=1, padding=1) before gating; 0=no smoothing')
    p.add_argument('--zero_threshold', type=float, default=0.0, help='after inverse transformation, assign zero where |prediction| is below this value; 0 disables and -1 selects an automatic scale')
    p.add_argument('--print_prompt_once', action='store_true', help='print the first LLM prompt during evaluation')
    p.add_argument('--output_tag', type=str, default='', help='append tag to output filenames to avoid overwriting (e.g. m5_10ch_zeroshot). Empty=keep old names.')
    p.add_argument(
        '--legacy_state_event_eval',
        action='store_true',
        help=(
            'legacy compatibility mode for the June 2026 state-event diagnostics: '
            'do not inject standardized zero thresholds or per-sample channel ids. '
            'Disabled by default and not intended for new experiments'
        ),
    )
    p.add_argument(
        '--legacy_disable_zero_norm_values',
        action='store_true',
        help='legacy compatibility: omit standardized raw-zero thresholds while retaining channel ids',
    )
    p.add_argument(
        '--legacy_disable_feature_ids',
        action='store_true',
        help='legacy compatibility: omit per-sample channel ids while retaining standardized raw-zero thresholds',
    )
    return p


def build_setting(args, ii=0):
    return '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data,
        args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
        args.factor, args.embed, args.des, ii)


def main():
    parser = get_parser()
    args = parser.parse_args()
    if getattr(args, 'gate_decision_mode', 'prob') == 'expected_loss':
        args.use_aux_head = True

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gated_models = ('TimeLLM', 'TimeLLM_TransformerOnly', 'DLinearGate', 'AutoformerGate', 'iTransformerGate')
    if args.model == 'TimeLLM':
        args.content = load_content(args) if args.prompt_domain else ''
        if getattr(args, 'use_aux_head', False):
            args.use_aux_loss = True
    elif args.model == 'TimeLLM_TransformerOnly':
        args.content = ''
        if getattr(args, 'use_aux_head', False):
            args.use_aux_loss = True
    else:
        args.content = ''

    # 评估时用较小 batch 降低显存
    args.batch_size = getattr(args, 'eval_batch_size', args.batch_size)
    eval_split = getattr(args, 'eval_split', 'test')
    test_set, _ = data_provider(args, eval_split)
    disable_zero_norm = bool(
        getattr(args, 'legacy_state_event_eval', False)
        or getattr(args, 'legacy_disable_zero_norm_values', False)
    )
    if disable_zero_norm:
        args.zero_norm_values = []
    elif getattr(test_set, 'scale', False) and hasattr(test_set, 'scaler'):
        scaler = test_set.scaler
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            args.zero_norm_values = (
                -np.asarray(scaler.mean_, dtype=np.float32) / np.asarray(scaler.scale_, dtype=np.float32)
            ).tolist()
        else:
            args.zero_norm_values = []
    else:
        args.zero_norm_values = []
    # 评估时 drop_last=False，保证拿到全部样本，便于按通道重组为 4 个指标
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    if args.model == 'TimeLLM':
        model = TimeLLM.Model(args).float()
    elif args.model == 'TimeLLM_TransformerOnly':
        model = TimeLLM_TransformerOnly.Model(args).float()
    elif args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'iTransformer':
        model = iTransformer.Model(args).float()
    elif args.model == 'DLinearGate':
        model = DLinearGate(args).float()
    elif args.model == 'AutoformerGate':
        model = AutoformerGate(args).float()
    elif args.model == 'iTransformerGate':
        model = iTransformerGate(args).float()
    else:
        model = DLinear.Model(args).float()

    setting = build_setting(args, ii=args.itr - 1)
    ckpt_dir = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
    if getattr(args, 'ablate_reprogramming', False):
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_reprogram'
    if getattr(args, 'ablate_prompt', False):
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_prompt'
    elif getattr(args, 'ablate_prompt_description', False):
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_prompt_desc'
    elif getattr(args, 'ablate_prompt_task', False):
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_prompt_task'
    elif getattr(args, 'ablate_prompt_stats', False):
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_prompt_stats'
    if getattr(args, 'llm_layers', 32) == 8:
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_llm8'
    if getattr(args, 'dropout', 0.1) == 0.15:
        ckpt_dir = ckpt_dir.rstrip('/') + '_dropout015'
    elif getattr(args, 'dropout', 0.1) == 0.05:
        ckpt_dir = ckpt_dir.rstrip('/') + '_dropout005'
    if getattr(args, 'weight_decay', 0.0) > 0:
        ckpt_dir = ckpt_dir.rstrip('/') + '_ablate_weight_decay'
    load_ckpt_dir = getattr(args, 'load_ckpt_dir', '') or ''
    ckpt_path = os.path.join(load_ckpt_dir.strip() if load_ckpt_dir.strip() else ckpt_dir, 'checkpoint')
    os.makedirs(ckpt_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print('Checkpoint not found:', ckpt_path)
        print('Make sure task_name, model_id, model_comment, data, features, seq_len, etc. match your training (e.g. Iron.sh).')
        return

    print('Loading checkpoint:', os.path.abspath(ckpt_path))
    state = torch.load(ckpt_path, map_location='cpu')
    if getattr(args, 'strict_checkpoint_load', False):
        # Exact result reconstruction must not retain randomly initialized
        # parameters when a checkpoint key or shape differs.
        model.load_state_dict(state, strict=True)
        print('Strict checkpoint load: all keys and tensor shapes matched.')
    else:
        model_state = model.state_dict()
        compatible_state = {}
        skipped_shape = []
        for key, value in state.items():
            if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
                skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                continue
            compatible_state[key] = value
        # 兼容不同版本的模型（是否包含 aux_loss 相关参数）和跨通道 transfer 评估；
        # shape 不一致的输入/输出层保持当前模型初始化，其他可复用权重正常加载。
        missing, unexpected = model.load_state_dict(compatible_state, strict=False)
        if skipped_shape:
            print('Warning: skipped shape-mismatched keys when loading checkpoint (first 20):')
            for key, old_shape, new_shape in skipped_shape[:20]:
                print('  {}: ckpt {} -> model {}'.format(key, old_shape, new_shape))
            if len(skipped_shape) > 20:
                print('  ... {} more'.format(len(skipped_shape) - 20))
        if missing:
            print('Warning: missing keys when loading checkpoint (ignored):', missing)
        if unexpected:
            print('Warning: unexpected keys when loading checkpoint (ignored):', unexpected)
    try:
        model = model.to(device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if 'out of memory' in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            print('CUDA OOM, falling back to CPU (slower).')
            device = torch.device('cpu')
            model = model.to(device)
        else:
            raise
    model.eval()

    f_dim = -1 if args.features == 'MS' else 0
    all_pred, all_true = [], []
    all_aux_logits_list = []  # 每 batch 的辅助头 logits，用于辅助置信度置零
    all_expected_loss_list = []  # 每 batch 的 open_loss/close_loss logits，用于 expected-loss 决策
    all_window_occurrence_logits_list = []  # 每 batch 的窗口级 occurrence logits，用于 window head 诊断
    first_batch_reprogramming_attn = None

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_feat_ids = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_feat_ids = None
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1)

            need_reprogramming_attn = (
                batch_idx == 0 and args.model == 'TimeLLM' and not getattr(args, 'ablate_reprogramming', False)
            )
            need_aux_outputs = (
                args.model in gated_models and getattr(args, 'use_aux_head', False)
            ) or (args.model == 'TimeLLM' and getattr(args, 'print_prompt_once', False))
            if need_reprogramming_attn or need_aux_outputs:
                if args.model == 'TimeLLM':
                    disable_feature_ids = bool(
                        getattr(args, 'legacy_state_event_eval', False)
                        or getattr(args, 'legacy_disable_feature_ids', False)
                    )
                    model_feat_ids = None if disable_feature_ids else batch_feat_ids
                    out = model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        return_reprogramming_attention=need_reprogramming_attn,
                        return_aux_repr=need_aux_outputs,
                        feat_ids=model_feat_ids,
                    )
                else:
                    out = model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        return_reprogramming_attention=need_reprogramming_attn,
                        return_aux_repr=need_aux_outputs,
                    )
                if isinstance(out, tuple):
                    out, extra = out
                    if need_reprogramming_attn:
                        first_batch_reprogramming_attn = extra.get('reprogramming_attn')
                    if batch_idx == 0 and extra.get('prompt_sample') is not None:
                        print('[Eval] 第一条喂入 LLM 的 Prompt (print_prompt_once):\n', extra['prompt_sample'][:500] + ('...' if len(extra.get('prompt_sample', '')) > 500 else ''))
                    if getattr(args, 'use_aux_head', False):
                        aux_this = extra.get('aux_logits')
                        if aux_this is not None and 'has_shipment' in aux_this:
                            a = aux_this['has_shipment']
                            if hasattr(a, 'cpu'):
                                a = a.cpu().numpy()
                            all_aux_logits_list.append(np.asarray(a))
                        if aux_this is not None and 'expected_loss' in aux_this:
                            e = aux_this['expected_loss']
                            if hasattr(e, 'cpu'):
                                e = e.cpu().numpy()
                            all_expected_loss_list.append(np.asarray(e))
                        if aux_this is not None and 'window_occurrence' in aux_this:
                            w = aux_this['window_occurrence']
                            if hasattr(w, 'cpu'):
                                w = w.cpu().numpy()
                            all_window_occurrence_logits_list.append(np.asarray(w))
            else:
                out = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            out = out[:, -args.pred_len:, f_dim:]
            true = batch_y[:, -args.pred_len:, f_dim:]
            out_np = out.cpu().numpy()
            true_np = true.cpu().numpy()
            # multivariate：模型输出 (B*4, pred_len, 4)，按窗口取对角得到 (B, pred_len, 4)
            if getattr(args, 'multivariate', False) and out_np.shape[0] == true_np.shape[0] * test_set.enc_in:
                B = true_np.shape[0]
                enc_in = test_set.enc_in
                out_agg = np.zeros((B, out_np.shape[1], enc_in), dtype=out_np.dtype)
                for b in range(B):
                    for c in range(enc_in):
                        out_agg[b, :, c] = out_np[b * enc_in + c, :, c]
                out_np = out_agg
            all_pred.append(out_np)
            all_true.append(true_np)

    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    # Custom/Iron 独立通道：index = c*tot_len + t，重组为 (tot_len, pred_len, enc_in)
    n_channels = pred.shape[-1]
    if getattr(test_set, 'enc_in', None) and getattr(test_set, 'tot_len', None) and test_set.enc_in > 1 and pred.shape[-1] == 1:
        tot_len = test_set.tot_len
        enc_in = test_set.enc_in
        n_full = tot_len * enc_in
        # 正确顺序：先按通道再按窗口，所以用 (enc_in, tot_len, pred_len, 1) 再 transpose -> (tot_len, pred_len, enc_in)
        pred = pred[:n_full].reshape(enc_in, tot_len, args.pred_len, 1).transpose(1, 2, 0, 3).squeeze(-1)
        true = true[:n_full].reshape(enc_in, tot_len, args.pred_len, 1).transpose(1, 2, 0, 3).squeeze(-1)
        n_channels = enc_in

    # 列名（逆变换前后共用）
    try:
        import pandas as pd
        df = pd.read_csv(os.path.join(args.root_path, args.data_path), nrows=0)
        col_names = list(df.columns[1:])  # 去掉 date
        if len(col_names) != n_channels:
            col_names = ['指标{}'.format(i + 1) for i in range(n_channels)]
    except Exception:
        col_names = ['指标{}'.format(i + 1) for i in range(n_channels)]

    # 先算标准化空间（不逆变换）的 MAE/RMSE
    mae_all_scaled = np.mean(np.abs(pred - true))
    rmse_all_scaled = np.sqrt(np.mean((pred - true) ** 2))
    mae_per_scaled = np.mean(np.abs(pred - true), axis=(0, 1))
    rmse_per_scaled = np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)))

    lines = []
    # 记录评估环境与关键开关，便于复现实验与排查「同一 checkpoint 不同机器结果不同」的问题
    cuda_info = ''
    if device.type == 'cuda':
        try:
            cuda_info = ' | cuda={} | gpu="{}"'.format(
                torch.version.cuda,
                torch.cuda.get_device_name(0),
            )
        except Exception:
            cuda_info = ' | cuda={}'.format(torch.version.cuda)
    lines.append('Env: python={} | torch={}{}'.format(
        platform.python_version(),
        getattr(torch, '__version__', 'unknown'),
        cuda_info,
    ))
    try:
        lines.append('Determinism: cudnn.deterministic={} | cudnn.benchmark={} | use_deterministic_algorithms={}'.format(
            bool(getattr(torch.backends.cudnn, 'deterministic', False)),
            bool(getattr(torch.backends.cudnn, 'benchmark', False)),
            bool(torch.are_deterministic_algorithms_enabled()),
        ))
    except Exception:
        pass
    lines.append('Eval device: {} | gate_decision_mode={} | aux_confidence_threshold={} | aux_smooth_kernel={} | zero_threshold={}'.format(
        str(device),
        getattr(args, 'gate_decision_mode', 'prob'),
        getattr(args, 'aux_confidence_threshold', 0.5),
        getattr(args, 'aux_smooth_kernel', 0),
        getattr(args, 'zero_threshold', 0.0),
    ))
    lines.append('Eval split: {}'.format(eval_split))
    split_label = eval_split.capitalize()
    lines.append('Legacy state-event evaluation: {}'.format(bool(getattr(args, 'legacy_state_event_eval', False))))
    lines.append('Legacy disable zero thresholds: {}'.format(disable_zero_norm))
    lines.append('Legacy disable channel ids: {}'.format(bool(
        getattr(args, 'legacy_state_event_eval', False)
        or getattr(args, 'legacy_disable_feature_ids', False)
    )))
    lines.append('Checkpoint: {}'.format(ckpt_path))
    lines.append('-' * 50)
    lines.append('【不逆变换，标准化空间】')
    lines.append('{} 整体  MAE = {:.6f}  RMSE = {:.6f}'.format(split_label, mae_all_scaled, rmse_all_scaled))
    lines.append('各指标:')
    for i in range(n_channels):
        lines.append('  {}  MAE = {:.6f}  RMSE = {:.6f}'.format(col_names[i], mae_per_scaled[i], rmse_per_scaled[i]))
    lines.append('-' * 50)
    for s in lines:
        print(s)

    # 逆变换到原始量纲后再算 MAE/RMSE
    if getattr(test_set, 'scale', False) and hasattr(test_set, 'inverse_transform'):
        pred_flat = pred.reshape(-1, n_channels)
        true_flat = true.reshape(-1, n_channels)
        pred = test_set.inverse_transform(pred_flat).reshape(pred.shape)
        true = test_set.inverse_transform(true_flat).reshape(true.shape)

    raw_magnitude_pred = pred.copy()
    gate_prob = None
    gate_mask = None
    window_occurrence_logits_save = None
    window_occurrence_prob_save = None

    # Apply the auxiliary event decision before any optional small-value rule.
    aux_conf_thresh = getattr(args, 'aux_confidence_threshold', 0.5)
    record_gate_prob_only = bool(
        getattr(args, 'record_gate_prob_only', False)
    )
    if record_gate_prob_only and float(aux_conf_thresh) != 0.0:
        raise ValueError(
            '--record_gate_prob_only requires '
            '--aux_confidence_threshold 0'
        )
    n_windows_zeroed_by_aux = 0
    gate_mode_str = '通道对通道'
    gate_decision_mode = getattr(args, 'gate_decision_mode', 'prob')
    has_gate_signal = (
        (gate_decision_mode == 'prob' and len(all_aux_logits_list) > 0)
        or (gate_decision_mode == 'expected_loss' and len(all_expected_loss_list) > 0)
    )
    collect_gate_probability = bool(
        aux_conf_thresh > 0 or record_gate_prob_only
    )
    gate_threshold = (
        float('-inf') if record_gate_prob_only else aux_conf_thresh
    )
    if collect_gate_probability and has_gate_signal:
        if gate_decision_mode == 'expected_loss':
            all_expected = np.concatenate(all_expected_loss_list, axis=0)
            logits = np.asarray(all_expected, dtype=np.float64)
            # Model outputs raw costs. Softplus keeps open/close expected losses positive.
            costs = np.logaddexp(logits, 0.0)
            score = costs[..., 1] - costs[..., 0]
            prob_ship = np.where(
                score >= 0,
                1.0 / (1.0 + np.exp(-score)),
                np.exp(score) / (1.0 + np.exp(score)),
            )
            probs = np.stack([1.0 - prob_ship, prob_ship], axis=-1)
        else:
            all_aux = np.concatenate(all_aux_logits_list, axis=0)
            logits = np.asarray(all_aux, dtype=np.float64)
            logits_max = logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits - logits_max) / np.exp(logits - logits_max).sum(axis=-1, keepdims=True)
        n_windows = pred.shape[0]
        # 点对点辅助头：all_aux (n_windows*n_channels, pred_len, 2) -> gate (n_windows, pred_len, n_channels)
        if pred.ndim == 3 and logits.ndim == 3 and logits.shape[1] == pred.shape[1]:
            # 点对点：[B*N, pred_len, 2] 或 多元 channel_mixing 下 (B, pred_len, 2) -> (n_windows, n_channels, pred_len, 2)
            n_win, pred_len_dim, n_ch = pred.shape[0], pred.shape[1], pred.shape[2]
            if logits.size == n_win * n_ch * pred_len_dim * 2:
                prob_ship_3d = probs.reshape(n_win, n_ch, pred_len_dim, 2)[..., 1].astype(np.float32)  # (n_win, n_ch, pred_len)
            elif logits.size == n_win * pred_len_dim * 2:
                # 多元 channel_mixing：辅助头 (n_win, pred_len, 2)，同一预测复用到各通道
                prob_ship_3d = np.broadcast_to(probs[:, None, :, 1], (n_win, n_ch, pred_len_dim)).astype(np.float32)  # (n_win, n_ch, pred_len)
            else:
                raise ValueError('aux logits size {} does not match pred (n_win={}, pred_len={}, n_ch={})'.format(
                    logits.size, n_win, pred_len_dim, n_ch))
            aux_probs_bln = np.transpose(prob_ship_3d, (0, 2, 1))  # (n_win, pred_len, n_ch) = [B, L, N]
            smooth_k = getattr(args, 'aux_smooth_kernel', 0)
            if smooth_k > 0:
                t = torch.from_numpy(aux_probs_bln).float()
                smooth_probs = F.avg_pool1d(t.permute(0, 2, 1), kernel_size=smooth_k, stride=1, padding=smooth_k // 2).permute(0, 2, 1)
                aux_probs_bln = smooth_probs.numpy()
            gate = (aux_probs_bln > gate_threshold).astype(pred.dtype)  # (n_windows, pred_len, n_channels)
            gate_prob = aux_probs_bln.astype(np.float32, copy=True)
            gate_mask = gate.astype(np.int8, copy=True)
            if not record_gate_prob_only:
                pred = pred * gate
            n_windows_zeroed_by_aux = int(np.sum(gate == 0))
            # 统计临界区间占比：若大量 aux_probs 接近阈值 0.5，不同设备/精度/非确定性算子会导致 gate 翻转进而大幅影响指标
            try:
                flat = aux_probs_bln.reshape(-1)
                near = float(np.mean((flat > (aux_conf_thresh - 0.01)) & (flat < (aux_conf_thresh + 0.01))))
                lines.append('Aux probs near threshold: P(|p-{:.2f}|<0.01)={:.4f} (pointwise)'.format(aux_conf_thresh, near))
            except Exception:
                pass
            if record_gate_prob_only:
                print(
                    '  [Gate diagnostics] recorded point-wise P(active) '
                    'without applying a gate decision.'
                )
            else:
                for c in range(n_channels):
                    pc = gate[:, :, c]
                    n_below_c = int(np.sum(pc <= 0))
                    print('  [辅助头 P(有发运) 通道{} 点对点] ≤{:.2f} 的(窗,步)数={}/{}'.format(
                        c, aux_conf_thresh, n_below_c, n_win * pred_len_dim))
            if smooth_k > 0:
                gate_mode_str = '点对点(平滑k={})'.format(smooth_k)
            else:
                gate_mode_str = '点对点'
            if gate_decision_mode == 'expected_loss':
                gate_mode_str = 'expected-loss ' + gate_mode_str
            print('  ({}：gate [B, Seq_Len, N]，共 {} 个(窗,步,通)被置零)'.format(gate_mode_str, n_windows_zeroed_by_aux))
        else:
            prob_ship = probs[:, 1]  # (total_samples,) 旧版整窗/整通道
            if pred.ndim == 3 and prob_ship.size == n_windows * n_channels:
                aux_2d = prob_ship.reshape(n_windows, n_channels).astype(pred.dtype)
                gate = (aux_2d > gate_threshold).astype(pred.dtype)
                gate_prob = np.broadcast_to(aux_2d[:, None, :], pred.shape).astype(np.float32, copy=True)
                gate_mask = np.broadcast_to(gate[:, None, :], pred.shape).astype(np.int8, copy=True)
                if not record_gate_prob_only:
                    pred = pred * gate[:, None, :]
                n_windows_zeroed_by_aux = int(np.sum(gate == 0))
                for c in range(n_channels):
                    pc = aux_2d[:, c]
                    n_below_c = int(np.sum(pc <= aux_conf_thresh))
                    print('  [辅助头 P(有发运) 通道{}] min={:.4f}, max={:.4f}, mean={:.4f}, ≤{:.2f} 的(窗,通)数={}/{}'.format(
                        c, float(np.min(pc)), float(np.max(pc)), float(np.mean(pc)), aux_conf_thresh, n_below_c, n_windows))
                print('  (解耦裁决：{}，同一窗口同通道内 pred_len 步共用同一 gate)'.format(
                    'expected-loss 通道对通道' if gate_decision_mode == 'expected_loss' else '通道对通道'
                ))
            elif pred.ndim == 3 and prob_ship.size == n_windows:
                aux_1d = np.asarray(prob_ship, dtype=pred.dtype)
                gate = (aux_1d > gate_threshold).astype(pred.dtype)
                gate_prob = np.broadcast_to(aux_1d[:, None, None], pred.shape).astype(np.float32, copy=True)
                gate_mask = np.broadcast_to(gate[:, None, None], pred.shape).astype(np.int8, copy=True)
                if not record_gate_prob_only:
                    pred = pred * gate[:, None, None]
                n_windows_zeroed_by_aux = int(np.sum(gate == 0))

    if len(all_window_occurrence_logits_list) > 0:
        try:
            window_logits = np.asarray(np.concatenate(all_window_occurrence_logits_list, axis=0), dtype=np.float64)
            logits_max = window_logits.max(axis=-1, keepdims=True)
            window_probs = np.exp(window_logits - logits_max) / np.exp(window_logits - logits_max).sum(axis=-1, keepdims=True)
            window_prob_active = window_probs[..., 1]
            if pred.ndim == 3:
                n_win, pred_len_dim, n_ch = pred.shape
                n_h = window_prob_active.shape[1] if window_prob_active.ndim >= 2 else 0
                if n_h > 0 and window_prob_active.size == n_win * n_ch * n_h:
                    window_occurrence_logits_save = window_logits.reshape(n_win, n_ch, n_h, 2).transpose(0, 2, 1, 3).astype(np.float32)
                    window_occurrence_prob_save = window_prob_active.reshape(n_win, n_ch, n_h).transpose(0, 2, 1).astype(np.float32)
                elif n_h > 0 and window_prob_active.size == n_win * n_h:
                    window_occurrence_logits_save = window_logits.reshape(n_win, n_h, 2).astype(np.float32)
                    window_occurrence_prob_save = window_prob_active.reshape(n_win, n_h).astype(np.float32)
                else:
                    print('[Window occurrence] skip save: logits shape {} incompatible with pred {}'.format(
                        window_logits.shape, pred.shape))
            else:
                window_occurrence_logits_save = window_logits.astype(np.float32)
                window_occurrence_prob_save = window_prob_active.astype(np.float32)
            if window_occurrence_prob_save is not None:
                print('[Window occurrence] active probability shape:', window_occurrence_prob_save.shape)
        except Exception as e:
            print('[Window occurrence] failed to prepare diagnostics:', e)

    pred_after_gate = pred.copy()

    # 零阈值：接近 0 的预测置为真 0。业务上先有无发运再看发多少；仅评估阶段做，训练不截断以保留梯度
    zero_thresh = getattr(args, 'zero_threshold', 0.0)
    n_zeroed = 0
    zero_mask = np.zeros_like(pred, dtype=bool)
    if zero_thresh != 0:
        if zero_thresh == -1 or zero_thresh < 0:
            # auto：用训练集各通道 std 的最小值的 5% 作为阈值（量纲挂钩）；若仍过小则用预测值最小量级的 10%
            if getattr(test_set, 'scale', False) and hasattr(test_set, 'scaler') and hasattr(test_set.scaler, 'scale_'):
                std_per = np.asarray(test_set.scaler.scale_)
                zero_thresh = float(np.maximum(1e-10, 0.05 * np.min(std_per)))
            else:
                zero_thresh = 1e-4
            # 逆变换后 pred 在原始量纲，若 0.05*min(std) 仍很小（如<0.01），改用「预测绝对值 5% 分位」避免阈值过小导致零作用
            abs_pred = np.abs(pred)
            if zero_thresh < 0.01 and np.any(abs_pred > 0):
                pct = np.percentile(abs_pred[abs_pred > 0], 5)
                if not np.isnan(pct):
                    zero_thresh = max(zero_thresh, float(pct))
        if zero_thresh > 0:
            mask = np.abs(pred) < zero_thresh
            zero_mask = mask.copy()
            n_zeroed = int(np.sum(mask))
            pred = np.where(mask, 0.0, pred)

    mae_all = np.mean(np.abs(pred - true))
    rmse_all = np.sqrt(np.mean((pred - true) ** 2))
    mae_per = np.mean(np.abs(pred - true), axis=(0, 1))
    rmse_per = np.sqrt(np.mean((pred - true) ** 2, axis=(0, 1)))

    # 仅非零真值（true>0）上的 MAE/RMSE，便于对比主实验与 w/o RMGM 等消融在「有货日」上的精度
    mask_nonzero = (true > 0)
    n_nonzero = int(np.sum(mask_nonzero))
    if n_nonzero > 0:
        err_abs = np.abs(pred - true)
        err_sq = (pred - true) ** 2
        mae_nonzero_all = np.sum(err_abs * mask_nonzero) / n_nonzero
        rmse_nonzero_all = np.sqrt(np.sum(err_sq * mask_nonzero) / n_nonzero)
        mae_nonzero_per = np.array([
            np.sum(err_abs[:, :, c] * mask_nonzero[:, :, c]) / max(1, int(np.sum(mask_nonzero[:, :, c])))
            for c in range(n_channels)
        ])
        rmse_nonzero_per = np.array([
            np.sqrt(np.sum(err_sq[:, :, c] * mask_nonzero[:, :, c]) / max(1, int(np.sum(mask_nonzero[:, :, c]))))
            for c in range(n_channels)
        ])
    else:
        mae_nonzero_all = rmse_nonzero_all = float('nan')
        mae_nonzero_per = rmse_nonzero_per = np.full(n_channels, np.nan)

    lines.append('【逆变换到原始量纲】')
    if aux_conf_thresh > 0 and has_gate_signal:
        lines.append('  (C_aux 解耦裁决：preds = preds * (aux_probs > {:.2f})，{}，共 {} 个置零；在 MAE/零阈值之前执行)'.format(aux_conf_thresh, gate_mode_str, n_windows_zeroed_by_aux))
        print('  [C_aux 裁决] preds = preds * (aux_probs > {:.2f})，{}，{} 个置零（裁决在 MAE/零阈值之前已执行）'.format(aux_conf_thresh, gate_mode_str, n_windows_zeroed_by_aux))
        if n_windows_zeroed_by_aux == 0:
            print('  → 若为 0：无(窗口,通道)被置零；MAE/图中已是裁决后结果')
    if zero_thresh > 0:
        lines.append('  (已应用零阈值 |pred|<{:.4g} → 0，本批共 {} 个预测被置零；仅评估阶段)'.format(zero_thresh, n_zeroed))
        print('  [零阈值] threshold={:.4g}, 被置零元素数={}'.format(zero_thresh, n_zeroed))
        if n_zeroed == 0 and zero_thresh < 1000:
            hint = '  提示：逆变换后数据量纲较大，当前阈值过小导致无预测被置零；建议改用 --zero_threshold -1（自动）或更大数值（如 10000）'
            lines.append(hint)
            print(hint)
    lines.append('{} 整体  MAE = {:.6f}  RMSE = {:.6f}'.format(split_label, mae_all, rmse_all))
    lines.append('各指标:')
    for i in range(n_channels):
        lines.append('  {}  MAE = {:.6f}  RMSE = {:.6f}'.format(col_names[i], mae_per[i], rmse_per[i]))
    # 各列量纲不同，逆变换后 MAE/RMSE 会差很多；打印相对误差 MAE/std、RMSE/std 说明标准化下表现是否接近
    if getattr(test_set, 'scale', False) and hasattr(test_set, 'scaler') and hasattr(test_set.scaler, 'scale_'):
        std_per = np.asarray(test_set.scaler.scale_)
        lines.append('各指标 MAE/训练集std（相对误差，越接近说明量纲差异导致逆变换MAE不同）:')
        for i in range(n_channels):
            rel = mae_per[i] / std_per[i] if std_per[i] > 0 else float('nan')
            lines.append('  {}  MAE/std = {:.4f}'.format(col_names[i], rel))
        lines.append('各指标 RMSE/训练集std（标准化RMSE，与MAE/std同理）:')
        for i in range(n_channels):
            rel_rmse = rmse_per[i] / std_per[i] if std_per[i] > 0 else float('nan')
            lines.append('  {}  RMSE/std = {:.4f}'.format(col_names[i], rel_rmse))
    # 仅非零真值（true>0）上的 MAE/RMSE，便于对比主实验与 w/o RMGM
    lines.append('仅非零真值（true>0）上的 MAE/RMSE（有货日精度，共 {} 点）:'.format(n_nonzero))
    lines.append('{} 整体  MAE_nonzero = {:.6f}  RMSE_nonzero = {:.6f}'.format(split_label, mae_nonzero_all, rmse_nonzero_all))
    lines.append('各指标:')
    for i in range(n_channels):
        lines.append('  {}  MAE_nonzero = {:.6f}  RMSE_nonzero = {:.6f}'.format(col_names[i], mae_nonzero_per[i], rmse_nonzero_per[i]))
    lines.append('-' * 50)
    # 打印逆变换后的结果（上面已打印标准化空间部分）
    n_first = 6 + n_channels  # 第一段行数（Checkpoint + 分隔 + 标题 + Test整体 + "各指标" + n_channels行 + 分隔）
    for s in lines[n_first:]:
        print(s)
    # 输出文件名：支持加 tag，避免不同评估互相覆盖
    tag = (getattr(args, 'output_tag', '') or '').strip()
    tag_suffix = ('_' + tag) if tag else ''

    # 保存完整结果到 checkpoint 目录，便于查找各消融结果
    eval_result_path = os.path.join(ckpt_dir, f'eval_result{tag_suffix}.txt')
    try:
        with open(eval_result_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print('结果已保存:', eval_result_path)
    except Exception as e:
        print('保存结果文件失败:', e)
    print('-' * 50)

    # 可选：保存 pred/true 供混淆矩阵等后处理使用
    if getattr(args, 'save_pred_true', False):
        try:
            np.save(os.path.join(ckpt_dir, f'pred{tag_suffix}.npy'), pred)
            np.save(os.path.join(ckpt_dir, f'true{tag_suffix}.npy'), true)
            print('已保存', f'pred{tag_suffix}.npy / true{tag_suffix}.npy', '到', ckpt_dir)
        except Exception as e:
            print('保存 pred/true 失败:', e)

    if getattr(args, 'save_diagnostics', False):
        try:
            diagnostics_path = os.path.join(ckpt_dir, f'diagnostics{tag_suffix}.npz')
            np.savez_compressed(
                diagnostics_path,
                true_y=true,
                raw_magnitude_pred=raw_magnitude_pred,
                gate_prob=gate_prob if gate_prob is not None else np.array([], dtype=np.float32),
                gate_mask=gate_mask if gate_mask is not None else np.array([], dtype=np.int8),
                pred_after_gate=pred_after_gate,
                zero_mask=zero_mask.astype(np.int8),
                final_pred=pred,
                aux_confidence_threshold=np.array(aux_conf_thresh, dtype=np.float64),
                aux_smooth_kernel=np.array(getattr(args, 'aux_smooth_kernel', 0), dtype=np.int64),
                zero_threshold_requested=np.array(getattr(args, 'zero_threshold', 0.0), dtype=np.float64),
                zero_threshold_effective=np.array(zero_thresh if zero_thresh > 0 else 0.0, dtype=np.float64),
                n_zeroed_by_aux=np.array(n_windows_zeroed_by_aux, dtype=np.int64),
                n_zeroed_by_postprocess=np.array(n_zeroed, dtype=np.int64),
                gate_mode=np.array(gate_mode_str),
                gate_decision_mode=np.array(gate_decision_mode),
                record_gate_prob_only=np.array(
                    record_gate_prob_only, dtype=np.bool_
                ),
                col_names=np.asarray(col_names, dtype=str),
                window_horizons=np.asarray([
                    int(x) for x in str(getattr(args, 'window_horizons', '7,14,30,48')).split(',')
                    if str(x).strip()
                ], dtype=np.int64),
                window_occurrence_logits=(
                    window_occurrence_logits_save if window_occurrence_logits_save is not None
                    else np.array([], dtype=np.float32)
                ),
                window_occurrence_prob=(
                    window_occurrence_prob_save if window_occurrence_prob_save is not None
                    else np.array([], dtype=np.float32)
                ),
                reprogramming_attn_first_batch=(
                    first_batch_reprogramming_attn.detach().float().cpu().numpy()
                    if first_batch_reprogramming_attn is not None
                    else np.array([], dtype=np.float32)
                ),
            )
            print('已保存诊断张量:', diagnostics_path)
        except Exception as e:
            print('保存诊断张量失败:', e)

    # 四通道预测图像：pred vs true，首窗口；中文显示 + 清晰优雅版
    if getattr(args, 'save_plot', False) and pred.ndim >= 3 and pred.shape[0] > 0 and pred.shape[2] >= n_channels:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            # 中文字体：若图中中文为方框，请安装其一后重跑，如 Ubuntu: sudo apt install fonts-wqy-microhei fonts-noto-cjk
            plt.rcParams['font.sans-serif'] = [
                'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Noto Sans SC',
                'SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans'
            ]
            plt.rcParams['axes.unicode_minus'] = False
            pred_len = pred.shape[1]
            n_plot = min(4, n_channels)
            fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=100)
            axes = axes.flatten()
            t = np.arange(pred_len)
            # 配色：真实值深蓝实线，预测值橙红虚线，更易区分
            color_true, color_pred = '#1f77b4', '#d62728'
            for c in range(n_plot):
                ax = axes[c]
                ax.plot(t, true[0, :, c], color=color_true, linestyle='-', label='真实值', alpha=0.95, linewidth=2.2)
                ax.plot(t, pred[0, :, c], color=color_pred, linestyle='--', label='预测值', alpha=0.9, linewidth=1.5)
                ax.set_title(col_names[c] if c < len(col_names) else 'Ch{}'.format(c), fontsize=12, fontweight='medium')
                ax.set_xlabel('预测步', fontsize=10)
                ax.set_ylabel('发运量（原始量纲）', fontsize=10)
                ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                ax.grid(True, which='both', alpha=0.25, linestyle='-')
                ax.tick_params(axis='both', labelsize=9)
                ax.set_xlim(-0.5, pred_len - 0.5)
            plt.tight_layout()
            plot_png = os.path.join(ckpt_dir, f'pred_true_4channels{tag_suffix}.png')
            plot_svg = os.path.join(ckpt_dir, f'pred_true_4channels{tag_suffix}.svg')
            fig.savefig(plot_png, dpi=180, bbox_inches='tight', facecolor='white')
            fig.savefig(plot_svg, format='svg', bbox_inches='tight', facecolor='white')
            plt.close()
            # 保存绘图数据，之后改图时可直接加载重绘，无需重跑评估
            data_path = os.path.join(ckpt_dir, f'pred_true_4channels_data{tag_suffix}.npz')
            np.savez(
                data_path,
                pred_first=pred[0],
                true_first=true[0],
                pred_len=np.array(pred_len),
                col_names=np.array(col_names[:n_plot], dtype=object),
            )
            print('已保存四通道预测图:', plot_png, ',', plot_svg)
            print('已保存绘图数据:', data_path)
        except Exception as e:
            print('绘制预测图失败:', e)

if __name__ == '__main__':
    main()
