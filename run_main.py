import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, TimeLLM, TimeLLM_TransformerOnly, iTransformer
from models.GatedBackbones import AutoformerGate, DLinearGate, iTransformerGate

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import datetime
import hashlib
import importlib.metadata
import json
import platform
import shlex
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

# SSL/TLS: don't blank CA bundle (can break HuggingFace downloads).
# If the user hasn't provided a custom CA bundle, default to certifi.
if 'CURL_CA_BUNDLE' not in os.environ or not os.environ.get('CURL_CA_BUNDLE'):
    try:
        import certifi  # type: ignore
        os.environ['CURL_CA_BUNDLE'] = certifi.where()
        os.environ.setdefault('REQUESTS_CA_BUNDLE', os.environ['CURL_CA_BUNDLE'])
        os.environ.setdefault('SSL_CERT_FILE', os.environ['CURL_CA_BUNDLE'])
    except Exception:
        # Fall back to system defaults if certifi isn't available.
        os.environ.pop('CURL_CA_BUNDLE', None)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import EarlyStopping, adjust_learning_rate, vali, vali_with_rmse, load_content
from utils.losses import ZeroInflatedLoss, JointMaskedMSEAuxBCE, MaskedMSE, MaskedMAE
from utils.auxiliary_labels import compute_derived_auxiliary_labels

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear, iTransformer, TimeLLM, TimeLLM_TransformerOnly, DLinearGate, AutoformerGate, iTransformerGate]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')
parser.add_argument('--use_deepspeed', action='store_true',
                    help='enable DeepSpeed ZeRO via ds_config_zero2.json (default: off for portability)')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='optional seasonal label retained for upstream data loaders')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--use_multiscale_patch', action='store_true', help='use multi-scale patch embedding (ablation)')
parser.add_argument('--no_revin', action='store_true', help='ablation: disable RevIN (instance normalization)')
parser.add_argument('--ablate_reprogramming', action='store_true', help='ablation: remove reprogramming layer, use linear projection only (no text prototypes cross-attention)')
parser.add_argument('--ablate_prompt', action='store_true', help='ablation: no prompt, only reprogrammed patches as LLM input')
parser.add_argument('--ablate_prompt_description', action='store_true', help='ablation: remove dataset description from prompt')
parser.add_argument('--ablate_prompt_task', action='store_true', help='ablation: remove task instruction (e.g. forecast steps) from prompt')
parser.add_argument('--ablate_prompt_stats', action='store_true', help='ablation: remove input statistics (min/max/median/trend/lags) from prompt')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--prompt_bank_name', type=str, default='',
                    help='explicit prompt-bank basename, independent of model_comment; recommended for immutable backbone comparisons')
parser.add_argument('--prompt_type', type=str, default='full', choices=['full', 'short'], help='full=description+task+input stats; short=description+task only (ablation)')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT, AUTO
parser.add_argument('--llm_model_id', type=str, default='',
                    help='optional HuggingFace model id/path overriding the built-in llm_model default')
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768
parser.add_argument('--llm_random_init', action='store_true', help='use random-initialized LLM backbone with the same architecture instead of loading pre-trained weights')
parser.add_argument('--num_tokens', type=int, default=1000,
                    help='number of text prototype tokens used by the mapping_layer; historical default is 1000')
parser.add_argument('--prototype_mode', type=str, default='vocab_mapping', choices=['vocab_mapping', 'direct'],
                    help='text prototype parameterization: vocab_mapping keeps TimeLLM-style W@E; direct trains E_prime directly')
parser.add_argument('--reprogramming_d_keys', type=int, default=0,
                    help='inner key/value dimension per head for ReprogrammingLayer. 0 keeps historical behavior: use d_ff')
parser.add_argument('--history_gate_fusion', action='store_true',
                    help='augment the auxiliary gate with compact history features; default keeps old LLM-only gate')
parser.add_argument('--history_gate_dim', type=int, default=32,
                    help='hidden dimension of the history-fusion gate branch')
parser.add_argument('--state_event_gate', action='store_true',
                    help='augment the auxiliary gate with per-channel event-state history features')
parser.add_argument('--state_event_gate_dim', type=int, default=32,
                    help='hidden dimension of the per-channel state-event gate branch')
parser.add_argument('--state_event_feature_set', type=str, default='v1', choices=['v1', 'v2'],
                    help='state-event feature set: v1=compact 32-dim, v2=enhanced 64-dim event-state features')
parser.add_argument('--expected_loss_head', action='store_true',
                    help='add an explicit expected-loss head that estimates open_loss and close_loss for gate decisions')
parser.add_argument('--expected_loss_head_dim', type=int, default=64,
                    help='hidden dimension of the expected-loss head')
parser.add_argument('--window_occurrence_head', action='store_true',
                    help='add a lightweight auxiliary head for any-active occurrence within coarse future windows')
parser.add_argument('--window_occurrence_head_dim', type=int, default=64,
                    help='hidden dimension of the window-occurrence auxiliary head')
parser.add_argument('--window_horizons', type=str, default='7,14,30,48',
                    help='comma-separated horizons for window occurrence labels, e.g. 7,14,30,48')


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Adam weight decay (L2 penalty), e.g. 1e-5')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function: MSE, ZeroInflated, MaskedMSE, MaskedMAE, JointMaskedAux, or JointMaskedAuxMAE')
parser.add_argument('--zero_weight', type=float, default=2.0, help='ZeroInflated loss: weight for target==0 (default 2.0)')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--use_aux_loss', action='store_true', help='add auxiliary loss: 2-class 是否有发运 (has_shipment) for implicit regularization')
parser.add_argument('--aux_loss_weight', type=float, default=0.2, help='weight for auxiliary loss when use_aux_loss (default 0.2)')
parser.add_argument('--aux_loss_type', type=str, default='bce', choices=['bce', 'weighted_bce', 'focal', 'asym_focal'],
                    help='gate auxiliary loss: bce=historical, weighted_bce=cost-sensitive, focal=symmetric focal, asym_focal=class-asymmetric focal BCE')
parser.add_argument('--aux_channel_balance', action='store_true',
                    help='average point-wise auxiliary event loss by commodity channel before reducing, to avoid aggregate event loss being dominated by one channel')
parser.add_argument('--aux_train_channel_index', type=int, default=-1,
                    help='train the point-wise auxiliary event head from one channel only; -1 uses all channels')
parser.add_argument('--aux_pos_weight', type=float, default=1.0,
                    help='positive/active class weight for weighted_bce or focal gate loss')
parser.add_argument('--aux_neg_weight', type=float, default=1.0,
                    help='negative/zero class weight for weighted_bce or focal gate loss')
parser.add_argument('--aux_focal_gamma', type=float, default=2.0,
                    help='positive/symmetric focal gamma for --aux_loss_type focal or asym_focal')
parser.add_argument('--aux_focal_gamma_neg', type=float, default=2.0,
                    help='negative focal gamma for --aux_loss_type asym_focal')
parser.add_argument('--aux_rank_loss_weight', type=float, default=0.0,
                    help='pairwise gate ranking loss weight; encourages active logits to exceed zero logits')
parser.add_argument('--aux_rank_margin', type=float, default=0.0,
                    help='margin for pairwise gate ranking loss')
parser.add_argument('--aux_rank_max_pairs', type=int, default=4096,
                    help='maximum active-zero pairs sampled per batch for ranking loss')
parser.add_argument('--num_loss_scale', type=float, default=1.0, help='回归损失放大倍数：total = num_loss_scale*L_reg + aux_weight*L_aux；5 或 10 可逼模型更关注非零峰值。仅 JointMaskedAux/JointMaskedAuxMAE 有效')
parser.add_argument('--ablate_no_rmgm', action='store_true', help='消融 w/o RMGM: 训练时数值头不掩码 0 值，mask 全 1，全部算 MSE；测试仍保留门控')
parser.add_argument('--mask_zero_weight', type=float, default=0.0, help='软掩码：0=硬掩码(0值日权重0)；>0 时 0值日权重为该值(如0.1)，非0日权重1.0，保持梯度流动。仅 JointMaskedAux/JointMaskedAuxMAE 有效')
parser.add_argument('--final_loss_weight', type=float, default=0.0,
                    help='decision-aware soft final loss 权重：soft_final=p*m+(1-p)*zero_target，用于把 gate 与最终 MAE 绑定。仅 JointMaskedAux/JointMaskedAuxMAE 有效')
parser.add_argument('--final_loss_type', type=str, default='mae', choices=['mae', 'mse', 'huber'],
                    help='decision-aware soft final loss 类型')
parser.add_argument('--final_huber_delta', type=float, default=1.0,
                    help='final_loss_type=huber 时的 smooth_l1 beta')
parser.add_argument('--zero_magnitude_weight', type=float, default=0.0,
                    help='true-zero 位置 magnitude regularization 权重，用于抑制 false-open magnitude。仅 JointMaskedAux/JointMaskedAuxMAE 有效')
parser.add_argument('--zero_magnitude_loss_type', type=str, default='mae', choices=['mae', 'mse', 'huber'],
                    help='zero-position magnitude regularization 类型')
parser.add_argument('--zero_huber_delta', type=float, default=1.0,
                    help='zero_magnitude_loss_type=huber 时的 smooth_l1 beta')
parser.add_argument('--expected_loss_weight', type=float, default=0.0,
                    help='expected-loss head 权重：监督 open_loss/close_loss，使推理可用 open_loss < close_loss 决策')
parser.add_argument('--expected_loss_type', type=str, default='huber', choices=['mae', 'mse', 'huber'],
                    help='expected-loss head 的拟合损失类型')
parser.add_argument('--expected_loss_huber_delta', type=float, default=1.0,
                    help='expected_loss_type=huber 时的 smooth_l1 beta')
parser.add_argument('--expected_loss_decision_weight', type=float, default=0.0,
                    help='expected-loss decision/ranking loss 权重：直接监督 open_loss < close_loss 的决策符号')
parser.add_argument('--expected_loss_decision_margin', type=float, default=0.0,
                    help='>0 时使用 margin ranking loss；0 时使用 BCEWithLogits(close_loss-open_loss)')
parser.add_argument('--expected_loss_decision_pos_weight', type=float, default=1.0,
                    help='expected-loss decision BCE 中 open_better 正类权重')
parser.add_argument('--expected_loss_gap_weight', type=float, default=1.0,
                    help='expected-loss decision loss 的代价差距加权强度；0=不按 |close-open| 加权')
parser.add_argument('--regression_head_mlp', action='store_true', help='回归头用 MLP（Linear->GELU->Dropout->Linear）增强非线性，缓解 Mask=0 导致预测平稳直线')
parser.add_argument('--reprog_lr_scale', type=float, default=1.0, help='重编程层+mapping_layer 学习率倍数，相对全局 LR；如 10 表示该层用 10 倍 LR（缓解特征饥饿）')
parser.add_argument('--freeze_except_history_gate', action='store_true',
                    help='freeze all loaded parameters and train only the history-fusion gate delta branch')
parser.add_argument('--freeze_state_event_gate', action='store_true',
                    help='freeze only state_event_mlp while training the base model; enables exact checkpoint-compatible staged residual training')
parser.add_argument('--save_last_checkpoint', action='store_true',
                    help='save the final epoch checkpoint over the validation-best checkpoint; useful when training only gate parameters')
parser.add_argument('--save_epoch_snapshots', action='store_true',
                    help='save selected epoch checkpoints under epoch_snapshots/ for mechanism diagnostics; off by default')
parser.add_argument('--snapshot_epochs', type=str, default='0,1,5,10',
                    help='comma-separated epoch numbers to save when --save_epoch_snapshots is used; epoch 0 is the initialized model')
parser.add_argument('--skip_test_during_training', action='store_true',
                    help='do not evaluate the test split after each epoch; use for validation-only candidate selection')
parser.add_argument('--skip_final_test', action='store_true',
                    help='do not evaluate the final checkpoint on test; evaluate it separately after validation selection')
parser.add_argument('--history_delta_anchor_weight', type=float, default=0.0,
                    help='penalize history-fusion logit delta on confident base-gate points; keeps fusion as a boundary correction')
parser.add_argument('--history_delta_mean_weight', type=float, default=0.0,
                    help='penalize the mean active-logit delta to avoid global probability shifts')
parser.add_argument('--history_delta_anchor_margin', type=float, default=0.10,
                    help='base-gate probability distance from 0.5 treated as confident for delta anchoring')
parser.add_argument('--chronos_teacher_pred_path', type=str, default='',
                    help='optional train-split Chronos prediction .npy in original scale for event-gate distillation')
parser.add_argument('--chronos_teacher_temp', type=float, default=2.0,
                    help='temperature for Chronos teacher score: sigmoid((pred/std)/temp)')
parser.add_argument('--chronos_teacher_distill_weight', type=float, default=0.0,
                    help='weight for distilling Chronos event soft labels into the auxiliary gate')
parser.add_argument('--chronos_teacher_distill_type', type=str, default='bce', choices=['bce', 'mse', 'huber'],
                    help='loss type for Chronos teacher distillation')
parser.add_argument('--chronos_teacher_huber_delta', type=float, default=1.0,
                    help='smooth_l1 beta for chronos_teacher_distill_type=huber')
parser.add_argument('--window_aux_loss_weight', type=float, default=0.0,
                    help='auxiliary loss weight for window occurrence head')
parser.add_argument('--window_aux_pos_weight', type=float, default=1.0,
                    help='positive class weight for window occurrence auxiliary BCE')
parser.add_argument('--window_aux_neg_weight', type=float, default=1.0,
                    help='negative class weight for window occurrence auxiliary BCE')
parser.add_argument('--return_sample_index', action='store_true',
                    help='custom data: return window index for per-window auxiliary supervision')
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--transformer_encoder_layers', type=int, default=4, help='TimeLLM_TransformerOnly: 随机初始化 Transformer Encoder 层数')
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--multivariate', action='store_true', help='custom data: return (seq_len, enc_in) per sample for joint-window training')
parser.add_argument('--channel_mixing', action='store_true', help='use channel mixing layer in TimeLLM for cross-channel synergy (recommended with --multivariate)')
parser.add_argument('--load_ckpt_dir', type=str, default='', help='两阶段训练 Stage 2：从此目录加载 Stage 1 的 checkpoint（目录内需有 checkpoint 文件），再微调。例如 ./checkpoints/...-iron_stage1')
parser.add_argument('--strict_load_ckpt', action='store_true',
                    help='require an exact Stage-II checkpoint/model key and shape match')
parser.add_argument('--provenance_launcher_path', type=str, default='',
                    help='optional launcher path to hash in the reproducibility manifest')

args = parser.parse_args()

# Re-apply the user-declared seed after parsing. Historically this entry point
# seeded all libraries with the hard-coded value 2021 before argument parsing,
# which made --seed informational only. Re-seeding here makes the recorded CLI
# value effective while preserving the historical default.
fix_seed = int(args.seed)
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(fix_seed)
if float(getattr(args, 'window_aux_loss_weight', 0.0)) > 0:
    args.window_occurrence_head = True
args.return_sample_index = bool(
    getattr(args, 'return_sample_index', False)
    or (
        str(getattr(args, 'chronos_teacher_pred_path', '') or '').strip()
        and float(getattr(args, 'chronos_teacher_distill_weight', 0.0)) > 0
    )
)
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = None
if getattr(args, 'use_deepspeed', False):
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


PROJECT_ROOT = Path(__file__).resolve().parent


def _sha256_file(file_path):
    file_path = Path(file_path)
    digest = hashlib.sha256()
    with file_path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def _atomic_write_text(destination, content):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix='.' + destination.name + '.',
        suffix='.tmp',
        dir=str(destination.parent),
    )
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _atomic_write_json(destination, payload):
    _atomic_write_text(
        destination,
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True) + '\n',
    )


def _file_record(file_path):
    if not file_path:
        return None
    path_obj = Path(file_path).expanduser()
    if not path_obj.is_absolute():
        path_obj = (PROJECT_ROOT / path_obj).resolve()
    if not path_obj.is_file():
        return {
            'path': str(path_obj),
            'exists': False,
        }
    stat = path_obj.stat()
    return {
        'path': str(path_obj),
        'exists': True,
        'size_bytes': int(stat.st_size),
        'sha256': _sha256_file(path_obj),
    }


def _git_record():
    record = {'repository': str(PROJECT_ROOT)}
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--short'],
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        record.update({
            'commit': commit,
            'dirty': bool(status),
            'status_short': status,
        })
    except (OSError, subprocess.CalledProcessError) as exc:
        record['error'] = repr(exc)
    return record


def _package_versions(package_names):
    versions = {}
    for package_name in package_names:
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def _source_records():
    model_module = {
        'Autoformer': 'models/Autoformer.py',
        'DLinear': 'models/DLinear.py',
        'iTransformer': 'models/iTransformer.py',
        'TimeLLM': 'models/TimeLLM.py',
        'TimeLLM_TransformerOnly': 'models/TimeLLM_TransformerOnly.py',
        'DLinearGate': 'models/GatedBackbones.py',
        'AutoformerGate': 'models/GatedBackbones.py',
        'iTransformerGate': 'models/GatedBackbones.py',
    }.get(args.model)
    candidates = [
        Path(__file__).resolve(),
        PROJECT_ROOT / 'utils/tools.py',
        PROJECT_ROOT / 'utils/losses.py',
        PROJECT_ROOT / 'data_provider/data_factory.py',
        PROJECT_ROOT / 'data_provider_pretrain/data_loader.py',
    ]
    if model_module:
        candidates.append(PROJECT_ROOT / model_module)
    if str(getattr(args, 'provenance_launcher_path', '') or '').strip():
        candidates.append(Path(args.provenance_launcher_path).expanduser())
    if str(getattr(args, 'resolved_prompt_bank_path', '') or '').strip():
        candidates.append(Path(args.resolved_prompt_bank_path).expanduser())

    records = {}
    for candidate in candidates:
        candidate = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
        candidate = candidate.resolve()
        try:
            label = str(candidate.relative_to(PROJECT_ROOT))
        except ValueError:
            label = str(candidate)
        records[label] = _file_record(candidate)
    return records


def write_reproducibility_manifest(
    output_dir,
    phase,
    stage_input_checkpoint=None,
    output_checkpoint=None,
    early_stopping=None,
):
    """Persist run provenance without changing model behavior."""
    if not accelerator.is_main_process:
        return

    output_dir = Path(output_dir).resolve()
    dataset_path = Path(args.root_path).expanduser() / args.data_path
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    chronos_path = str(getattr(args, 'chronos_teacher_pred_path', '') or '').strip()
    resolved_args = {key: _json_safe(value) for key, value in sorted(vars(args).items())}
    command = shlex.join([sys.executable] + sys.argv)
    environment = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python': sys.version,
        'python_executable': sys.executable,
        'torch': torch.__version__,
        'packages': _package_versions([
            'accelerate',
            'numpy',
            'pandas',
            'scikit-learn',
            'torch',
            'transformers',
        ]),
        'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count_visible': torch.cuda.device_count(),
        'cuda_device_names_visible': [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ] if torch.cuda.is_available() else [],
        'accelerate_distributed_type': str(accelerator.distributed_type),
        'accelerate_mixed_precision': str(accelerator.mixed_precision),
        'world_size': int(accelerator.num_processes),
        'batch_size_per_process': int(args.batch_size),
        'effective_global_training_batch_size': int(args.batch_size) * int(accelerator.num_processes),
        'seed': int(args.seed),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'TRANSFORMERS_OFFLINE': os.environ.get('TRANSFORMERS_OFFLINE'),
        'HF_HUB_OFFLINE': os.environ.get('HF_HUB_OFFLINE'),
        'outer_launcher_command': os.environ.get('REPRO_OUTER_LAUNCH_COMMAND'),
    }
    training_state = None
    if early_stopping is not None:
        training_state = {
            'best_score': early_stopping.best_score,
            'best_validation_loss': (
                early_stopping.val_loss_min
                if np.isfinite(early_stopping.val_loss_min)
                else None
            ),
            'early_stop': bool(early_stopping.early_stop),
            'early_stop_counter': int(early_stopping.counter),
        }
    manifest = {
        'schema_version': 1,
        'phase': str(phase),
        'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'output_directory': str(output_dir),
        'command': command,
        'resolved_args': resolved_args,
        'test_hygiene': {
            'epoch_test_disabled': bool(args.skip_test_during_training),
            'final_test_disabled': bool(args.skip_final_test),
            'test_loader_constructed_during_training': not bool(args.skip_test_during_training),
        },
        'environment': environment,
        'git': _git_record(),
        'source_files': _source_records(),
        'input_artifacts': {
            'dataset': _file_record(dataset_path.resolve()),
            'chronos_teacher': _file_record(chronos_path) if chronos_path else None,
            'stage_input_checkpoint': _file_record(stage_input_checkpoint),
        },
        'output_artifacts': {
            'validation_best_checkpoint': _file_record(output_checkpoint),
        },
        'training_state': training_state,
    }
    _atomic_write_json(output_dir / 'resolved_args.json', resolved_args)
    _atomic_write_text(output_dir / 'launch_command.txt', command + '\n')
    _atomic_write_json(output_dir / 'reproducibility_manifest.json', manifest)


def add_history_delta_regularization(args, aux_logits, loss):
    """Constrain history-fusion to behave as a residual boundary correction."""
    anchor_w = float(getattr(args, 'history_delta_anchor_weight', 0.0))
    mean_w = float(getattr(args, 'history_delta_mean_weight', 0.0))
    if anchor_w <= 0 and mean_w <= 0:
        return loss
    if not aux_logits or 'has_shipment_base' not in aux_logits:
        return loss

    base = aux_logits['has_shipment_base'].float()
    delta_terms = []
    if 'has_shipment_delta' in aux_logits:
        delta_terms.append(aux_logits['has_shipment_delta'].float())
    if 'has_shipment_state_delta' in aux_logits:
        delta_terms.append(aux_logits['has_shipment_state_delta'].float())
    if not delta_terms:
        return loss
    delta = torch.stack(delta_terms, dim=0).sum(dim=0)
    base_margin = (base[..., 1] - base[..., 0]).detach()
    delta_margin = delta[..., 1] - delta[..., 0]
    reg = loss.new_tensor(0.0)

    if anchor_w > 0:
        base_prob = torch.sigmoid(base_margin)
        margin = float(getattr(args, 'history_delta_anchor_margin', 0.10))
        confident_mask = ((base_prob - 0.5).abs() >= margin).float()
        anchor_loss = (delta_margin.pow(2) * confident_mask).sum() / (confident_mask.sum() + 1e-8)
        reg = reg + anchor_w * anchor_loss

    if mean_w > 0:
        mean_loss = delta_margin.mean().pow(2)
        reg = reg + mean_w * mean_loss

    return loss + reg


def attach_zero_norm_values(args, dataset):
    """Expose standardized raw-zero values to models that build event-state features."""
    if getattr(dataset, 'scale', False) and hasattr(dataset, 'scaler'):
        scaler = dataset.scaler
        if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
            zero_norm = (-np.asarray(scaler.mean_, dtype=np.float32) / np.asarray(scaler.scale_, dtype=np.float32))
            args.zero_norm_values = zero_norm.tolist()
            return
    args.zero_norm_values = []


def load_chronos_teacher_score(args, dataset, accelerator):
    """Load train-split Chronos predictions and convert them to event soft labels."""
    path = str(getattr(args, 'chronos_teacher_pred_path', '') or '').strip()
    weight = float(getattr(args, 'chronos_teacher_distill_weight', 0.0))
    if not path or weight <= 0:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError('--chronos_teacher_pred_path not found: {}'.format(path))

    pred = np.load(path).astype(np.float32)
    expected = (int(dataset.tot_len), int(args.pred_len), int(dataset.enc_in))
    if pred.shape != expected:
        raise ValueError(
            'Chronos teacher shape {} does not match train windows {}. '
            'Expected (tot_len, pred_len, enc_in) = {}'.format(pred.shape, path, expected)
        )

    if getattr(dataset, 'scale', False) and hasattr(dataset, 'scaler') and hasattr(dataset.scaler, 'scale_'):
        stds = np.asarray(dataset.scaler.scale_, dtype=np.float32)
    else:
        stds = np.ones((int(dataset.enc_in),), dtype=np.float32)
    stds = np.maximum(stds, 1e-6).reshape(1, 1, -1)
    temp = max(float(getattr(args, 'chronos_teacher_temp', 2.0)), 1e-6)
    z = np.clip(pred / stds / temp, -50.0, 50.0)
    score = 1.0 / (1.0 + np.exp(-z))
    score = np.clip(score, 1e-4, 1.0 - 1e-4).astype(np.float32)
    if accelerator.is_local_main_process:
        accelerator.print(
            '[ChronosTeacher] loaded {} | shape={} | temp={} | distill_weight={} | mean={:.4f} std={:.4f}'.format(
                path,
                score.shape,
                temp,
                weight,
                float(score.mean()),
                float(score.std()),
            )
        )
    return torch.from_numpy(score)


def attach_chronos_teacher_to_aux(aux_logits, chronos_teacher_score, batch_feat_ids, batch_sample_ids, ref_tensor):
    """Attach aligned Chronos teacher scores to the auxiliary-output dict."""
    if aux_logits is None or chronos_teacher_score is None or batch_sample_ids is None:
        return aux_logits
    if not isinstance(aux_logits, dict):
        aux_logits = {'has_shipment': aux_logits}
    else:
        aux_logits = dict(aux_logits)

    sample_ids = batch_sample_ids.detach().cpu().long()
    if ref_tensor.dim() == 3 and ref_tensor.shape[-1] == 1 and batch_feat_ids is not None:
        feat_ids = batch_feat_ids.detach().cpu().long()
        teacher = chronos_teacher_score[sample_ids, :, feat_ids].unsqueeze(-1)
    else:
        teacher = chronos_teacher_score[sample_ids, :, :]
    if teacher.shape != ref_tensor.shape and teacher.numel() == ref_tensor.numel():
        teacher = teacher.reshape(ref_tensor.shape)
    aux_logits['chronos_teacher_score'] = teacher.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
    return aux_logits


for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    # Do not even construct the test Dataset/DataLoader while validation is
    # selecting the checkpoint. If a legacy run requests final-test reporting
    # only, its test loader is created after training and checkpoint locking.
    test_data = None
    test_loader = None
    if not getattr(args, 'skip_test_during_training', False):
        test_data, test_loader = data_provider(args, 'test')
    attach_zero_norm_values(args, train_data)
    chronos_teacher_score = load_chronos_teacher_score(args, train_data, accelerator)

    # A custom description is read only when prompt_domain is enabled.
    if args.model == 'TimeLLM' and args.prompt_domain:
        args.content = load_content(args)
    else:
        args.content = ''

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    elif args.model == 'iTransformer':
        model = iTransformer.Model(args).float()
    elif args.model == 'DLinearGate':
        model = DLinearGate(args).float()
    elif args.model == 'AutoformerGate':
        model = AutoformerGate(args).float()
    elif args.model == 'iTransformerGate':
        model = iTransformerGate(args).float()
    elif args.model == 'TimeLLM_TransformerOnly':
        model = TimeLLM_TransformerOnly.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    if getattr(args, 'ablate_reprogramming', False):
        path = path.rstrip('/') + '_ablate_reprogram'
    if getattr(args, 'ablate_prompt', False):
        path = path.rstrip('/') + '_ablate_prompt'
    elif getattr(args, 'ablate_prompt_description', False):
        path = path.rstrip('/') + '_ablate_prompt_desc'
    elif getattr(args, 'ablate_prompt_task', False):
        path = path.rstrip('/') + '_ablate_prompt_task'
    elif getattr(args, 'ablate_prompt_stats', False):
        path = path.rstrip('/') + '_ablate_prompt_stats'
    if getattr(args, 'llm_layers', 32) == 8:
        path = path.rstrip('/') + '_ablate_llm8'
    if getattr(args, 'dropout', 0.1) == 0.15:
        path = path.rstrip('/') + '_dropout015'
    elif getattr(args, 'dropout', 0.1) == 0.05:
        path = path.rstrip('/') + '_dropout005'
    if getattr(args, 'weight_decay', 0.0) > 0:
        path = path.rstrip('/') + '_ablate_weight_decay'
    if not os.path.exists(path) and accelerator.is_main_process:
        os.makedirs(path)
    accelerator.wait_for_everyone()

    # 两阶段训练 Stage 2：从 Stage 1 的 checkpoint 加载权重后再训练
    load_ckpt_dir = getattr(args, 'load_ckpt_dir', None) or getattr(args, 'load_ckpt_dir', '')
    stage_input_checkpoint = None
    if load_ckpt_dir and load_ckpt_dir.strip():
        load_ckpt_dir = load_ckpt_dir.strip()
        ckpt_file = os.path.join(load_ckpt_dir, 'checkpoint')
        stage_input_checkpoint = os.path.abspath(ckpt_file)
        if os.path.isfile(ckpt_file):
            state = torch.load(ckpt_file, map_location='cpu')
            if getattr(args, 'strict_load_ckpt', False):
                model.load_state_dict(state, strict=True)
                if accelerator.is_main_process:
                    accelerator.print(
                        '[Stage 2] Strict checkpoint load succeeded: {}'.format(
                            stage_input_checkpoint
                        )
                    )
            else:
                model_state = model.state_dict()
                compatible_state = {}
                skipped_shape = []
                for key, value in state.items():
                    if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
                        skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
                        continue
                    compatible_state[key] = value
                # Compatibility mode is retained for historical runs whose
                # transformer-library buffers changed across versions.
                missing, unexpected = model.load_state_dict(compatible_state, strict=False)
                if accelerator.is_main_process:
                    accelerator.print('[Stage 2] Compatibility checkpoint load: {}'.format(stage_input_checkpoint))
                    if skipped_shape:
                        accelerator.print('[Stage 2] Warning: skipped shape-mismatched keys (first 20):')
                        for key, old_shape, new_shape in skipped_shape[:20]:
                            accelerator.print('  {}: ckpt {} -> model {}'.format(key, old_shape, new_shape))
                        if len(skipped_shape) > 20:
                            accelerator.print('  ... {} more'.format(len(skipped_shape) - 20))
                    if missing:
                        accelerator.print('[Stage 2] Warning: missing keys (ignored):', missing)
                    if unexpected:
                        accelerator.print('[Stage 2] Warning: unexpected keys (ignored):', unexpected)
        else:
            raise FileNotFoundError('--load_ckpt_dir 下未找到 checkpoint 文件: {}'.format(ckpt_file))

    write_reproducibility_manifest(
        path,
        phase='initialized',
        stage_input_checkpoint=stage_input_checkpoint,
    )
    accelerator.wait_for_everyone()

    if getattr(args, 'freeze_except_history_gate', False):
        trainable_names = []
        for name, p in model.named_parameters():
            keep = (
                'history_fusion_mlp' in name
                or 'state_event_mlp' in name
                or 'expected_loss_head' in name
                or 'window_occurrence_head' in name
            )
            p.requires_grad = bool(keep)
            if keep:
                trainable_names.append(name)
        if accelerator.is_local_main_process:
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            accelerator.print('[Freeze] Only history/state gate residual branch is trainable: {} params'.format(trainable_count))
            if not trainable_names:
                accelerator.print('[Freeze] Warning: no history/state gate parameters found. Did you pass --history_gate_fusion or --state_event_gate?')

    if getattr(args, 'freeze_state_event_gate', False):
        if not getattr(args, 'state_event_gate', False):
            raise ValueError(
                '--freeze_state_event_gate requires --state_event_gate'
            )
        frozen_state_event_names = []
        for name, p in model.named_parameters():
            if 'state_event_mlp' in name:
                p.requires_grad = False
                frozen_state_event_names.append(name)
        frozen_count = sum(
            p.numel()
            for name, p in model.named_parameters()
            if 'state_event_mlp' in name
        )
        if len(frozen_state_event_names) != 4 or frozen_count != 10400:
            raise RuntimeError(
                '--freeze_state_event_gate expected exactly four '
                'state_event_mlp tensors / 10,400 parameters, found '
                '{} tensors / {} parameters'.format(
                    len(frozen_state_event_names), frozen_count
                )
            )
        if accelerator.is_local_main_process:
            accelerator.print(
                '[Freeze] Base-stage state-event residual held fixed at its '
                'zero-output initialization: {} params'.format(frozen_count)
            )

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    reprog_lr_scale = getattr(args, 'reprog_lr_scale', 1.0)
    if reprog_lr_scale != 1.0 and args.model == 'TimeLLM' and not getattr(args, 'ablate_reprogramming', False):
        reprog_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'reprogramming_layer' in name or 'mapping_layer' in name or 'prototype_embeddings' in name:
                reprog_params.append(p)
            else:
                other_params.append(p)
        if reprog_params:
            model_optim = optim.Adam([
                {'params': other_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
                {'params': reprog_params, 'lr': args.learning_rate * reprog_lr_scale, 'weight_decay': args.weight_decay},
            ])
            accelerator.print('[Optim] 重编程层+prototype interface 使用 {:.0f}x 学习率 (lr={:.6f})'.format(reprog_lr_scale, args.learning_rate * reprog_lr_scale))
        else:
            model_optim = optim.Adam(trained_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        max_lr = args.learning_rate
        if reprog_lr_scale != 1.0 and args.model == 'TimeLLM' and not getattr(args, 'ablate_reprogramming', False) and len(getattr(model_optim, 'param_groups', [])) == 2:
            max_lr = [args.learning_rate, args.learning_rate * reprog_lr_scale]
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=max_lr)

    gated_models = ('TimeLLM', 'TimeLLM_TransformerOnly', 'DLinearGate', 'AutoformerGate', 'iTransformerGate')
    use_aux = getattr(args, 'use_aux_loss', False) and args.model in gated_models
    if args.loss in ('JointMaskedAux', 'JointMaskedAuxMAE') and args.model == 'TimeLLM_TransformerOnly':
        use_aux = True
    aux_w = float(getattr(args, 'aux_loss_weight', 0.2))
    if args.loss == 'JointMaskedAux' or args.loss == 'JointMaskedAuxMAE':
        if not use_aux:
            raise ValueError('--loss JointMaskedAux/JointMaskedAuxMAE 需同时开启 --use_aux_loss（且 model 需带辅助 gate）')
        use_raw_mask = not getattr(args, 'ablate_no_rmgm', False)
        use_mae_num = (args.loss == 'JointMaskedAuxMAE')
        mask_zero_w = float(getattr(args, 'mask_zero_weight', 0.0))
        num_scale = float(getattr(args, 'num_loss_scale', 1.0))
        criterion = JointMaskedMSEAuxBCE(
            lambda_weight=aux_w,
            use_raw_mask=use_raw_mask,
            use_mae_for_num=use_mae_num,
            mask_zero_weight=mask_zero_w,
            num_loss_scale=num_scale,
            aux_loss_type=getattr(args, 'aux_loss_type', 'bce'),
            aux_channel_balance=getattr(args, 'aux_channel_balance', False),
            aux_train_channel_index=getattr(args, 'aux_train_channel_index', -1),
            aux_pos_weight=getattr(args, 'aux_pos_weight', 1.0),
            aux_neg_weight=getattr(args, 'aux_neg_weight', 1.0),
            aux_focal_gamma=getattr(args, 'aux_focal_gamma', 2.0),
            aux_focal_gamma_neg=getattr(args, 'aux_focal_gamma_neg', 2.0),
            final_loss_weight=getattr(args, 'final_loss_weight', 0.0),
            final_loss_type=getattr(args, 'final_loss_type', 'mae'),
            final_huber_delta=getattr(args, 'final_huber_delta', 1.0),
            zero_magnitude_weight=getattr(args, 'zero_magnitude_weight', 0.0),
            zero_magnitude_loss_type=getattr(args, 'zero_magnitude_loss_type', 'mae'),
            zero_huber_delta=getattr(args, 'zero_huber_delta', 1.0),
            aux_rank_loss_weight=getattr(args, 'aux_rank_loss_weight', 0.0),
            aux_rank_margin=getattr(args, 'aux_rank_margin', 0.0),
            aux_rank_max_pairs=getattr(args, 'aux_rank_max_pairs', 4096),
            expected_loss_weight=getattr(args, 'expected_loss_weight', 0.0),
            expected_loss_type=getattr(args, 'expected_loss_type', 'huber'),
            expected_loss_huber_delta=getattr(args, 'expected_loss_huber_delta', 1.0),
            expected_loss_decision_weight=getattr(args, 'expected_loss_decision_weight', 0.0),
            expected_loss_decision_margin=getattr(args, 'expected_loss_decision_margin', 0.0),
            expected_loss_decision_pos_weight=getattr(args, 'expected_loss_decision_pos_weight', 1.0),
            expected_loss_gap_weight=getattr(args, 'expected_loss_gap_weight', 1.0),
            aux_teacher_distill_weight=getattr(args, 'chronos_teacher_distill_weight', 0.0),
            aux_teacher_distill_type=getattr(args, 'chronos_teacher_distill_type', 'bce'),
            aux_teacher_huber_delta=getattr(args, 'chronos_teacher_huber_delta', 1.0),
            window_aux_loss_weight=getattr(args, 'window_aux_loss_weight', 0.0),
            window_horizons=getattr(args, 'window_horizons', '7,14,30,48'),
            window_aux_pos_weight=getattr(args, 'window_aux_pos_weight', 1.0),
            window_aux_neg_weight=getattr(args, 'window_aux_neg_weight', 1.0),
        )
        criterion_vali = nn.MSELoss()
        if accelerator.is_local_main_process:
            accelerator.print('[GateLoss] aux_loss_type={} | channel_balance={} | train_channel={} | pos_weight={} | neg_weight={} | focal_gamma_pos={} | focal_gamma_neg={}'.format(
                getattr(args, 'aux_loss_type', 'bce'),
                bool(getattr(args, 'aux_channel_balance', False)),
                getattr(args, 'aux_train_channel_index', -1),
                getattr(args, 'aux_pos_weight', 1.0),
                getattr(args, 'aux_neg_weight', 1.0),
                getattr(args, 'aux_focal_gamma', 2.0),
                getattr(args, 'aux_focal_gamma_neg', 2.0),
            ))
            if getattr(args, 'aux_rank_loss_weight', 0.0) > 0:
                accelerator.print('[GateRankLoss] weight={} margin={} max_pairs={}'.format(
                    getattr(args, 'aux_rank_loss_weight', 0.0),
                    getattr(args, 'aux_rank_margin', 0.0),
                    getattr(args, 'aux_rank_max_pairs', 4096),
                ))
            accelerator.print('[DecisionAware] final_loss_weight={} final_loss_type={} | zero_magnitude_weight={} zero_magnitude_loss_type={}'.format(
                getattr(args, 'final_loss_weight', 0.0),
                getattr(args, 'final_loss_type', 'mae'),
                getattr(args, 'zero_magnitude_weight', 0.0),
                getattr(args, 'zero_magnitude_loss_type', 'mae'),
            ))
            if (
                getattr(args, 'expected_loss_head', False)
                or getattr(args, 'expected_loss_weight', 0.0) > 0
                or getattr(args, 'expected_loss_decision_weight', 0.0) > 0
            ):
                accelerator.print('[ExpectedLossHead] enabled={} cost_weight={} type={} huber_delta={} | decision_weight={} margin={} pos_weight={} gap_weight={}'.format(
                    bool(getattr(args, 'expected_loss_head', False)),
                    getattr(args, 'expected_loss_weight', 0.0),
                    getattr(args, 'expected_loss_type', 'huber'),
                    getattr(args, 'expected_loss_huber_delta', 1.0),
                    getattr(args, 'expected_loss_decision_weight', 0.0),
                    getattr(args, 'expected_loss_decision_margin', 0.0),
                    getattr(args, 'expected_loss_decision_pos_weight', 1.0),
                    getattr(args, 'expected_loss_gap_weight', 1.0),
                ))
            if getattr(args, 'history_delta_anchor_weight', 0.0) > 0 or getattr(args, 'history_delta_mean_weight', 0.0) > 0:
                accelerator.print('[HistoryDeltaAnchor] anchor_weight={} mean_weight={} anchor_margin={}'.format(
                    getattr(args, 'history_delta_anchor_weight', 0.0),
                    getattr(args, 'history_delta_mean_weight', 0.0),
                    getattr(args, 'history_delta_anchor_margin', 0.10),
                ))
            if getattr(args, 'chronos_teacher_distill_weight', 0.0) > 0:
                accelerator.print('[ChronosTeacherDistill] weight={} type={} temp={} path={}'.format(
                    getattr(args, 'chronos_teacher_distill_weight', 0.0),
                    getattr(args, 'chronos_teacher_distill_type', 'bce'),
                    getattr(args, 'chronos_teacher_temp', 2.0),
                    getattr(args, 'chronos_teacher_pred_path', ''),
                ))
            if getattr(args, 'window_occurrence_head', False) or getattr(args, 'window_aux_loss_weight', 0.0) > 0:
                accelerator.print('[WindowOccurrenceHead] enabled={} horizons={} loss_weight={} pos_weight={} neg_weight={}'.format(
                    bool(getattr(args, 'window_occurrence_head', False)),
                    getattr(args, 'window_horizons', '7,14,30,48'),
                    getattr(args, 'window_aux_loss_weight', 0.0),
                    getattr(args, 'window_aux_pos_weight', 1.0),
                    getattr(args, 'window_aux_neg_weight', 1.0),
                ))
        if num_scale != 1.0 and accelerator.is_local_main_process:
            accelerator.print('[JointMaskedMSEAuxBCE] 回归损失放大 {}x，总损失 = {}*L_reg + {}*L_aux'.format(num_scale, num_scale, aux_w))
        if use_mae_num:
            if mask_zero_w > 0:
                accelerator.print('[JointMaskedMSEAuxBCE] 数值头 Masked MAE + 软掩码(0值日权重={}) + 辅助头 BCE, lambda = {}'.format(mask_zero_w, aux_w))
            else:
                accelerator.print('[JointMaskedMSEAuxBCE] 数值头 Masked MAE（减轻平稳）+ 辅助头 BCE, lambda = {}'.format(aux_w))
        elif use_raw_mask:
            if mask_zero_w > 0:
                accelerator.print('[JointMaskedMSEAuxBCE] 数值头 Masked MSE + 软掩码(0值日权重={}) + 辅助头 BCE, lambda = {}'.format(mask_zero_w, aux_w))
            else:
                accelerator.print('[JointMaskedMSEAuxBCE] 数值头纯 Masked MSE + 辅助头 BCE (两路分道扬镳，推理时用 C_aux 裁决置零), lambda = {}'.format(aux_w))
        else:
            accelerator.print('[JointMaskedMSEAuxBCE] 消融 w/o RMGM: 数值头 mask 全 1（全部算 MSE）+ 辅助头 BCE，推理仍保留门控, lambda = {}'.format(aux_w))
    elif args.loss == 'ZeroInflated':
        criterion = ZeroInflatedLoss(zero_weight=args.zero_weight)
        criterion_vali = criterion
        accelerator.print('[ZeroInflatedLoss] zero_weight = {} (对目标为0的样本的损失权重)'.format(args.zero_weight))
    elif args.loss == 'MaskedMSE':
        criterion = MaskedMSE()
        criterion_vali = nn.MSELoss()
        accelerator.print('[MaskedMSE] 仅数值头，有货日 Masked MSE（与 JointMaskedAux 数值头部分一致），无辅助头')
    elif args.loss == 'MaskedMAE':
        use_raw_mask_mae = not getattr(args, 'ablate_no_rmgm', False)
        criterion = MaskedMAE(use_raw_mask=use_raw_mask_mae)
        criterion_vali = nn.MSELoss()
        accelerator.print('[MaskedMAE] 仅数值头，有货日 Masked MAE（与 JointMaskedAuxMAE 数值头部分一致），无辅助头；use_raw_mask={}'.format(use_raw_mask_mae))
    else:
        criterion = nn.MSELoss()
        criterion_vali = criterion
    mae_metric = nn.L1Loss()
    if use_aux and args.loss not in ('JointMaskedAux', 'JointMaskedAuxMAE'):
        criterion_ce = nn.CrossEntropyLoss()
        accelerator.print('[AuxLoss] enabled, aux_loss_weight = {}'.format(aux_w))

    if test_loader is None:
        train_loader, vali_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, model, model_optim, scheduler)
    else:
        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    snapshot_epochs = set()
    if getattr(args, 'save_epoch_snapshots', False):
        for item in str(getattr(args, 'snapshot_epochs', '') or '').split(','):
            item = item.strip()
            if item:
                snapshot_epochs.add(int(item))

    def save_epoch_snapshot(epoch_number):
        if not getattr(args, 'save_epoch_snapshots', False):
            return
        if int(epoch_number) not in snapshot_epochs:
            return
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            snap_dir = os.path.join(path, 'epoch_snapshots', 'epoch_{:03d}'.format(int(epoch_number)))
            os.makedirs(snap_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            state = {k: v.detach().cpu() for k, v in unwrapped.state_dict().items()}
            torch.save(state, os.path.join(snap_dir, 'checkpoint'))
            accelerator.print('[Snapshot] Saved epoch {} checkpoint: {}'.format(epoch_number, snap_dir))
        accelerator.wait_for_everyone()

    save_epoch_snapshot(0)

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, batch in tqdm(enumerate(train_loader)):
            batch_sample_ids = None
            if len(batch) == 6:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_feat_ids, batch_sample_ids = batch
            elif len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_feat_ids = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_feat_ids = None
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            f_dim = -1 if args.features == 'MS' else 0
            batch_y_future = batch_y[:, -args.pred_len:, f_dim:]
            # Recorded zeros become negative after standardization; recover the
            # source scale before constructing event labels.
            # 辅助标签与 Masked MSE 的 mask 必须用「原始量纲」判有/无货。独立通道时 batch_y_future 为 (B,pred_len,1)，scaler 按 4 列 fit，需按通道逆变换
            raw_batch_y_future = None
            zero_batch_y_future = None
            _ds = getattr(train_loader, 'dataset', None)
            if _ds is not None and getattr(_ds, 'scale', False) and hasattr(_ds, 'scaler'):
                batch_size, forecast_length, n_channels = batch_y_future.shape
                scale_ = _ds.scaler.scale_
                mean_ = _ds.scaler.mean_
                zero_norm_ = (-mean_ / scale_).astype(np.float32)
                if n_channels == 1 and batch_feat_ids is not None:
                    # 独立通道：(B, pred_len, 1)，每样本对应一通道，用该通道的 mean_/scale_ 逆变换
                    raw_batch_y_future = torch.empty(batch_size, forecast_length, 1, dtype=torch.float32, device=batch_y_future.device)
                    zero_batch_y_future = torch.empty(batch_size, forecast_length, 1, dtype=torch.float32, device=batch_y_future.device)
                    for bi in range(batch_size):
                        c = int(batch_feat_ids[bi].item() if hasattr(batch_feat_ids[bi], 'item') else batch_feat_ids[bi])
                        raw_batch_y_future[bi, :, 0] = torch.from_numpy(
                            (batch_y_future[bi, :, 0].cpu().numpy() * scale_[c]) + mean_[c]
                        ).float().to(batch_y_future.device)
                        zero_batch_y_future[bi, :, 0] = float(zero_norm_[c])
                elif n_channels == 1 and len(zero_norm_) >= 1:
                    raw_batch_y_future = (batch_y_future * float(scale_[0])) + float(mean_[0])
                    zero_batch_y_future = torch.full_like(batch_y_future, float(zero_norm_[0]))
                elif n_channels == scale_.shape[0]:
                    flat = batch_y_future.detach().cpu().numpy().reshape(-1, n_channels)
                    raw_flat = _ds.inverse_transform(flat)
                    raw_batch_y_future = torch.from_numpy(
                        raw_flat.reshape(batch_size, forecast_length, n_channels).astype(np.float32)
                    ).to(batch_y_future.device)
                    zero_vec = torch.from_numpy(zero_norm_).float().to(batch_y_future.device).view(1, 1, n_channels)
                    zero_batch_y_future = zero_vec.expand(batch_size, forecast_length, n_channels)
            if zero_batch_y_future is None and (
                float(getattr(args, 'final_loss_weight', 0.0)) > 0
                or float(getattr(args, 'zero_magnitude_weight', 0.0)) > 0
            ):
                zero_batch_y_future = torch.zeros_like(batch_y_future)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if use_aux and (not args.output_attention):
                        if args.model == 'TimeLLM':
                            out_pack = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_aux_repr=True, feat_ids=batch_feat_ids)
                        else:
                            out_pack = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_aux_repr=True)
                        if isinstance(out_pack, tuple):
                            outputs, extra = out_pack
                            aux_logits = extra.get('aux_logits')
                        else:
                            outputs = out_pack
                            aux_logits = None
                    else:
                        if args.output_attention:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        aux_logits = None

                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y_future.to(accelerator.device)
                    aux_logits = attach_chronos_teacher_to_aux(
                        aux_logits,
                        chronos_teacher_score,
                        batch_feat_ids,
                        batch_sample_ids,
                        batch_y_future,
                    )
                    if isinstance(aux_logits, dict) and batch_feat_ids is not None:
                        aux_logits['channel_ids'] = batch_feat_ids
                    if args.loss in ('JointMaskedAux', 'JointMaskedAuxMAE') and use_aux and aux_logits is not None:
                        _y_aux = raw_batch_y_future if raw_batch_y_future is not None else batch_y_future
                        aux_labels = compute_derived_auxiliary_labels(_y_aux, enc_in=args.enc_in, point_to_point=True)
                        loss = criterion(
                            outputs,
                            aux_logits,
                            batch_y,
                            aux_labels['has_shipment'],
                            raw_targets_num=raw_batch_y_future,
                            zero_targets_num=zero_batch_y_future,
                        )
                    else:
                        if args.loss in ('MaskedMSE', 'MaskedMAE'):
                            loss = criterion(outputs, batch_y, raw_targets_num=raw_batch_y_future)
                        else:
                            loss = criterion(outputs, batch_y)
                        if use_aux and (aux_logits is not None):
                            _y_aux = raw_batch_y_future if raw_batch_y_future is not None else batch_y_future
                            aux_labels = compute_derived_auxiliary_labels(_y_aux, enc_in=args.enc_in)
                            aux_log = aux_logits['has_shipment']
                            if aux_log.dim() == 3:
                                aux_log = aux_log.mean(dim=1)
                            lab = aux_labels['has_shipment']
                            if lab.dim() == 1 and aux_log.size(0) == lab.size(0) * args.enc_in:
                                lab = lab.repeat_interleave(args.enc_in, dim=0)
                            aux_loss = criterion_ce(aux_log, lab)
                            loss = loss + aux_w * aux_loss
                    loss = add_history_delta_regularization(args, aux_logits, loss)
                    train_loss.append(loss.item())
            else:
                f_dim = -1 if args.features == 'MS' else 0
                batch_y_future = batch_y[:, -args.pred_len:, f_dim:]

                if use_aux and (not args.output_attention):
                    if args.model == 'TimeLLM':
                        out_pack = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_aux_repr=True, feat_ids=batch_feat_ids)
                    else:
                        out_pack = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_aux_repr=True)
                    if isinstance(out_pack, tuple):
                        outputs, extra = out_pack
                        aux_logits = extra.get('aux_logits')
                    else:
                        outputs = out_pack
                        aux_logits = None
                else:
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    aux_logits = None

                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y_future
                aux_logits = attach_chronos_teacher_to_aux(
                    aux_logits,
                    chronos_teacher_score,
                    batch_feat_ids,
                    batch_sample_ids,
                    batch_y_future,
                )
                if isinstance(aux_logits, dict) and batch_feat_ids is not None:
                    aux_logits['channel_ids'] = batch_feat_ids
                if args.loss in ('JointMaskedAux', 'JointMaskedAuxMAE') and use_aux and aux_logits is not None:
                    _y_aux = raw_batch_y_future if raw_batch_y_future is not None else batch_y_future
                    aux_labels = compute_derived_auxiliary_labels(_y_aux, enc_in=args.enc_in, point_to_point=True)
                    loss = criterion(
                        outputs,
                        aux_logits,
                        batch_y,
                        aux_labels['has_shipment'],
                        raw_targets_num=raw_batch_y_future,
                        zero_targets_num=zero_batch_y_future,
                    )
                else:
                    if args.loss in ('MaskedMSE', 'MaskedMAE'):
                        loss = criterion(outputs, batch_y, raw_targets_num=raw_batch_y_future)
                    else:
                        loss = criterion(outputs, batch_y)
                    if use_aux and (aux_logits is not None):
                        _y_aux = raw_batch_y_future if raw_batch_y_future is not None else batch_y_future
                        aux_labels = compute_derived_auxiliary_labels(_y_aux, enc_in=args.enc_in)
                        aux_log = aux_logits['has_shipment']
                        if aux_log.dim() == 3:
                            aux_log = aux_log.mean(dim=1)
                        lab = aux_labels['has_shipment']
                        if lab.dim() == 1 and aux_log.size(0) == lab.size(0) * args.enc_in:
                            lab = lab.repeat_interleave(args.enc_in, dim=0)
                        aux_loss = criterion_ce(aux_log, lab)
                        loss = loss + aux_w * aux_loss
                loss = add_history_delta_regularization(args, aux_logits, loss)
                train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion_vali, mae_metric)
        if getattr(args, 'skip_test_during_training', False):
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} | Test evaluation skipped".format(
                    epoch + 1, train_loss, vali_loss))
        else:
            if test_data is None or test_loader is None:
                raise RuntimeError(
                    'Internal error: epoch-level test was requested but its loader was not constructed.'
                )
            test_loss, test_mae_loss = vali(
                args, accelerator, model, test_data, test_loader, criterion_vali, mae_metric)
            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} MAE Loss: {4:.7f}".format(
                    epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss))

        early_stopping(vali_loss, model, path)
        save_epoch_snapshot(epoch + 1)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

accelerator.wait_for_everyone()
# Optionally evaluate the validation-selected checkpoint on the test split.
ckpt_file = os.path.join(path, 'checkpoint')
if getattr(args, 'save_last_checkpoint', False):
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_local_main_process:
        torch.save(unwrapped.state_dict(), ckpt_file)
        accelerator.print('[Checkpoint] Saved last epoch checkpoint for final evaluation: {}'.format(ckpt_file))
    accelerator.wait_for_everyone()

# Legacy final-test reporting remains available, but when epoch-level testing
# was disabled its Dataset/DataLoader is intentionally created only now, after
# validation has locked the checkpoint.
if not getattr(args, 'skip_final_test', False) and test_loader is None:
    test_data, test_loader = data_provider(args, 'test')
    test_loader = accelerator.prepare(test_loader)

if os.path.exists(ckpt_file) and not getattr(args, 'skip_final_test', False):
    state = torch.load(ckpt_file, map_location=accelerator.device)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.load_state_dict(state, strict=True)
    test_loss_final, test_mae_final, test_rmse_final = vali_with_rmse(
        args, accelerator, model, test_loader, criterion_vali, mae_metric)
    if accelerator.is_local_main_process:
        accelerator.print('=' * 60)
        if getattr(args, 'save_last_checkpoint', False):
            accelerator.print('Last checkpoint on Test Set:')
        else:
            accelerator.print('Best checkpoint (by Vali Loss) on Test Set:')
        accelerator.print('  Test Loss: {:.7f}  |  MAE: {:.7f}  |  RMSE: {:.7f}'.format(
            test_loss_final, test_mae_final, test_rmse_final))
        accelerator.print('=' * 60)
elif os.path.exists(ckpt_file) and accelerator.is_local_main_process:
    accelerator.print('[Test hygiene] Final test evaluation skipped. Select the candidate on validation first.')

accelerator.wait_for_everyone()
write_reproducibility_manifest(
    path,
    phase='training_complete',
    stage_input_checkpoint=stage_input_checkpoint,
    output_checkpoint=ckpt_file,
    early_stopping=early_stopping,
)
accelerator.wait_for_everyone()
# 不再在训练结束后删除 checkpoints，保留 EarlyStopping 保存的最佳模型
