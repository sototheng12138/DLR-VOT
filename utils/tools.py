import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
import os
import tempfile
from collections import OrderedDict

from tqdm import tqdm

plt.switch_backend('agg')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def _update_state(self, val_loss):
        """Update validation state on one process and report whether to save."""
        score = -float(val_loss)
        should_save = False
        if self.best_score is None:
            self.best_score = score
            should_save = bool(self.save_mode)
            self.val_loss_min = float(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            should_save = bool(self.save_mode)
            self.counter = 0
            self.val_loss_min = float(val_loss)
        return should_save

    def _broadcast_state(self, should_save):
        """Broadcast rank-0 early-stop state to every distributed rank."""
        if self.accelerator is None:
            return bool(should_save)

        initialized = self.best_score is not None
        payload = torch.tensor(
            [
                1.0 if initialized else 0.0,
                float(self.best_score) if initialized else 0.0,
                float(self.counter),
                1.0 if self.early_stop else 0.0,
                float(self.val_loss_min) if np.isfinite(self.val_loss_min) else 0.0,
                1.0 if should_save else 0.0,
            ],
            dtype=torch.float64,
            device=self.accelerator.device,
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(payload, src=0)

        values = payload.detach().cpu().tolist()
        self.best_score = float(values[1]) if values[0] >= 0.5 else None
        self.counter = int(round(values[2]))
        self.early_stop = bool(values[3] >= 0.5)
        self.val_loss_min = float(values[4]) if values[0] >= 0.5 else np.inf
        return bool(values[5] >= 0.5)

    def __call__(self, val_loss, model, path):
        if self.accelerator is None:
            should_save = self._update_state(val_loss)
            if should_save:
                self.save_checkpoint(val_loss, model, path)
            return

        # Validation metrics are gathered before this call, but only global
        # rank 0 owns the model-selection state. Broadcasting the full state
        # keeps all ranks on the same epoch and prevents divergent early exits.
        self.accelerator.wait_for_everyone()
        should_save = False
        if self.accelerator.is_main_process:
            should_save = self._update_state(val_loss)
        should_save = self._broadcast_state(should_save)
        if should_save:
            self.save_checkpoint(val_loss, model, path)
        self.accelerator.wait_for_everyone()

    def save_checkpoint(self, val_loss, model, path):
        is_main_process = self.accelerator is None or self.accelerator.is_main_process
        if self.verbose and is_main_process:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss improved to {float(val_loss):.6f}. Saving model ...')
            else:
                print(
                    f'Validation loss improved to {float(val_loss):.6f}. Saving model ...')

        state_dict = None
        if self.accelerator is not None:
            # DeepSpeed state consolidation may be collective. Standard DDP
            # does not require that work on non-main ranks.
            if 'DEEPSPEED' in str(self.accelerator.distributed_type).upper():
                state_dict = self.accelerator.get_state_dict(model)
            elif self.accelerator.is_main_process:
                state_dict = self.accelerator.unwrap_model(model).state_dict()
        else:
            state_dict = model.state_dict()

        if not is_main_process:
            return

        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, 'checkpoint')
        fd, temporary_path = tempfile.mkstemp(
            prefix='.checkpoint.',
            suffix='.tmp',
            dir=path,
        )
        os.close(fd)
        try:
            cpu_state = OrderedDict(
                (
                    key,
                    value.detach().cpu() if torch.is_tensor(value) else value,
                )
                for key, value in state_dict.items()
            )
            if hasattr(state_dict, '_metadata'):
                cpu_state._metadata = state_dict._metadata
            torch.save(cpu_state, temporary_path)
            with open(temporary_path, 'rb') as handle:
                os.fsync(handle.fileno())
            os.replace(temporary_path, checkpoint_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(vali_loader)):
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()

            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            pred = outputs.detach()
            true = batch_y.detach()

            loss = criterion(pred, true)

            mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            total_mae_loss.append(mae_loss.item())

    total_loss = np.average(total_loss)
    total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss


def vali_with_rmse(args, accelerator, model, data_loader, criterion, mae_metric):
    """与 vali 相同，但多返回 RMSE，用于训练结束后对 best checkpoint 做一次测试集评估。"""
    total_loss = []
    total_mae = []
    total_rmse = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if len(batch) == 5:
                batch_x, batch_y, batch_x_mark, batch_y_mark, _ = batch
            else:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) if not args.output_attention else model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark) if not args.output_attention else model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
            pred = outputs.detach()
            true = batch_y.detach()
            total_loss.append(criterion(pred, true).item())
            total_mae.append(mae_metric(pred, true).item())
            total_rmse.append(torch.sqrt(torch.mean((pred - true) ** 2)).item())
    model.train()
    return np.average(total_loss), np.average(total_mae), np.average(total_rmse)
def load_content(args):
    explicit_prompt = str(
        getattr(args, 'prompt_bank_name', '') or ''
    ).strip()
    if explicit_prompt:
        file = explicit_prompt
    elif 'ETT' in args.data:
        file = 'ETT'
    elif getattr(args, 'model_comment', None) and args.model_comment not in ('none', ''):
        file = args.model_comment
    else:
        file = args.data
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompt_path = os.path.join(
        project_root, 'dataset', 'prompt_bank', '{0}.txt'.format(file)
    )
    if not os.path.exists(prompt_path):
        fallback_file = None
        if str(file).startswith('iron_stage2_linear_'):
            fallback_file = 'iron_stage2_linear'
        elif str(file).startswith('iron_stage1_linear_'):
            fallback_file = 'iron_stage1_linear'
        elif str(file).startswith('stage1_vocab_mapping_snapshots'):
            fallback_file = 'iron_stage1_linear'
        elif str(file).startswith('stage1_vocab_mapping_epoch'):
            fallback_file = 'iron_stage1_linear'
        elif str(file).startswith('ablate_no_reprogramming_stage1'):
            fallback_file = 'iron_stage1_linear'
        elif str(file).startswith('ablate_no_reprogramming_conversion'):
            fallback_file = 'iron_stage2_linear'
        elif str(file).startswith('ablate_no_reprogramming_stage2'):
            fallback_file = 'iron_stage2_linear'
        elif str(file).startswith('twostage_direct_stage2'):
            fallback_file = 'iron_stage2_linear'
        elif str(file).startswith('aux2_'):
            fallback_file = 'aux2_fewshot_from_proposed'
        if fallback_file is not None:
            fallback_path = os.path.join(
                project_root,
                'dataset',
                'prompt_bank',
                '{0}.txt'.format(fallback_file),
            )
            if os.path.exists(fallback_path):
                print('[PromptBank] {} not found; fallback to {}'.format(prompt_path, fallback_path))
                prompt_path = fallback_path
    args.resolved_prompt_bank_path = os.path.abspath(prompt_path)
    with open(prompt_path, 'r') as f:
        content = f.read()
    return content
