from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoConfig, AutoModel, AutoTokenizer
from layers.Embed import PatchEmbedding, MultiScalePatchEmbedding
import transformers
from layers.StandardNorm import Normalize

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class MLPFlattenHead(nn.Module):
    """Nonlinear magnitude head used by the paper configuration."""
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.fc1 = nn.Linear(nf, nf)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(head_dropout)
        self.fc2 = nn.Linear(nf, target_window)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.enc_in = configs.enc_in
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        if configs.llm_model == 'LLAMA':
            model_id = getattr(configs, 'llm_model_id', '') or 'huggyllama/llama-7b'
            self.llama_config = LlamaConfig.from_pretrained(model_id)
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            self.llm_config = self.llama_config
            if getattr(configs, 'llm_random_init', False):
                # 随机初始化 LLaMA 结构：同样层数与维度，但不加载预训练权重
                self.llm_model = LlamaModel(self.llama_config)
            else:
                try:
                    self.llm_model = LlamaModel.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        local_files_only=True,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
                except EnvironmentError:  # downloads model from HF is not already done
                    print("Local model files not found. Attempting to download...")
                    self.llm_model = LlamaModel.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        local_files_only=False,
                        config=self.llama_config,
                        # load_in_4bit=True
                    )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            model_id = getattr(configs, 'llm_model_id', '') or 'openai-community/gpt2'
            self.gpt2_config = GPT2Config.from_pretrained(model_id)

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_config = self.gpt2_config
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            model_id = getattr(configs, 'llm_model_id', '') or 'google-bert/bert-base-uncased'
            self.bert_config = BertConfig.from_pretrained(model_id)

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            self.llm_config = self.bert_config
            try:
                self.llm_model = BertModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'AUTO':
            model_id = getattr(configs, 'llm_model_id', '')
            if not model_id:
                raise ValueError('--llm_model AUTO requires --llm_model_id')
            self.auto_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            if hasattr(self.auto_config, 'num_hidden_layers'):
                self.auto_config.num_hidden_layers = configs.llm_layers
            self.auto_config.output_attentions = True
            self.auto_config.output_hidden_states = True
            self.llm_config = self.auto_config
            try:
                self.llm_model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.auto_config,
                )
            except EnvironmentError:
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModel.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.auto_config,
                )

            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=True,
                    use_fast=True,
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    local_files_only=False,
                    use_fast=True,
                )
        else:
            raise Exception('LLM model is not defined')

        config_hidden_size = getattr(self.llm_config, 'hidden_size', None) or getattr(self.llm_config, 'n_embd', None)
        if config_hidden_size is not None and int(self.d_llm) != int(config_hidden_size):
            raise ValueError(
                'llm_dim={} does not match {} hidden size={}. Please set --llm_dim {}.'.format(
                    self.d_llm, configs.llm_model, config_hidden_size, config_hidden_size
                )
            )

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'
        self.prompt_type = getattr(configs, 'prompt_type', 'full')  # full | short (ablation)

        self.dropout = nn.Dropout(configs.dropout)

        if getattr(configs, 'use_multiscale_patch', False):
            self.patch_embedding = MultiScalePatchEmbedding(
                configs.d_model, configs.seq_len, self.patch_len, self.stride, configs.dropout,
                scales=getattr(configs, 'multiscale_patch_scales', None))
        else:
            self.patch_embedding = PatchEmbedding(
                configs.d_model, self.patch_len, self.stride, configs.dropout)

        self.ablate_reprogramming = getattr(configs, 'ablate_reprogramming', False)
        if self.ablate_reprogramming:
            # Ablation: remove reprogramming and project patch features directly to the LLM dimension.
            self.reprogram_proj = nn.Linear(configs.d_model, self.d_llm)
        else:
            self.word_embeddings = self.llm_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
            self.num_tokens = int(getattr(configs, 'num_tokens', 1000))
            if self.num_tokens <= 0:
                raise ValueError('num_tokens must be positive')
            self.prototype_mode = getattr(configs, 'prototype_mode', 'vocab_mapping')
            if self.prototype_mode not in ('vocab_mapping', 'direct'):
                raise ValueError("prototype_mode must be 'vocab_mapping' or 'direct'")
            if self.prototype_mode == 'vocab_mapping':
                self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
            else:
                self.prototype_embeddings = nn.Parameter(torch.empty(self.num_tokens, self.d_llm))
                with torch.no_grad():
                    if self.num_tokens <= self.vocab_size:
                        # Start from real frozen LLM token embeddings so direct prototypes stay in the LLM embedding space.
                        idx = torch.linspace(0, self.vocab_size - 1, steps=self.num_tokens, device=self.word_embeddings.device).long()
                        self.prototype_embeddings.copy_(self.word_embeddings[idx].detach())
                    else:
                        emb = self.word_embeddings.detach().float()
                        nn.init.normal_(self.prototype_embeddings, mean=float(emb.mean()), std=float(emb.std()))
            reprogramming_d_keys = int(getattr(configs, 'reprogramming_d_keys', 0) or self.d_ff)
            if reprogramming_d_keys <= 0:
                raise ValueError('reprogramming_d_keys must be positive when provided')
            self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, reprogramming_d_keys, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if getattr(configs, 'regression_head_mlp', False):
                self.output_projection = MLPFlattenHead(configs.enc_in, self.head_nf, self.pred_len, head_dropout=configs.dropout)
            else:
                self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                     head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.print_prompt_once = getattr(configs, 'print_prompt_once', False)
        self.ablate_prompt = getattr(configs, 'ablate_prompt', False)
        self.ablate_prompt_description = getattr(configs, 'ablate_prompt_description', False)
        self.ablate_prompt_task = getattr(configs, 'ablate_prompt_task', False)
        self.ablate_prompt_stats = getattr(configs, 'ablate_prompt_stats', False)
        no_revin = getattr(configs, 'no_revin', False)
        self.normalize_layers = Normalize(configs.enc_in, affine=False, non_norm=no_revin)

        # 通道混合层：多元协同时在 reshape(B*N,T,1) 前对 (B,T,N) 做线性混合，使各通道获得其他通道信息
        self.channel_mixing = getattr(configs, 'channel_mixing', False)
        if self.channel_mixing and configs.enc_in > 1:
            self.channel_mixer = nn.Linear(configs.enc_in, configs.enc_in)
        else:
            self.channel_mixer = None

        # 辅助任务头：点对点 [B, Seq_Len, N]，不池化时间维，与数值头对齐后做逐点门控
        self.use_aux_loss = getattr(configs, 'use_aux_loss', False)
        self.history_gate_fusion = bool(getattr(configs, 'history_gate_fusion', False))
        self.state_event_gate = bool(getattr(configs, 'state_event_gate', False))
        self.expected_loss_head_enabled = bool(getattr(configs, 'expected_loss_head', False))
        self.window_occurrence_head_enabled = bool(getattr(configs, 'window_occurrence_head', False))
        raw_window_horizons = getattr(configs, 'window_horizons', '7,14,30,48')
        if isinstance(raw_window_horizons, str):
            self.window_horizons = [int(x) for x in raw_window_horizons.split(',') if x.strip()]
        else:
            self.window_horizons = [int(x) for x in raw_window_horizons]
        self.window_horizons = [h for h in self.window_horizons if 0 < h <= self.pred_len]
        if self.window_occurrence_head_enabled and not self.window_horizons:
            raise ValueError('window_occurrence_head requires at least one valid horizon <= pred_len')
        zero_norm_values = getattr(configs, 'zero_norm_values', []) or []
        if len(zero_norm_values) > 0:
            zero_norm_tensor = torch.tensor(zero_norm_values, dtype=torch.float32)
        else:
            zero_norm_tensor = torch.zeros(int(getattr(configs, 'enc_in', 1)), dtype=torch.float32)
        self.register_buffer('zero_norm_values', zero_norm_tensor, persistent=False)
        if self.use_aux_loss:
            # 输入 block 展平为 (B*N, head_nf)，输出 (B*N, pred_len, 2) -> 对应 (B, pred_len, N) 的 P(有发运)
            self.aux_has_shipment = nn.Linear(self.head_nf, self.pred_len * 2)
            if self.history_gate_fusion:
                self.history_feature_dim = 72
                hidden = int(getattr(configs, 'history_gate_dim', 32))
                self.history_fusion_mlp = nn.Sequential(
                    nn.Linear(self.head_nf + self.history_feature_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(hidden, self.pred_len * 2),
                )
                # Start exactly from the loaded LLM gate; the new branch learns a delta.
                nn.init.zeros_(self.history_fusion_mlp[-1].weight)
                nn.init.zeros_(self.history_fusion_mlp[-1].bias)
            if self.state_event_gate:
                self.state_event_feature_set = str(getattr(configs, 'state_event_feature_set', 'v1')).lower()
                if self.state_event_feature_set not in ('v1', 'v2'):
                    raise ValueError('state_event_feature_set must be v1 or v2')
                self.state_event_feature_dim = 64 if self.state_event_feature_set == 'v2' else 32
                hidden = int(getattr(configs, 'state_event_gate_dim', 32))
                self.state_event_mlp = nn.Sequential(
                    nn.Linear(self.state_event_feature_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(hidden, self.pred_len * 2),
                )
                # Residual branch: start from the loaded LLM gate and learn only event-state corrections.
                nn.init.zeros_(self.state_event_mlp[-1].weight)
                nn.init.zeros_(self.state_event_mlp[-1].bias)
            if self.expected_loss_head_enabled:
                hidden = int(getattr(configs, 'expected_loss_head_dim', 64))
                expected_in_dim = self.head_nf
                if self.state_event_gate:
                    expected_in_dim += self.state_event_feature_dim
                self.expected_loss_head = nn.Sequential(
                    nn.Linear(expected_in_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(hidden, self.pred_len * 2),
                )
            if self.window_occurrence_head_enabled:
                hidden = int(getattr(configs, 'window_occurrence_head_dim', 64))
                self.window_occurrence_head = nn.Sequential(
                    nn.Linear(self.head_nf, hidden),
                    nn.GELU(),
                    nn.Dropout(configs.dropout),
                    nn.Linear(hidden, len(self.window_horizons) * 2),
                )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, return_reprogramming_attention=False, return_aux_repr=False, feat_ids=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, return_reprogramming_attention=return_reprogramming_attention, return_aux_repr=return_aux_repr, feat_ids=feat_ids)
            if isinstance(dec_out, tuple):
                dec_out, extra = dec_out
                return dec_out[:, -self.pred_len:, :], extra
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, return_reprogramming_attention=False, return_aux_repr=False, feat_ids=None):
        reprogramming_attn = None
        aux_logits = None

        history_for_gate = x_enc.float()
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        if self.channel_mixer is not None:
            # (B, T, N) -> 通道维线性混合；仅当 channel_mixer 为 bf16 时转换，避免 eval 时 float32 权重报错
            if x_enc.is_cuda:
                first_w = next(self.channel_mixer.parameters())
                if first_w.dtype == torch.bfloat16:
                    x_enc = x_enc.to(torch.bfloat16)
            x_enc = self.channel_mixer(x_enc)
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        if self.ablate_prompt:
            # 消融：不要 prompt，仅用重编程后的 patch 作为 LLM 输入
            prompt_embeddings = None
        else:
            # 按需包含：数据集背景、任务指令、输入统计（三部分可单独消融）
            need_stats = not self.ablate_prompt_stats and self.prompt_type != 'short'
            if need_stats:
                # median/min/max 等 reduction 在 BFloat16 上部分未实现，统计时用 float
                x_for_stats = x_enc.float() if x_enc.dtype == torch.bfloat16 else x_enc
                min_values = torch.min(x_for_stats, dim=1)[0]
                max_values = torch.max(x_for_stats, dim=1)[0]
                medians = torch.median(x_for_stats, dim=1).values
                lags = self.calcute_lags(x_for_stats)
                trends = x_for_stats.diff(dim=1).sum(dim=1)

            prompt = []
            for b in range(x_enc.shape[0]):
                parts = []
                if not self.ablate_prompt_description:
                    parts.append(f"Dataset description: {self.description}")
                if not self.ablate_prompt_task:
                    parts.append(
                        f"Task description: forecast the next {self.pred_len} steps given the previous {self.seq_len} steps information; "
                    )
                if need_stats:
                    min_values_str = str(min_values[b].tolist()[0])
                    max_values_str = str(max_values[b].tolist()[0])
                    median_values_str = str(medians[b].tolist()[0])
                    lags_values_str = str(lags[b].tolist())
                    parts.append(
                        "Input statistics: "
                        f"min value {min_values_str}, "
                        f"max value {max_values_str}, "
                        f"median value {median_values_str}, "
                        f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                        f"top 5 lags are : {lags_values_str}"
                    )
                prompt.append("<|start_prompt|>" + " ".join(parts) + "<|<end_prompt>|>")

        prompt_sample = None
        if getattr(self, 'print_prompt_once', False) and not getattr(self, '_prompt_printed', False) and len(prompt) > 0:
            prompt_sample = prompt[0] if isinstance(prompt[0], str) else str(prompt[0])
            self._prompt_printed = True

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        if not self.ablate_prompt:
            prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
            prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        # 仅当 patch_embedding 权重为 bf16 时转换输入，否则 Conv1d 会报 dtype 不匹配（eval 时模型为 float32）
        if x_enc.is_cuda:
            first_param = next(self.patch_embedding.parameters())
            if first_param.dtype == torch.bfloat16:
                x_enc = x_enc.to(torch.bfloat16)
        enc_out, n_vars = self.patch_embedding(x_enc)
        if self.ablate_reprogramming:
            enc_out = self.reprogram_proj(enc_out)
        else:
            if self.prototype_mode == 'direct':
                source_embeddings = self.prototype_embeddings
            else:
                source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
            if return_reprogramming_attention:
                enc_out, reprogramming_attn = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings, return_attention=True)
            else:
                enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
        if self.ablate_prompt:
            llama_enc_out = enc_out
        else:
            llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        block = dec_out[:, :, :, -self.patch_nums:]
        if return_aux_repr and self.use_aux_loss and hasattr(self, 'aux_has_shipment'):
            # 保留时间信息：展平 (B*N, n_vars, d_ff, patch_nums) -> (B*N, head_nf)，再映射到 (B*N, pred_len, 2)
            repr_flat = block.mean(dim=1).reshape(block.size(0), -1)
            aux_logits_base_flat = self.aux_has_shipment(repr_flat)
            aux_logits_flat = aux_logits_base_flat
            aux_logits_delta_flat = None
            aux_logits_state_flat = None
            state_feat_for_heads = None
            if self.history_gate_fusion and hasattr(self, 'history_fusion_mlp'):
                hist_feat = self._history_gate_features(history_for_gate)
                if hist_feat.size(0) != repr_flat.size(0):
                    if hist_feat.size(0) == 1:
                        hist_feat = hist_feat.expand(repr_flat.size(0), -1)
                    else:
                        raise ValueError(
                            'history feature batch {} does not match aux repr batch {}'.format(
                                hist_feat.size(0), repr_flat.size(0)
                            )
                        )
                fusion_input = torch.cat([repr_flat, hist_feat.to(device=repr_flat.device, dtype=repr_flat.dtype)], dim=-1)
                aux_logits_delta_flat = self.history_fusion_mlp(fusion_input)
                aux_logits_flat = aux_logits_base_flat + aux_logits_delta_flat
            if self.state_event_gate and hasattr(self, 'state_event_mlp'):
                state_feat = self._state_event_features(history_for_gate, feat_ids=feat_ids)
                state_feat = state_feat.to(device=repr_flat.device, dtype=repr_flat.dtype)
                if state_feat.size(0) != repr_flat.size(0):
                    raise ValueError(
                        'state-event feature batch {} does not match aux repr batch {}'.format(
                            state_feat.size(0), repr_flat.size(0)
                        )
                    )
                state_feat_for_heads = state_feat
                aux_logits_state_flat = self.state_event_mlp(state_feat)
                aux_logits_flat = aux_logits_flat + aux_logits_state_flat
            aux_logits = {'has_shipment': aux_logits_flat.view(block.size(0), self.pred_len, 2)}
            if self.window_occurrence_head_enabled and hasattr(self, 'window_occurrence_head'):
                window_occurrence_flat = self.window_occurrence_head(repr_flat)
                aux_logits['window_occurrence'] = window_occurrence_flat.view(
                    block.size(0), len(self.window_horizons), 2
                )
            if self.expected_loss_head_enabled and hasattr(self, 'expected_loss_head'):
                expected_input = repr_flat
                if self.state_event_gate and state_feat_for_heads is not None:
                    expected_input = torch.cat([repr_flat, state_feat_for_heads], dim=-1)
                expected_loss_flat = self.expected_loss_head(expected_input)
                # Channel order: [..., 0] = predicted open-loss, [..., 1] = predicted close-loss.
                aux_logits['expected_loss'] = expected_loss_flat.view(block.size(0), self.pred_len, 2)
            if aux_logits_delta_flat is not None:
                aux_logits['has_shipment_base'] = aux_logits_base_flat.view(block.size(0), self.pred_len, 2)
                aux_logits['has_shipment_delta'] = aux_logits_delta_flat.view(block.size(0), self.pred_len, 2)
            if aux_logits_state_flat is not None:
                aux_logits['has_shipment_base'] = aux_logits_base_flat.view(block.size(0), self.pred_len, 2)
                aux_logits['has_shipment_state_delta'] = aux_logits_state_flat.view(block.size(0), self.pred_len, 2)
        dec_out = self.output_projection(block)
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        extra = {}
        if return_reprogramming_attention and reprogramming_attn is not None:
            extra['reprogramming_attn'] = reprogramming_attn
        if return_aux_repr and aux_logits is not None:
            extra['aux_logits'] = aux_logits
        if prompt_sample is not None:
            extra['prompt_sample'] = prompt_sample
        if extra:
            return dec_out, extra
        return dec_out

    def _agg_channel_stats(self, stat):
        stat = stat.float()
        mean = stat.mean(dim=1)
        std = stat.std(dim=1, unbiased=False)
        return mean, std

    def _history_gate_features(self, x_hist):
        """Compact history features for the fusion gate.

        The input is the dataset-scaled encoder history before RevIN. Features
        are aggregated over channels so the auxiliary gate keeps the historical
        output shape used by the existing independent-channel pipeline.
        """
        x = x_hist.float()
        feats = []
        windows = (7, 14, 30, 60, min(self.seq_len, x.size(1)))
        for win in windows:
            cur = x[:, -min(win, x.size(1)):, :]
            stats = (
                cur.mean(dim=1),
                cur.std(dim=1, unbiased=False),
                cur.max(dim=1).values,
                cur.min(dim=1).values,
                cur.abs().mean(dim=1),
                (cur > 0).float().mean(dim=1),
            )
            for stat in stats:
                feats.extend(self._agg_channel_stats(stat))

        last = x[:, -1, :]
        delta1 = x[:, -1, :] - x[:, -2, :] if x.size(1) >= 2 else torch.zeros_like(last)
        last7 = x[:, -min(7, x.size(1)):, :].mean(dim=1)
        prev7 = x[:, -14:-7, :].mean(dim=1) if x.size(1) >= 14 else last7
        last30 = x[:, -min(30, x.size(1)):, :].mean(dim=1)
        prev30 = x[:, -60:-30, :].mean(dim=1) if x.size(1) >= 60 else last30
        extras = (
            last,
            delta1,
            last7 - prev7,
            last30 - prev30,
            last.abs(),
            (last > 0).float(),
        )
        for stat in extras:
            feats.extend(self._agg_channel_stats(stat))
        return torch.stack(feats, dim=-1)

    def _channel_zero_thresholds(self, x, feat_ids=None):
        B, _, N = x.shape
        z = self.zero_norm_values.to(device=x.device, dtype=x.dtype)
        if z.numel() == 0:
            z = torch.zeros(N, device=x.device, dtype=x.dtype)
        if N == 1 and feat_ids is not None:
            ids = feat_ids.to(device=x.device).long().view(-1).clamp(0, z.numel() - 1)
            return z[ids].view(B, 1, 1)
        if z.numel() >= N:
            return z[:N].view(1, 1, N).expand(B, 1, N)
        return torch.zeros(B, 1, N, device=x.device, dtype=x.dtype)

    def _channel_id_feature(self, x, feat_ids=None):
        B, _, N = x.shape
        denom = max(int(getattr(self, 'enc_in', N)) - 1, 1)
        if N == 1 and feat_ids is not None:
            ids = feat_ids.to(device=x.device).float().view(B, 1) / float(denom)
            return ids
        ids = torch.arange(N, device=x.device, dtype=x.dtype).view(1, N).expand(B, N)
        return ids / float(max(N - 1, 1))

    def _state_event_features(self, x_hist, feat_ids=None):
        feature_set = getattr(self, 'state_event_feature_set', 'v1')
        if feature_set == 'v2':
            return self._state_event_features_v2(x_hist, feat_ids=feat_ids)
        return self._state_event_features_v1(x_hist, feat_ids=feat_ids)

    def _state_event_features_v1(self, x_hist, feat_ids=None):
        """Per-channel event-state features for sparse shipment occurrence.

        x_hist is StandardScaler-space history before RevIN. Raw zero is not 0
        after scaling, so active history is computed by comparing with the
        per-channel standardized raw-zero threshold.
        """
        x = x_hist.float()
        B, T, N = x.shape
        zero_thr = self._channel_zero_thresholds(x, feat_ids=feat_ids)
        excess = x - zero_thr
        active = excess > 1e-6
        pos_excess = torch.relu(excess)

        feats = []
        rate_by_win = {}
        mean_by_win = {}
        windows = (7, 14, 30, 60, min(self.seq_len, T))
        for win in windows:
            w = min(int(win), T)
            act_w = active[:, -w:, :].float()
            pos_w = pos_excess[:, -w:, :]
            rate = act_w.mean(dim=1)
            mean_pos = pos_w.mean(dim=1)
            max_pos = pos_w.max(dim=1).values
            std_pos = pos_w.std(dim=1, unbiased=False)
            rate_by_win[int(win)] = rate
            mean_by_win[int(win)] = mean_pos
            feats.extend([rate, mean_pos, max_pos, std_pos])

        last_active = active[:, -1, :].float()
        last_excess = excess[:, -1, :]
        delta1 = excess[:, -1, :] - excess[:, -2, :] if T >= 2 else torch.zeros_like(last_excess)

        rev_active = torch.flip(active, dims=[1]).float()
        any_active = active.any(dim=1)
        last_active_offset = rev_active.argmax(dim=1).float()
        zero_run = torch.where(any_active, last_active_offset, torch.full_like(last_active_offset, float(T)))
        zero_run_norm = zero_run / float(max(T, 1))

        r7 = rate_by_win.get(7, list(rate_by_win.values())[-1])
        r14 = rate_by_win.get(14, r7)
        r30 = rate_by_win.get(30, r14)
        r60 = rate_by_win.get(60, r30)
        m7 = mean_by_win.get(7, list(mean_by_win.values())[-1])
        m14 = mean_by_win.get(14, m7)
        m30 = mean_by_win.get(30, m14)
        m60 = mean_by_win.get(60, m30)

        zero_thr_ch = zero_thr.expand(B, 1, N).squeeze(1)
        channel_id = self._channel_id_feature(x, feat_ids=feat_ids).to(dtype=x.dtype)
        extras = [
            last_active,
            zero_run_norm,
            last_excess,
            delta1,
            r7 - r30,
            r14 - r60,
            m7 - m30,
            m14 - m60,
            zero_thr_ch,
            channel_id,
            active.float().mean(dim=1),
            pos_excess.mean(dim=1),
        ]
        feats.extend(extras)
        out = torch.stack(feats, dim=-1).reshape(B * N, -1)
        if out.size(-1) != 32:
            raise RuntimeError('state-event feature dimension expected 32, got {}'.format(out.size(-1)))
        return out

    def _state_event_features_v2(self, x_hist, feat_ids=None):
        """Enhanced per-channel event-state features for gate ranking.

        v2 keeps the compact v1 features and appends recent-event, interval,
        streak, and short/long contrast features. The branch remains a residual
        gate correction on top of the loaded LLM gate.
        """
        x = x_hist.float()
        B, T, N = x.shape
        base = self._state_event_features_v1(x_hist, feat_ids=feat_ids).view(B, N, -1)
        zero_thr = self._channel_zero_thresholds(x, feat_ids=feat_ids)
        excess = x - zero_thr
        active = excess > 1e-6
        active_f = active.float()
        pos_excess = torch.relu(excess)

        def window_stats(win):
            w = min(int(win), T)
            act = active_f[:, -w:, :]
            pos = pos_excess[:, -w:, :]
            return (
                act.mean(dim=1),
                (act.sum(dim=1) > 0).float(),
                pos.mean(dim=1),
                pos.max(dim=1).values,
            )

        r3, any3, m3, max3 = window_stats(3)
        r5, any5, m5, max5 = window_stats(5)
        r7, _, m7, _ = window_stats(7)
        r10, any10, m10, max10 = window_stats(10)
        r14, _, m14, _ = window_stats(14)
        r21, any21, m21, max21 = window_stats(21)
        r30, _, m30, _ = window_stats(30)
        r60, _, m60, _ = window_stats(60)

        rev_active = torch.flip(active_f, dims=[1])
        pos_idx = torch.arange(T, device=x.device, dtype=x.dtype).view(1, T, 1).expand(B, T, N)
        large = torch.full_like(pos_idx, float(T))
        first_offset = torch.where(rev_active > 0.5, pos_idx, large).min(dim=1).values
        first_offset = torch.clamp(first_offset, max=float(T))

        rank_from_recent = torch.cumsum(rev_active, dim=1)
        second_mask = (rev_active > 0.5) & (rank_from_recent >= 2.0)
        second_offset = torch.where(second_mask, pos_idx, large).min(dim=1).values
        second_offset = torch.clamp(second_offset, max=float(T))
        interval_last_two = torch.clamp(second_offset - first_offset, min=0.0, max=float(T))

        active_streak = torch.cumprod(rev_active, dim=1).sum(dim=1)
        prev1 = active_f[:, -2, :] if T >= 2 else torch.zeros_like(active_f[:, -1, :])
        prev2 = active_f[:, -3, :] if T >= 3 else torch.zeros_like(active_f[:, -1, :])
        last_pos = pos_excess[:, -1, :]

        extras = [
            r3, r5, r10, r21,
            m3, m5, m10, m21,
            any3, any5, any10, any21,
            max3, max5, max10, max21,
            r3 - r7, r5 - r14, r10 - r30, r21 - r60,
            m3 - m7, m5 - m14, m10 - m30, m21 - m60,
            first_offset / float(max(T, 1)),
            active_streak / float(max(T, 1)),
            second_offset / float(max(T, 1)),
            interval_last_two / float(max(T, 1)),
            active_f[:, -1, :],
            prev1,
            prev2,
            last_pos,
        ]
        extra = torch.stack(extras, dim=-1)
        out = torch.cat([base, extra], dim=-1).reshape(B * N, -1)
        if out.size(-1) != 64:
            raise RuntimeError('state-event v2 feature dimension expected 64, got {}'.format(out.size(-1)))
        return out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding, return_attention=False):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out, A = self.reprogramming(target_embedding, source_embedding, value_embedding, return_attention=return_attention)

        out = out.reshape(B, L, -1)
        out = self.out_projection(out)
        if return_attention:
            return out, A
        return out

    def reprogramming(self, target_embedding, source_embedding, value_embedding, return_attention=False):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)
        return reprogramming_embedding, (A if return_attention else None)
