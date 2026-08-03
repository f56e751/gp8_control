"""Minimal GPT-2 backbone for the Decision Transformer.

The official implementation (ThrowBot/.../model_dt/trajectory_gpt2.py) is the
HuggingFace GPT2Model with the positional embeddings removed -- the DT adds
its own learned time embeddings instead (Chen et al., 2021).  The HF file the
authors vendored depends on transformers<4.30 internals, so this is a clean,
dependency-free re-implementation of exactly the same computation graph:

    inputs_embeds -> embd dropout -> [ln_1 -> masked self-attn -> residual
    -> ln_2 -> MLP(4x, act) -> residual] x n_layer -> ln_f

Parameter shapes match GPT2Model (including the vocab-size-1 token embedding
`wte`, unused by the DT but part of the checkpointed model), so the total
trainable-parameter count reproduces the paper's reported 210,058.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ACT2FN = {
    'relu': F.relu,
    'gelu': F.gelu,
    'gelu_new': F.gelu,
    'tanh': torch.tanh,
}


class GPT2Config:
    def __init__(self, vocab_size=1, n_embd=128, n_layer=1, n_head=1,
                 n_inner=None, activation_function='relu', n_positions=1024,
                 resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
                 layer_norm_epsilon=1e-5, initializer_range=0.02, **kwargs):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.n_positions = n_positions
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(config.n_positions, config.n_positions,
                                  dtype=torch.bool)).view(
                1, 1, config.n_positions, config.n_positions),
            persistent=False)

    def forward(self, x, attention_bias=None):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
        att = att.masked_fill(~self.causal_mask[:, :, :T, :T],
                              torch.finfo(att.dtype).min)
        if attention_bias is not None:
            att = att + attention_bias
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_inner)
        self.c_proj = nn.Linear(config.n_inner, config.n_embd)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)

    def forward(self, x, attention_bias=None):
        x = x + self.attn(self.ln_1(x), attention_bias=attention_bias)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    """GPT-2 without positional embeddings (they are added by the DT)."""

    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config
        # Unused by the DT (inputs are continuous embeddings), but present in
        # the official model; kept for exact parameter-count parity (vocab=1).
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs_embeds, attention_mask=None):
        B, T, _ = inputs_embeds.shape
        if attention_mask is not None:
            # (B, T) with 1 = attend, 0 = masked  ->  additive bias
            bias = (1.0 - attention_mask[:, None, None, :].to(
                inputs_embeds.dtype)) * torch.finfo(inputs_embeds.dtype).min
        else:
            bias = None
        h = self.drop(inputs_embeds)
        for block in self.h:
            h = block(h, attention_bias=bias)
        h = self.ln_f(h)
        return {'last_hidden_state': h}
