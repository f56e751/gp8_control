"""Decision Transformer of the paper (S4.3, S5.2).

Faithful port of the official model
(ThrowBot/src/scripts/decision_transformer/agent/model_dt/model_dt.py):
GPT backbone over interleaved (R_1, s_1, a_1, R_2, s_2, a_2, ...) tokens,
one linear embedding per modality, learned time embeddings added like
positional encodings, and an action head with Tanh on the motor velocities
and Sigmoid on the gripper command.

With the published configuration (embed 128, 1 layer, 1 head, state 5,
action 4, max_ep_len 64) the model has exactly 210,058 trainable
parameters, as reported in S5.2.
"""

import os

import torch
import torch.nn as nn

from .trajectory_gpt2 import GPT2Config, GPT2Model   # vendored


class DecisionTransformer(nn.Module):
    """Uses GPT to model (Return_1, state_1, action_1, Return_2, ...)."""

    def __init__(self, state_dim, act_dim, hidden_size, max_length=None,
                 max_ep_len=4096, **kwargs):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        config = GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)
        self.transformer = GPT2Model(config)

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = nn.Linear(1, hidden_size)
        self.embed_state = nn.Linear(state_dim, hidden_size)
        self.embed_action = nn.Linear(act_dim, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)

        # State/return heads are unused for the throwing task but are part of
        # the official model (and of its parameter count).
        self.predict_state = nn.Linear(hidden_size, state_dim)
        self.predict_action = nn.Linear(hidden_size, act_dim)
        self.predict_action_activation_m = nn.Tanh()
        self.predict_action_activation_g = nn.Sigmoid()
        self.predict_return = nn.Linear(hidden_size, 1)

    def forward(self, states, actions, rewards, returns_to_go, timesteps,
                attention_mask=None):
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length),
                                        dtype=torch.long,
                                        device=states.device)

        state_embeddings = self.embed_state(states)
        action_embeddings = self.embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_length,
                                      self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3 * seq_length)

        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # x[:, 0/1/2, t] = token for R_t / s_t / a_t
        x = x.reshape(batch_size, seq_length, 3,
                      self.hidden_size).permute(0, 2, 1, 3)

        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])

        raw = self.predict_action(x[:, 1])   # next action given state
        action_preds = torch.cat(
            [self.predict_action_activation_m(raw[:, :, :-1]),
             self.predict_action_activation_g(raw[:, :, -1:])], dim=-1)

        return state_preds, action_preds, return_preds

    def get_action(self, states, actions, rewards, returns_to_go, timesteps,
                   **kwargs):
        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            pad = self.max_length - states.shape[1]
            attention_mask = torch.cat(
                [torch.zeros(pad), torch.ones(states.shape[1])]).to(
                dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((1, pad, self.state_dim), device=states.device),
                 states], dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((1, pad, self.act_dim), device=actions.device),
                 actions], dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((1, pad, 1), device=returns_to_go.device),
                 returns_to_go], dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((1, pad), device=timesteps.device), timesteps],
                dim=1).to(dtype=torch.long)
        else:
            attention_mask = None

        _, action_preds, _ = self.forward(
            states, actions, None, returns_to_go, timesteps,
            attention_mask=attention_mask, **kwargs)

        return action_preds[0, -1]

    # -- checkpoint helpers (same format as the official code) -------------
    def save(self, file_name='dt.pth', folder='weights', extra=None):
        os.makedirs(folder, exist_ok=True)
        payload = {'state_dict': self.state_dict()}
        if extra:
            payload.update(extra)
        torch.save(payload, os.path.join(folder, file_name))

    def load(self, path='./weights/dt.pth'):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint


def build_model(model_cfg, dropout=None):
    """Instantiate the DT exactly as published (S5.2 / repo agent_dt.py)."""
    d = model_cfg.dropout if dropout is None else dropout
    return DecisionTransformer(
        state_dim=model_cfg.state_dim,
        act_dim=model_cfg.act_dim,
        max_length=model_cfg.K,
        max_ep_len=model_cfg.max_ep_len,
        hidden_size=model_cfg.embed_dim,
        n_layer=model_cfg.n_layer,
        n_head=model_cfg.n_head,
        n_inner=4 * model_cfg.embed_dim,
        activation_function=model_cfg.activation,
        n_positions=model_cfg.n_positions,
        resid_pdrop=d,
        attn_pdrop=d,
        embd_pdrop=d,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
