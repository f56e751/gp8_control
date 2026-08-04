"""Training loop, port of ThrowBot/.../agent/trainer.py.

Loss (paper S5.2): MSE on the actuator velocities + BCE on the gripper
action.  The official code weights the BCE term by 0.5
(trainer.py:39: 0.5*loss_gripper + loss_motors); gripper labels are
converted from {-1, 1} to {0, 1} for the BCE.
Gradient-norm clipping at 0.25 as in trainer.py:73.
"""

import torch
from torch.nn import BCELoss, MSELoss


class Trainer:
    def __init__(self, model, optimizer, batch_size, get_batch, device,
                 scheduler, bce_weight=0.5, grad_clip=0.25):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.device = device
        self.scheduler = scheduler
        self.bce_weight = bce_weight
        self.grad_clip = grad_clip
        self.step_number = 0

    def loss_fn(self, a_pred, a_real):
        loss_motors = MSELoss()(a_pred[:, :-1], a_real[:, :-1])

        # gripper labels {-1, 1} -> {0, 1}
        z = torch.zeros(a_real[:, -1].shape, device=self.device)
        a_real_act = torch.where(a_real[:, -1] == -1., z, a_real[:, -1])
        loss_gripper = BCELoss()(a_pred[:, -1], a_real_act)

        return self.bce_weight * loss_gripper + loss_motors

    def train_iteration(self, num_steps, verbose=True, bc=False):
        train_loss = None
        for i in range(num_steps):
            train_loss = self.train_step_bc() if bc else self.train_step()
            self.scheduler.step()
            if verbose:
                print(f"\r \rTraining... step {i + 1}/{num_steps}", end='')
        if verbose:
            print(" Done.")
        return train_loss

    def train_step(self):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = \
            self.get_batch(batch_size=self.batch_size)
        action_target = torch.clone(actions)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg, timesteps,
            attention_mask=attention_mask)

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[
            attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(action_preds, action_target)

        self.model.train()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.step_number += 1
        return loss.detach().cpu().item()

    def train_step_bc(self):
        """BC baseline step [repo agent/trainer.py:80-101]: predict the last
        action of the trajectory from the (padded) state history."""
        states, actions, rewards, dones, rtg, _, attention_mask = \
            self.get_batch(batch_size=self.batch_size)
        action_target = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states, actions, rewards, attention_mask=attention_mask,
            target_return=rtg[:, 0])

        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)
        action_target = action_target[:, -1].reshape(-1, act_dim)

        loss = self.loss_fn(action_preds, action_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_number += 1
        return loss.detach().cpu().item()
