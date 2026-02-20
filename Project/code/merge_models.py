#!/usr/bin/env python3
"""
merge_models.py  —  Merge pre-trained Atari DQN models using HTCL consolidation.

Uses CRL-Atari's Taylor-series consolidation to produce a single multi-game agent
from individually trained best_model.pth checkpoints.

Usage (from code/ or project root):
    conda activate rl_dqn

    # Merge Breakout + Pong (ddqn_duel variant)
    python merge_models.py --games breakout pong --variant ddqn_duel

    # Merge all three games using the ddqn_duel variant
    python merge_models.py --games breakout pong freeway --variant ddqn_duel

    # Use ddqn_duel_per for Breakout — falls back to ddqn_duel for Pong/Freeway automatically
    python merge_models.py --games breakout pong freeway --variant ddqn_duel_per

    # Tune HTCL hyperparameters
    python merge_models.py --games breakout pong freeway --variant ddqn_duel \\
        --collect-steps 8000 --catchup-iters 10 --eta 0.85

After merging, evaluate the result with evaluate_models.py:
    python evaluate_models.py --games breakout pong freeway \\
        --merged ../output/merged/breakout_pong_freeway_ddqn_duel.pth \\
        --episodes 100
"""

import os
import sys
import copy
import json
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    pass

from gymnasium.wrappers import AtariPreprocessing
try:
    from gymnasium.wrappers import FrameStack
except ImportError:
    from gymnasium.wrappers import FrameStackObservation as FrameStack

_HERE = os.path.dirname(os.path.abspath(__file__))
# (No CRL-Atari import needed — we use action-mask-aware versions below)


# ── Paths ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.normpath(os.path.join(_HERE, '..', 'output'))
MERGED_DIR = os.path.join(OUTPUT_DIR, 'merged')


# ── Game registry ─────────────────────────────────────────────────────────────

GAME_INFO = {
    'breakout':      dict(env_id='BreakoutNoFrameskip-v4',     valid_actions=[0, 1, 3, 4]),
    'pong':          dict(env_id='PongNoFrameskip-v4',          valid_actions=[0, 1, 3, 4, 11, 12]),
    'freeway':       dict(env_id='FreewayNoFrameskip-v4',       valid_actions=[0, 2, 5]),
    'spaceinvaders': dict(env_id='SpaceInvadersNoFrameskip-v4', valid_actions=[0, 1, 3, 4, 11, 12]),
}

ACTION_DIM  = 18
INPUT_SHAPE = (4, 84, 84)


# ── Network definitions (identical to code/dqn.py) ───────────────────────────

class CNNDQN(nn.Module):
    """Standard CNN DQN — no dueling streams."""

    def __init__(self, input_shape=INPUT_SHAPE, output_dim=ACTION_DIM):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c,  32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def _out(s, k, st): return (s - (k - 1) - 1) // st + 1
        flat = (_out(_out(_out(w, 8, 4), 4, 2), 3, 1) *
                _out(_out(_out(h, 8, 4), 4, 2), 3, 1) * 64)

        self.fc1 = nn.Linear(flat, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


class DuelingCNNDQN(nn.Module):
    """Dueling CNN DQN — separate value and advantage streams."""

    def __init__(self, input_shape=INPUT_SHAPE, output_dim=ACTION_DIM):
        super().__init__()
        c, h, w = input_shape
        self.conv1 = nn.Conv2d(c,  32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        def _out(s, k, st): return (s - (k - 1) - 1) // st + 1
        flat = (_out(_out(_out(w, 8, 4), 4, 2), 3, 1) *
                _out(_out(_out(h, 8, 4), 4, 2), 3, 1) * 64)

        self.fc_value  = nn.Linear(flat, 512)
        self.value     = nn.Linear(512, 1)
        self.fc_adv    = nn.Linear(flat, 512)
        self.advantage = nn.Linear(512, output_dim)

    def forward(self, x):
        x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        v = self.value(F.relu(self.fc_value(x)))
        a = self.advantage(F.relu(self.fc_adv(x)))
        return v + a - a.mean(dim=1, keepdim=True)


# ── Model helpers ─────────────────────────────────────────────────────────────

def _build_model(dueling: bool, device: torch.device) -> nn.Module:
    cls = DuelingCNNDQN if dueling else CNNDQN
    return cls(INPUT_SHAPE, ACTION_DIM).to(device)


def _resolve_variant(game: str, requested: str) -> tuple:
    """
    Return (model_dir, actual_variant) for the best available match.
    Falls back: ddqn_duel_per → ddqn_duel → ddqn.
    """
    fallbacks = [requested]
    if '_per' in requested:
        fallbacks.append(requested.replace('_per', ''))
    if 'duel' in requested:
        fallbacks.append('ddqn')

    for v in fallbacks:
        path = os.path.join(OUTPUT_DIR, game, v, 'best_model.pth')
        if os.path.exists(path):
            if v != requested:
                print(f'  [WARN] {game}/{requested} not found, using {game}/{v}')
            return os.path.join(OUTPUT_DIR, game, v), v

    available = [
        d for d in os.listdir(os.path.join(OUTPUT_DIR, game))
        if os.path.isdir(os.path.join(OUTPUT_DIR, game, d))
        and os.path.exists(os.path.join(OUTPUT_DIR, game, d, 'best_model.pth'))
    ] if os.path.isdir(os.path.join(OUTPUT_DIR, game)) else []

    raise FileNotFoundError(
        f"No model found for {game}/{requested} (tried: {fallbacks}).\n"
        f"  Available variants in output/{game}/: {available}"
    )


def load_model(game: str, variant: str, dueling: bool, device: torch.device):
    """
    Load best_model.pth for a game/variant.
    Returns (model, actual_variant_used).
    """
    model_dir, actual = _resolve_variant(game, variant)
    model_path = os.path.join(model_dir, 'best_model.pth')
    ckpt   = torch.load(model_path, map_location=device, weights_only=False)
    model  = _build_model(dueling, device)
    model.load_state_dict(ckpt['policy_net_state_dict'])
    model.eval()
    return model, actual


# ── Environment ───────────────────────────────────────────────────────────────

def make_env(game: str, seed: int = 0):
    info = GAME_INFO[game]
    env  = gym.make(info['env_id'], full_action_space=True, frameskip=1)
    env  = AtariPreprocessing(env, noop_max=30, frame_skip=4,
                              screen_size=84, grayscale_obs=True,
                              terminal_on_life_loss=True, scale_obs=False)
    env  = FrameStack(env, 4)
    env.reset(seed=seed)
    return env


# ── Transition collection ─────────────────────────────────────────────────────

def collect_transitions(model, game: str, n_steps: int, device) -> dict:
    """
    Roll out model for n_steps environment steps.
    Returns a dict of numpy arrays including 'action_mask' — a per-transition
    float32 array of shape [N, ACTION_DIM] with 0 for valid actions and -1e9 for
    invalid ones. Used by the masked HTCL functions to correctly compute TD targets.
    """
    info  = GAME_INFO[game]
    va    = info['valid_actions']
    env   = make_env(game, seed=0)
    S, A, R, S2, D = [], [], [], [], []
    state, _ = env.reset()

    for _ in range(n_steps):
        # Epsilon-greedy with small epsilon for diversity
        if np.random.random() < 0.05:
            action = int(np.random.choice(va))
        else:
            with torch.no_grad():
                s = torch.from_numpy(
                    np.array(state, dtype=np.uint8)).unsqueeze(0).to(device)
                q = model(s).cpu().numpy()[0]
            mask = np.full(ACTION_DIM, -np.inf)
            mask[va] = q[va]
            action = int(np.argmax(mask))

        nxt, reward, term, trunc, _ = env.step(action)
        done = term or trunc

        S.append(np.array(state, dtype=np.uint8))
        A.append(action)
        R.append(float(np.clip(reward, -1.0, 1.0)))
        S2.append(np.array(nxt,   dtype=np.uint8))
        D.append(float(done))

        state = nxt if not done else env.reset()[0]

    env.close()

    # Build per-transition action mask: 0 for valid actions, -1e9 for invalid
    action_mask = np.full((n_steps, ACTION_DIM), -1e9, dtype=np.float32)
    action_mask[:, va] = 0.0

    return dict(
        states      = np.array(S,  dtype=np.uint8),
        actions     = np.array(A,  dtype=np.int64),
        rewards     = np.array(R,  dtype=np.float32),
        next_states = np.array(S2, dtype=np.uint8),
        dones       = np.array(D,  dtype=np.float32),
        action_mask = action_mask,
    )


class _TransitionBuffer:
    """Thin wrapper so ConsolidationBuffer.add_game() can call sample_all()."""

    def __init__(self, t: dict):
        self._t = t

    def sample_all(self, n: int) -> dict:
        idx = np.random.permutation(len(self._t['states']))[:n]
        return {k: v[idx] for k, v in self._t.items()}


# ── Masked HTCL consolidation ─────────────────────────────────────────────────
#
# CRL-Atari's taylor.py computes:
#   next_q = model(next_states).max(1)[0]          ← no action masking
#
# This can select invalid actions whose Q-values were never trained, producing
# corrupted TD targets, gradients, and Hessian estimates.
#
# These masked versions add per-transition action_mask (0 for valid, -1e9 for
# invalid) so the max always picks from valid actions only.

def _combine_transitions(game_buffers: dict) -> dict:
    """
    Concatenate and shuffle transitions from all games (preserving action_mask
    alignment, which ConsolidationBuffer.get_combined() cannot do because it
    only handles a fixed set of 5 keys).
    """
    keys = ['states', 'actions', 'rewards', 'next_states', 'dones', 'action_mask']
    combined = {k: np.concatenate([buf[k] for buf in game_buffers.values()], axis=0)
                for k in keys}
    idx = np.random.permutation(len(combined['states']))
    return {k: v[idx] for k, v in combined.items()}


def _dqn_loss_masked(model, states, actions, rewards, next_states, dones,
                     action_masks, gamma: float) -> torch.Tensor:
    """Huber TD loss with valid-action masking for the next-state max Q-value."""
    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_all = model(next_states) + action_masks   # -1e9 makes invalid → -inf
        next_q     = next_q_all.max(1)[0]
        targets    = rewards + gamma * (1.0 - dones) * next_q
    return F.smooth_l1_loss(q_values, targets)


def _estimate_hessian_masked(model, transitions: dict, device,
                              gamma=0.99, batch_size=256) -> dict:
    """Diagonal Fisher approximation (squared gradients) using masked TD loss."""
    model.eval()
    hessian = {n: torch.zeros_like(p, device=device)
               for n, p in model.named_parameters() if p.requires_grad}

    states       = torch.from_numpy(transitions['states']).to(device)
    actions      = torch.from_numpy(transitions['actions']).long().to(device)
    rewards      = torch.from_numpy(transitions['rewards']).to(device)
    next_states  = torch.from_numpy(transitions['next_states']).to(device)
    dones        = torch.from_numpy(transitions['dones']).to(device)
    action_masks = torch.from_numpy(transitions['action_mask']).to(device)

    n_batches = 0
    for i in range(0, len(states), batch_size):
        j = min(i + batch_size, len(states))
        model.zero_grad()
        loss = _dqn_loss_masked(model, states[i:j], actions[i:j], rewards[i:j],
                                next_states[i:j], dones[i:j], action_masks[i:j], gamma)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                hessian[name] += param.grad.detach().pow(2)
        n_batches += 1

    for name in hessian:
        hessian[name] /= max(n_batches, 1)
    return hessian


def _compute_gradients_masked(model, transitions: dict, device,
                               gamma=0.99, batch_size=256) -> dict:
    """Mean gradient of masked DQN loss over transition buffer."""
    model.train()
    grads = {n: torch.zeros_like(p, device=device)
             for n, p in model.named_parameters() if p.requires_grad}

    states       = torch.from_numpy(transitions['states']).to(device)
    actions      = torch.from_numpy(transitions['actions']).long().to(device)
    rewards      = torch.from_numpy(transitions['rewards']).to(device)
    next_states  = torch.from_numpy(transitions['next_states']).to(device)
    dones        = torch.from_numpy(transitions['dones']).to(device)
    action_masks = torch.from_numpy(transitions['action_mask']).to(device)

    n_batches = 0
    for i in range(0, len(states), batch_size):
        j = min(i + batch_size, len(states))
        model.zero_grad()
        loss = _dqn_loss_masked(model, states[i:j], actions[i:j], rewards[i:j],
                                next_states[i:j], dones[i:j], action_masks[i:j], gamma)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grads[name] += param.grad.detach()
        n_batches += 1

    for name in grads:
        grads[name] /= max(n_batches, 1)
    return grads


def _taylor_update_masked(global_model, local_model, transitions: dict, device,
                           gamma=0.99, eta=0.9, max_norm=1.0,
                           lambda_reg=None, verbose=False):
    """
    HTCL Taylor update (Eq. 6) using masked TD loss:
        Δw* = (H + λI)^{-1} [λ(w_local - w_global) - g]
        w_global ← w_global + η · Δw*
    """
    grads    = _compute_gradients_masked(global_model, transitions, device, gamma)
    hessians = _estimate_hessian_masked(global_model, transitions, device, gamma)

    if lambda_reg is None:
        diag    = torch.cat([v.flatten() for v in hessians.values()])
        max_eig = float(diag.max().item())
        min_eig = float(diag.min().item())
        if not np.isfinite(max_eig) or max_eig <= 0:
            lambda_reg = max(1e-6, 10.0 * (abs(min_eig) + 1e-6))
        else:
            lambda_reg = 1000.0 * max_eig

    if verbose:
        print(f'    Taylor update: lambda_reg={lambda_reg:.4f}, eta={eta}')

    local_state = local_model.state_dict()
    with torch.no_grad():
        for name, param in global_model.named_parameters():
            if not param.requires_grad or name not in grads:
                continue
            h     = hessians[name].to(device)
            w_loc = local_state[name].to(device)
            denom = h + lambda_reg + 1e-8
            delta_d   = w_loc - param
            raw_delta = (1.0 / denom) * (lambda_reg * delta_d - grads[name])
            delta     = eta * raw_delta
            dnorm     = delta.norm().item()
            if dnorm > max_norm:
                delta = delta * (max_norm / (dnorm + 1e-12))
            if not (torch.isnan(delta).any() or torch.isinf(delta).any()):
                param.add_(delta)
            elif verbose:
                print(f'    Warning: NaN/Inf in delta for {name}, skipping')
    return global_model


def _global_catchup_masked(global_model, local_model, transitions: dict, device,
                            num_iterations=5, gamma=0.99, eta=0.9, max_norm=1.0,
                            lambda_reg=10_000.0, catchup_lr=0.001,
                            patience=2, verbose=False):
    """Catch-up refinement phase using masked TD loss."""
    if num_iterations <= 0:
        return global_model

    states       = torch.from_numpy(transitions['states']).to(device)
    actions      = torch.from_numpy(transitions['actions']).long().to(device)
    rewards      = torch.from_numpy(transitions['rewards']).to(device)
    next_states  = torch.from_numpy(transitions['next_states']).to(device)
    dones        = torch.from_numpy(transitions['dones']).to(device)
    action_masks = torch.from_numpy(transitions['action_mask']).to(device)

    global_model.eval()
    eval_n = min(512, len(states))
    with torch.no_grad():
        initial_loss = _dqn_loss_masked(
            global_model, states[:eval_n], actions[:eval_n], rewards[:eval_n],
            next_states[:eval_n], dones[:eval_n], action_masks[:eval_n], gamma,
        ).item()

    best_loss  = initial_loss
    best_state = copy.deepcopy(global_model.state_dict())
    no_improve = 0

    if verbose:
        print(f'    Catchup initial loss: {initial_loss:.4f}')

    for it in range(num_iterations):
        pre_state = copy.deepcopy(global_model.state_dict())

        # SGD fine-tuning on a temporary copy
        tmp     = copy.deepcopy(global_model).to(device)
        tmp_opt = torch.optim.Adam(tmp.parameters(), lr=catchup_lr)
        tmp.train()
        for i in range(0, len(states), 256):
            j = min(i + 256, len(states))
            tmp_opt.zero_grad()
            loss = _dqn_loss_masked(tmp, states[i:j], actions[i:j], rewards[i:j],
                                    next_states[i:j], dones[i:j], action_masks[i:j], gamma)
            loss.backward()
            nn.utils.clip_grad_norm_(tmp.parameters(), 10.0)
            tmp_opt.step()

        # Conservative Taylor pull toward fine-tuned model
        _taylor_update_masked(global_model, tmp, transitions, device,
                              gamma=gamma, eta=eta, max_norm=max_norm,
                              lambda_reg=lambda_reg, verbose=False)
        del tmp

        global_model.eval()
        with torch.no_grad():
            cur_loss = _dqn_loss_masked(
                global_model, states[:eval_n], actions[:eval_n], rewards[:eval_n],
                next_states[:eval_n], dones[:eval_n], action_masks[:eval_n], gamma,
            ).item()

        if verbose:
            status = 'improved' if cur_loss < best_loss else 'worse'
            print(f'    Catchup {it+1}/{num_iterations}: loss={cur_loss:.4f} ({status})')

        if cur_loss < best_loss:
            best_loss  = cur_loss
            best_state = copy.deepcopy(global_model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            global_model.load_state_dict(pre_state)
            if no_improve >= patience:
                if verbose:
                    print(f'    Early stopping at iteration {it+1}')
                break

    global_model.load_state_dict(best_state)
    return global_model


# ── Policy distillation merge ─────────────────────────────────────────────────

def _distillation_merge(student_model, teacher_models: dict, game_transitions: dict,
                        games: list, device,
                        n_epochs: int = 30, lr: float = 3e-4, batch_size: int = 256,
                        weight_decay: float = 0.0,
                        eval_fn=None, eval_every: int = 10,
                        verbose: bool = True) -> nn.Module:
    """
    Multi-task policy distillation: train student_model to imitate each teacher.

    For every game we minimise MSE between student Q-values and teacher Q-values
    only on valid actions.  Losses are averaged PER GAME (not per sample) so that
    all games contribute equal gradient weight regardless of Q-value scale.

    Critical: Breakout Q-values span ~19 units while Pong spans ~3 units.
    Without per-game averaging the Breakout loss ~34x dominates, preventing
    adequate Pong learning despite low overall MSE.

    eval_fn: optional callable(model) -> dict{game: score}.  When provided,
             best model is selected by total game score (not training loss).
             This prevents overfitting: training loss keeps decreasing but
             actual gameplay performance peaks and then degrades.
    """
    # ── Pre-compute teacher Q-values (cached — no gradient needed) ─────────────
    if verbose:
        print('  Pre-computing teacher Q-values ...')
    teacher_q_cache = {}
    for game in games:
        teacher = teacher_models[game]
        states  = torch.from_numpy(game_transitions[game]['states']).to(device)
        teacher.eval()
        with torch.no_grad():
            chunks = []
            for i in range(0, len(states), batch_size):
                chunks.append(teacher(states[i:min(i + batch_size, len(states))]).cpu())
            teacher_q_cache[game] = torch.cat(chunks)   # [N, ACTION_DIM]
        if verbose:
            tq = teacher_q_cache[game]
            print(f'    {game}: {len(tq)} states, '
                  f'Q range [{tq.min():.2f}, {tq.max():.2f}]')

    # ── Build combined dataset with per-game IDs ───────────────────────────────
    all_states    = torch.cat([torch.from_numpy(game_transitions[g]['states'])  for g in games])
    all_teacher_q = torch.cat([teacher_q_cache[g]                               for g in games])
    all_masks     = torch.cat([
        (torch.from_numpy(game_transitions[g]['action_mask']) > -1e8).float()   # [N, 18] bool→float
        for g in games
    ])
    # Track which game each sample belongs to (for per-game loss balancing)
    all_game_ids  = torch.cat([
        torch.full((len(game_transitions[g]['states']),), gid, dtype=torch.long)
        for gid, g in enumerate(games)
    ])
    n_total = len(all_states)

    if verbose:
        print(f'  Total training states: {n_total}')
        print(f'  Per-game loss balancing: each game weighted equally')

    # ── Training loop ─────────────────────────────────────────────────────────
    opt       = torch.optim.Adam(student_model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(n_epochs, 1), eta_min=lr * 0.1)

    # Best-model selection: by game score if eval_fn provided, else by training loss.
    # Game-score selection prevents overfitting (training loss keeps falling while
    # gameplay degrades — we want the peak-performance checkpoint, not lowest loss).
    use_score_select = eval_fn is not None
    best_loss        = float('inf')
    best_score       = -float('inf')
    best_state       = copy.deepcopy(student_model.state_dict())

    student_model.train()
    for epoch in range(n_epochs):
        perm       = torch.randperm(n_total)
        epoch_loss = 0.0
        n_batches  = 0

        for i in range(0, n_total, batch_size):
            j      = min(i + batch_size, n_total)
            b      = perm[i:j]
            s_b    = all_states[b].to(device)
            tq_b   = all_teacher_q[b].to(device)
            m_b    = all_masks[b].to(device)
            gid_b  = all_game_ids[b]               # CPU is fine (indexing only)

            opt.zero_grad()
            sq_b = student_model(s_b)              # [B, ACTION_DIM]

            # Per-game MSE averaged equally — prevents Q-scale imbalance.
            # Each game's loss = MSE on its own valid actions only.
            game_losses = []
            for gid in range(len(games)):
                gmask_b = (gid_b == gid)           # which batch rows belong to this game
                if gmask_b.sum() == 0:
                    continue
                diff_g = (sq_b[gmask_b] - tq_b[gmask_b]) * m_b[gmask_b]
                loss_g = diff_g.pow(2).sum() / (m_b[gmask_b].sum() + 1e-8)
                game_losses.append(loss_g)

            loss = sum(game_losses) / len(game_losses)
            loss.backward()
            nn.utils.clip_grad_norm_(student_model.parameters(), 10.0)
            opt.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss   = epoch_loss / max(n_batches, 1)
        loss_marker = ' *' if avg_loss < best_loss else ''
        if avg_loss < best_loss:
            best_loss = avg_loss
            if not use_score_select:
                best_state = copy.deepcopy(student_model.state_dict())

        # ── Periodic game-performance evaluation ──────────────────────────────
        score_line = ''
        if use_score_select and ((epoch + 1) % eval_every == 0 or epoch == n_epochs - 1):
            student_model.eval()
            scores    = eval_fn(student_model)
            student_model.train()
            total     = sum(scores.values())
            score_str = '  '.join(f'{g}={s:+.1f}' for g, s in scores.items())
            if total > best_score:
                best_score  = total
                best_state  = copy.deepcopy(student_model.state_dict())
                score_line  = f'  [{score_str}] <-- best'
            else:
                score_line  = f'  [{score_str}]'

        if verbose:
            print(f'  Epoch {epoch+1:3d}/{n_epochs}: loss={avg_loss:.6f}{loss_marker}{score_line}')

    student_model.load_state_dict(best_state)
    student_model.eval()
    return student_model


# ── Quick evaluation ──────────────────────────────────────────────────────────

def quick_eval(model, game: str, n_episodes: int, device) -> tuple:
    """Returns (mean_reward, std_reward) over n greedy episodes."""
    va  = GAME_INFO[game]['valid_actions']
    env = make_env(game, seed=42)
    totals = []
    for _ in range(n_episodes):
        state, _ = env.reset()
        total, done = 0.0, False
        while not done:
            with torch.no_grad():
                s = torch.from_numpy(
                    np.array(state, dtype=np.uint8)).unsqueeze(0).to(device)
                q = model(s).cpu().numpy()[0]
            mask = np.full(ACTION_DIM, -np.inf)
            mask[va] = q[va]
            state, r, term, trunc, _ = env.step(int(np.argmax(mask)))
            total += r
            done   = term or trunc
        totals.append(total)
    env.close()
    return float(np.mean(totals)), float(np.std(totals))


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Merge trained Atari DQN models with HTCL consolidation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    p.add_argument('--games', nargs='+', required=True,
                   choices=list(GAME_INFO),
                   metavar='GAME',
                   help='Games to merge in order  (e.g. breakout pong freeway). '
                        'The first game initialises the global model.')
    p.add_argument('--variant', required=True,
                   metavar='VARIANT',
                   help='Model variant to load  (e.g. ddqn  ddqn_duel  ddqn_duel_per). '
                        'Falls back to the closest available variant per game.')
    p.add_argument('--method', choices=['htcl', 'distill'], default='htcl',
                   help='Merge strategy: htcl (Taylor consolidation, designed for '
                        'sequentially-trained models) or distill (policy distillation, '
                        'recommended for independently-trained specialists)  (default: htcl)')

    # HTCL hyperparameters
    htcl = p.add_argument_group('HTCL hyperparameters (--method htcl)')
    htcl.add_argument('--collect-steps', type=int, default=5000,
                      help='Transitions collected per game for Hessian/gradient  (default: 5000)')
    htcl.add_argument('--eta', type=float, default=0.9,
                      help='Taylor update step size η  (default: 0.9)')
    htcl.add_argument('--max-norm', type=float, default=1.0,
                      help='Maximum update norm for clipping  (default: 1.0)')
    htcl.add_argument('--lambda-reg', type=float, default=None,
                      help='Hessian regularisation λ  (default: auto-computed)')
    htcl.add_argument('--catchup-iters', type=int, default=5,
                      help='Catch-up refinement iterations  (default: 5)')
    htcl.add_argument('--catchup-lr', type=float, default=0.001,
                      help='Catch-up Adam learning rate  (default: 0.001)')
    htcl.add_argument('--catchup-lambda', type=float, default=10_000.0,
                      help='Catch-up regularisation strength  (default: 10000)')

    # Distillation hyperparameters
    distill = p.add_argument_group('Distillation hyperparameters (--method distill)')
    distill.add_argument('--distill-epochs', type=int, default=60,
                         help='Training epochs for distillation  (default: 60)')
    distill.add_argument('--distill-lr', type=float, default=3e-4,
                         help='Learning rate for distillation Adam  (default: 3e-4)')
    distill.add_argument('--distill-wd', type=float, default=1e-4,
                         help='Weight decay (L2 reg) for distillation — prevents overfitting  (default: 1e-4)')
    distill.add_argument('--distill-init', choices=['warmstart', 'random'], default='warmstart',
                         help='Student init: warmstart from first game, or random  (default: warmstart)')
    distill.add_argument('--eval-every', type=int, default=10,
                         help='Evaluate game performance every N epochs; best checkpoint by score  (default: 10)')

    # Output / evaluation
    io = p.add_argument_group('Output')
    io.add_argument('--eval-episodes', type=int, default=10,
                    help='Episodes to evaluate each stage snapshot  (default: 10)')
    io.add_argument('--output-dir', type=str, default=MERGED_DIR,
                    help=f'Directory to save merged model  (default: output/merged/)')
    io.add_argument('--name', type=str, default=None,
                    help='Custom filename without .pth  '
                         '(default: <games>_<variant>  e.g. breakout_pong_ddqn_duel)')
    io.add_argument('--device', type=str, default=None,
                    choices=['auto', 'cpu', 'cuda', 'mps'],
                    help='Compute device  (default: auto-detect)')
    io.add_argument('--seed', type=int, default=42,
                    help='Random seed  (default: 42)')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if args.device and args.device != 'auto':
        device = torch.device(args.device)
    else:
        device = torch.device(
            'mps'  if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available()          else
            'cpu'
        )

    # Architecture from variant name
    dueling   = 'duel' in args.variant
    arch_name = 'DuelingCNNDQN' if dueling else 'CNNDQN'

    # Output path
    os.makedirs(args.output_dir, exist_ok=True)
    out_name = args.name or ('_'.join(args.games) + '_' + args.variant)
    out_path = os.path.join(args.output_dir, out_name + '.pth')

    method_label = 'HTCL Taylor Consolidation' if args.method == 'htcl' else 'Policy Distillation'
    print('=' * 60)
    print(f'Model Merge  —  {method_label}')
    print('=' * 60)
    print(f'Device        : {device}')
    print(f'Games (order) : {" → ".join(args.games)}')
    print(f'Variant       : {args.variant}  →  architecture: {arch_name}')
    print(f'Method        : {args.method}')
    print(f'Collect steps : {args.collect_steps} per game')
    if args.method == 'htcl':
        print(f'Catchup iters : {args.catchup_iters}')
    else:
        print(f'Distill epochs: {args.distill_epochs}  lr: {args.distill_lr}  '
              f'wd: {args.distill_wd}  init: {args.distill_init}'
              f'  eval-every: {args.eval_every}')
    print(f'Output        : {out_path}')
    print('=' * 60)

    # ── 1. Load individual models ─────────────────────────────────────────────
    print('\n=== Loading individual models ===')
    local_models    = {}
    actual_variants = {}
    for game in args.games:
        model, actual = load_model(game, args.variant, dueling, device)
        local_models[game]    = model
        actual_variants[game] = actual
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  {game:12s} [{actual:22s}]  params={n_params:,}')

    # ── 2. Baseline: individual model performance before merge ────────────────
    print(f'\n=== Individual performance (before merge, {args.eval_episodes} eps) ===')
    for game in args.games:
        mean, std = quick_eval(local_models[game], game, args.eval_episodes, device)
        print(f'  {game:12s}: {mean:+.2f} ± {std:.2f}')

    # ── 3. Collect transitions for consolidation ──────────────────────────────
    print(f'\n=== Collecting {args.collect_steps} transitions per game ===')
    raw = {}
    for game in args.games:
        print(f'  {game} ...', end=' ', flush=True)
        raw[game] = collect_transitions(
            local_models[game], game, args.collect_steps, device)
        print(f'done  ({len(raw[game]["states"])} steps)')

    # ── 4. Merge ───────────────────────────────────────────────────────────────
    global_model = _build_model(dueling, device)

    if args.method == 'htcl':
        # ── 4a. HTCL consolidation (sequential, mirrors CRL-Atari run_htcl) ─
        print('\n=== HTCL Taylor Consolidation (action-mask-aware) ===')
        accumulated = {}

        for stage_idx, game in enumerate(args.games):
            local             = local_models[game]
            accumulated[game] = raw[game]

            if stage_idx == 0:
                global_model.load_state_dict(copy.deepcopy(local.state_dict()))
                print(f'\n  Stage 0 — {game}: global initialised from local model')
            else:
                combined = _combine_transitions(
                    {g: accumulated[g] for g in args.games[:stage_idx + 1]}
                )

                print(f'\n  Stage {stage_idx} — {game}: Taylor update ...')
                _taylor_update_masked(
                    global_model, local, combined, device,
                    gamma=0.99,
                    eta=args.eta,
                    max_norm=args.max_norm,
                    lambda_reg=args.lambda_reg,
                    verbose=True,
                )

                print(f'  Catch-up phase ({args.catchup_iters} iterations) ...')
                _global_catchup_masked(
                    global_model, local, combined, device,
                    num_iterations=args.catchup_iters,
                    gamma=0.99,
                    eta=args.eta,
                    max_norm=args.max_norm,
                    lambda_reg=args.catchup_lambda,
                    catchup_lr=args.catchup_lr,
                    verbose=True,
                )

            # Snapshot after this stage
            print(f'  Snapshot after merging {game}:', end='')
            for g in args.games[:stage_idx + 1]:
                m, _ = quick_eval(global_model, g, args.eval_episodes, device)
                print(f'  {g}={m:+.1f}', end='')
            print()

    else:
        # ── 4b. Policy distillation ────────────────────────────────────────────
        print('\n=== Policy Distillation Merge ===')
        if args.distill_init == 'warmstart':
            global_model.load_state_dict(
                copy.deepcopy(local_models[args.games[0]].state_dict()))
            print(f'  Student init: warm-start from {args.games[0]}')
        else:
            print(f'  Student init: random (Xavier)')

        # Evaluation function: run a few quick episodes per game.
        # Used for game-score-based best-checkpoint selection, which avoids
        # overfitting (training loss falls monotonically but game score peaks).
        fast_eps = min(5, args.eval_episodes)

        def _distill_eval(model):
            model.eval()
            return {g: quick_eval(model, g, fast_eps, device)[0]
                    for g in args.games}

        _distillation_merge(
            global_model, local_models, raw, args.games, device,
            n_epochs=args.distill_epochs,
            lr=args.distill_lr,
            weight_decay=args.distill_wd,
            eval_fn=_distill_eval,
            eval_every=args.eval_every,
            verbose=True,
        )

    # ── 5. Final evaluation ───────────────────────────────────────────────────
    print(f'\n=== Final merged model ({args.eval_episodes} eps each) ===')
    final_scores = {}
    for game in args.games:
        mean, std = quick_eval(global_model, game, args.eval_episodes, device)
        final_scores[game] = dict(mean=mean, std=std)
        print(f'  {game:12s}: {mean:+.2f} ± {std:.2f}')

    # ── 6. Save merged model ──────────────────────────────────────────────────
    merge_config = dict(
        method        = args.method,
        collect_steps = args.collect_steps,
    )
    if args.method == 'htcl':
        merge_config.update(dict(
            eta            = args.eta,
            max_norm       = args.max_norm,
            lambda_reg     = args.lambda_reg,
            catchup_iters  = args.catchup_iters,
            catchup_lr     = args.catchup_lr,
            catchup_lambda = args.catchup_lambda,
        ))
    else:
        merge_config.update(dict(
            distill_epochs = args.distill_epochs,
            distill_lr     = args.distill_lr,
            distill_wd     = args.distill_wd,
            distill_init   = args.distill_init,
            eval_every     = args.eval_every,
        ))

    checkpoint = dict(
        state_dict      = global_model.state_dict(),
        architecture    = arch_name,
        action_dim      = ACTION_DIM,
        games           = args.games,
        variant         = args.variant,
        actual_variants = actual_variants,
        merge_config    = merge_config,
        final_scores    = final_scores,
        created         = datetime.datetime.now().isoformat(),
    )
    torch.save(checkpoint, out_path)

    print(f'\nMerged model saved → {out_path}')
    print('\nEvaluate on 100 episodes with:')
    print(f'  python evaluate_models.py'
          f' --games {" ".join(args.games)}'
          f' --merged {out_path}'
          f' --episodes 100')


if __name__ == '__main__':
    main()
