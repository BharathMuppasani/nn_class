#!/usr/bin/env python3
"""
evaluate_models.py  —  Benchmark trained Atari DQN models across variants and games.

Runs N episodes per model and writes a formatted results report to a text file.

Usage (from code/ or project root):
    conda activate rl_dqn

    # Test all auto-discovered variants for all three games (100 episodes each)
    python evaluate_models.py --games breakout pong freeway --episodes 100

    # Test only the ddqn_duel variant for Breakout and Pong
    python evaluate_models.py --games breakout pong --variants ddqn_duel --episodes 100

    # Also evaluate a merged model produced by merge_models.py
    python evaluate_models.py --games breakout pong freeway --episodes 100 \\
        --merged ../output/merged/breakout_pong_freeway_ddqn_duel.pth

    # Multiple merged models at once
    python evaluate_models.py --games breakout pong freeway --episodes 100 \\
        --merged ../output/merged/model1.pth ../output/merged/model2.pth

    # Custom output file name
    python evaluate_models.py --games breakout pong freeway \\
        --episodes 100 --output results_comparison.txt
"""

import os
import sys
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

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ── Paths ─────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.normpath(os.path.join(_HERE, '..', 'output'))


# ── Game registry ─────────────────────────────────────────────────────────────

GAME_INFO = {
    'breakout':      dict(env_id='BreakoutNoFrameskip-v4',     valid_actions=[0, 1, 3, 4]),
    'pong':          dict(env_id='PongNoFrameskip-v4',          valid_actions=[0, 1, 3, 4, 11, 12]),
    'freeway':       dict(env_id='FreewayNoFrameskip-v4',       valid_actions=[0, 2, 5]),
    'spaceinvaders': dict(env_id='SpaceInvadersNoFrameskip-v4', valid_actions=[0, 1, 3, 4, 11, 12]),
}

ACTION_DIM = 18   # full Atari action space used during training


# ── Network definitions (identical to code/dqn.py) ───────────────────────────

class CNNDQN(nn.Module):
    """Standard CNN DQN — no dueling streams."""

    def __init__(self, input_shape=(4, 84, 84), output_dim=ACTION_DIM):
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

    def __init__(self, input_shape=(4, 84, 84), output_dim=ACTION_DIM):
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


# ── Model loading ─────────────────────────────────────────────────────────────

def _build_model(dueling: bool, device: torch.device) -> nn.Module:
    cls = DuelingCNNDQN if dueling else CNNDQN
    return cls((4, 84, 84), ACTION_DIM).to(device)


def load_individual_model(model_path: str, config_path: str, device: torch.device):
    """
    Load a DQNAgent checkpoint produced by train.py.
    Auto-detects architecture from training_config.json.
    Returns (model, is_dueling).
    """
    with open(config_path) as f:
        cfg = json.load(f)
    dueling = cfg.get('dueling', False)
    model   = _build_model(dueling, device)
    ckpt    = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['policy_net_state_dict'])
    model.eval()
    return model, dueling


def load_merged_model(model_path: str, device: torch.device):
    """
    Load a merged model checkpoint produced by merge_models.py.
    Returns (model, metadata_dict).
    """
    ckpt    = torch.load(model_path, map_location=device, weights_only=False)
    dueling = ckpt.get('architecture', 'DuelingCNNDQN') == 'DuelingCNNDQN'
    model   = _build_model(dueling, device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt


# ── Environment ───────────────────────────────────────────────────────────────

def make_env(game: str, seed: int = 42):
    info = GAME_INFO[game]
    env  = gym.make(info['env_id'], full_action_space=True, frameskip=1)
    env  = AtariPreprocessing(env, noop_max=30, frame_skip=4,
                              screen_size=84, grayscale_obs=True,
                              terminal_on_life_loss=True, scale_obs=False)
    env  = FrameStack(env, 4)
    env.reset(seed=seed)
    return env


# ── Evaluation ────────────────────────────────────────────────────────────────

def _pick_action(model, state, valid_actions, device, epsilon):
    if np.random.random() < epsilon:
        return int(np.random.choice(valid_actions))
    with torch.no_grad():
        s = torch.from_numpy(np.array(state, dtype=np.uint8)).unsqueeze(0).to(device)
        q = model(s).cpu().numpy()[0]
    mask = np.full(ACTION_DIM, -np.inf)
    mask[valid_actions] = q[valid_actions]
    return int(np.argmax(mask))


def run_episodes(model, game: str, n_episodes: int, device,
                 seed: int = 42, epsilon: float = 0.01) -> list:
    """Run model on game for n_episodes. Returns list of episode rewards."""
    valid_actions = GAME_INFO[game]['valid_actions']
    env           = make_env(game, seed=seed)
    rewards       = []

    ep_iter = (tqdm(range(n_episodes), desc='    eps', leave=False)
               if HAS_TQDM else range(n_episodes))

    for _ in ep_iter:
        state, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action      = _pick_action(model, state, valid_actions, device, epsilon)
            state, r, term, trunc, _ = env.step(action)
            total      += r
            done        = term or trunc
        rewards.append(total)

    env.close()
    return rewards


def compute_stats(rewards: list) -> dict:
    a = np.array(rewards)
    return dict(
        mean=float(a.mean()), std=float(a.std()),
        median=float(np.median(a)),
        min=float(a.min()), max=float(a.max()),
        n=len(a),
    )


# ── Auto-discovery ────────────────────────────────────────────────────────────

def discover_variants(game: str) -> list:
    """Find all variant subdirectories that contain best_model.pth + training_config.json."""
    game_dir = os.path.join(OUTPUT_DIR, game)
    if not os.path.isdir(game_dir):
        return []
    variants = []
    for entry in sorted(os.listdir(game_dir)):
        sub = os.path.join(game_dir, entry)
        if (os.path.isdir(sub)
                and os.path.exists(os.path.join(sub, 'best_model.pth'))
                and os.path.exists(os.path.join(sub, 'training_config.json'))):
            variants.append(entry)
    return variants


# ── Report formatting ─────────────────────────────────────────────────────────

_COL = '-' * 86
_SEP = '=' * 86
_HDR = (f'{"Game":<16} {"Variant":<22}'
        f' {"Mean":>9} {"±Std":>9} {"Median":>9} {"Min":>9} {"Max":>9}')


def _row(game, variant, s):
    return (f'{game.capitalize():<16} {variant:<22}'
            f' {s["mean"]:>+9.2f} {s["std"]:>9.2f} {s["median"]:>+9.2f}'
            f' {s["min"]:>+9.2f} {s["max"]:>+9.2f}')


def build_report(results: dict, merged_results: dict, args) -> str:
    lines = [
        _SEP,
        'ATARI MODEL EVALUATION REPORT',
        _SEP,
        f'Generated  : {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'Games      : {", ".join(args.games)}',
        f'Episodes   : {args.episodes} per model',
        f'Epsilon    : {args.epsilon}',
        f'Device     : {args.device or "auto"}',
        '',
        'INDIVIDUAL MODELS',
        _COL,
        _HDR,
        _COL,
    ]

    any_individual = False
    for game in args.games:
        if game not in results or not results[game]:
            continue
        for variant, s in results[game].items():
            lines.append(_row(game, variant, s))
            any_individual = True
        lines.append('')

    if not any_individual:
        lines += ['  (none found)', '']

    if merged_results:
        lines += ['MERGED MODELS', _COL, _HDR, _COL]
        for label, game_stats in merged_results.items():
            lines.append(f'  [{label}]')
            means = []
            for game, s in game_stats.items():
                lines.append(_row(game, '(merged)', s))
                means.append(s['mean'])
            lines.append(
                f'  {"AVERAGE ACROSS GAMES":<38} {np.mean(means):>+9.2f}')
            lines.append('')

    lines.append(_SEP)
    return '\n'.join(lines)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate trained Atari DQN models on N episodes per game.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--games', nargs='+', required=True,
                   choices=list(GAME_INFO),
                   metavar='GAME',
                   help='Games to evaluate  (e.g. breakout pong freeway)')
    p.add_argument('--variants', nargs='+', default=None,
                   metavar='VARIANT',
                   help='Specific variants to test  (e.g. ddqn ddqn_duel ddqn_duel_per). '
                        'Default: auto-discover all available variants per game.')
    p.add_argument('--episodes', type=int, default=100,
                   help='Episodes per model per game  (default: 100)')
    p.add_argument('--epsilon', type=float, default=0.01,
                   help='Exploration rate during evaluation  (default: 0.01)')
    p.add_argument('--merged', nargs='+', default=None, metavar='PATH',
                   help='Path(s) to merged model .pth file(s) to also evaluate.')
    p.add_argument('--no-individual', action='store_true',
                   help='Skip individual model evaluation — only evaluate --merged models.')
    p.add_argument('--output', type=str, default=None,
                   help='Output text file path.  '
                        'Default: results_YYYYMMDD_HHMMSS.txt in code/')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed  (default: 42)')
    p.add_argument('--device', type=str, default=None,
                   choices=['auto', 'cpu', 'cuda', 'mps'],
                   help='Compute device  (default: auto-detect)')
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Device
    if args.device and args.device != 'auto':
        device = torch.device(args.device)
    else:
        device = torch.device(
            'mps'  if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available()          else
            'cpu'
        )
    args.device = str(device)   # store resolved device for report

    # Default output path
    if args.output is None:
        ts          = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(_HERE, f'results_{ts}.txt')

    print(f'Device   : {device}')
    print(f'Games    : {args.games}')
    print(f'Episodes : {args.episodes}')
    print(f'Output   : {args.output}')

    results        = {}
    merged_results = {}

    # ── Individual models ─────────────────────────────────────────────────────
    if args.no_individual:
        print('\n=== Individual Models === (skipped)')
    else:
        print('\n=== Individual Models ===')
        for game in args.games:
            variants = args.variants if args.variants else discover_variants(game)
            if not variants:
                print(f'  [WARN] No variants found for {game} in {OUTPUT_DIR}/{game}/')
                continue

            results[game] = {}
            for variant in variants:
                model_path  = os.path.join(OUTPUT_DIR, game, variant, 'best_model.pth')
                config_path = os.path.join(OUTPUT_DIR, game, variant, 'training_config.json')

                if not os.path.exists(model_path):
                    print(f'  [SKIP] {game}/{variant}  — best_model.pth not found')
                    continue
                if not os.path.exists(config_path):
                    print(f'  [SKIP] {game}/{variant}  — training_config.json not found')
                    continue

                print(f'  {game}/{variant} ({args.episodes} eps)')
                model, _ = load_individual_model(model_path, config_path, device)
                rewards   = run_episodes(model, game, args.episodes, device,
                                         seed=args.seed, epsilon=args.epsilon)
                s         = compute_stats(rewards)
                results[game][variant] = s
                print(f'    → mean={s["mean"]:+.2f}  std={s["std"]:.2f}'
                      f'  min={s["min"]:+.2f}  max={s["max"]:+.2f}')
                del model

    # ── Merged models ─────────────────────────────────────────────────────────
    if args.merged:
        print('\n=== Merged Models ===')
        for mp in args.merged:
            if not os.path.exists(mp):
                print(f'  [SKIP] {mp}  — file not found')
                continue

            label = os.path.basename(mp)
            print(f'  Evaluating: {label}')
            model, meta = load_merged_model(mp, device)
            merged_results[label] = {}

            for game in args.games:
                print(f'    {game} ({args.episodes} eps)')
                rewards = run_episodes(model, game, args.episodes, device,
                                       seed=args.seed, epsilon=args.epsilon)
                s       = compute_stats(rewards)
                merged_results[label][game] = s
                print(f'    → mean={s["mean"]:+.2f}  std={s["std"]:.2f}'
                      f'  min={s["min"]:+.2f}  max={s["max"]:+.2f}')

            del model

    # ── Write report ──────────────────────────────────────────────────────────
    report = build_report(results, merged_results, args)
    with open(args.output, 'w') as f:
        f.write(report + '\n')

    print('\n' + report)
    print(f'\nReport saved → {args.output}')


if __name__ == '__main__':
    main()
