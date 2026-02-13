# DQN Variants for CartPole and Atari

Implementation of Deep Q-Network (DQN) with multiple algorithmic extensions, built with PyTorch and Gymnasium.

## Variants

| Variant | Description |
|---------|-------------|
| **Plain DQN** | Vanilla DQN with experience replay and target network |
| **Double DQN** | Decouples action selection (policy net) from value estimation (target net) to reduce overestimation bias |
| **Dueling DQN** | Separates Q-value into state-value V(s) and advantage A(s,a) streams |
| **PER** | Prioritized Experience Replay — samples transitions with higher TD-error more frequently |

All variants can be freely combined (e.g., Double + Dueling + PER).

## Setup

```bash
conda activate rl_dqn
cd Project/code
```

## Training

### CartPole

```bash
# Plain DQN
python train.py --env CartPole-v1 --episodes 500 --save-dir ../output/cartpole_dqn

# Double DQN
python train.py --env CartPole-v1 --episodes 500 --double --save-dir ../output/cartpole_ddqn

# Double + Dueling DQN
python train.py --env CartPole-v1 --episodes 500 --double --dueling --save-dir ../output/cartpole_ddqn_duel

# Double DQN + PER
python train.py --env CartPole-v1 --episodes 500 --double --priority --save-dir ../output/cartpole_ddqn_per

# Double + Dueling DQN + PER
python train.py --env CartPole-v1 --episodes 500 --double --dueling --priority --save-dir ../output/cartpole_ddqn_duel_per
```

### Atari (Pong)

```bash
python train.py --env PongNoFrameskip-v4 --episodes 1000 --lr 1e-4 --buffer-size 100000 --batch-size 32 --save-dir ../output/pong_dqn
```

## Evaluation

```bash
python test.py --env CartPole-v1 --load-path ../output/cartpole_dqn/best_model.pth
python test.py --env PongNoFrameskip-v4 --load-path ../output/pong_dqn/best_model.pth
```

Variant flags are auto-detected from saved checkpoints.

## Architecture

### MLP (CartPole)
- **DQN**: Input(4) -> FC(128) -> FC(128) -> Output(2)
- **DuelingDQN**: Shared FC(128x2) -> Value stream(128->1) + Advantage stream(128->2) -> Q = V + A - mean(A)

### CNN (Atari)
- **CNNDQN**: Conv(32,8x8,s4) -> Conv(64,4x4,s2) -> Conv(64,3x3,s1) -> FC(512) -> Output
- **DuelingCNNDQN**: Same conv backbone -> Value stream(512->1) + Advantage stream(512->actions)

Input normalization (/255.0) is handled inside the CNN forward pass.

## File Structure

```
Project/
├── code/
│   ├── train.py          # Training loop with live plotting
│   ├── test.py           # Evaluation with rendering
│   ├── dqn.py            # Network architectures (DQN, CNNDQN, DuelingDQN, DuelingCNNDQN)
│   ├── dqn_agent.py      # Agent logic (action selection, update, soft target sync)
│   └── replay_buffer.py  # ReplayBuffer + PrioritizedReplayBuffer
├── output/               # Saved models, logs, and plots per variant
└── README.md
```

## Key Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Learning rate | 3e-4 | Use 1e-4 for Atari |
| Gamma | 0.99 | Discount factor |
| Buffer size | 100,000 | |
| Batch size | 128 | Use 32 for Atari |
| Epsilon decay | 0.99 | Per-episode multiplicative decay |
| Tau | 0.005 | Soft target update rate |
| Target update freq | 1 | Steps between soft updates |
