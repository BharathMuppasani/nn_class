# DQN Project — Comprehensive Fix & Enhancement Plan

## Context

After an initial round of bug fixes (tau, epsilon decay, weight broadcast, hyperparameter defaults) that got CartPole training working, a full audit revealed **9 additional bugs** across all DQN variants and Atari support. These range from shape mismatches that crash PER training, to missing Atari preprocessing that would prevent convergence on Pong. The user also wants a README documenting the project.

---

## Recently Completed (Round 1)

| Fix | File | Status |
|-----|------|--------|
| Weight broadcast `[B]*[B,1]→[B,B]` — added `weights.unsqueeze(1)` | `dqn_agent.py:83` | Done |
| `tau = 0.9` → `0.005` | `dqn_agent.py:60` | Done |
| Epsilon decay moved from per-step to per-episode | `dqn_agent.py` + `train.py` | Done |
| Default hyperparameters (buffer-size, target-update, epsilon-decay) | `train.py` | Done |
| Matplotlib live plot fix (TkAgg backend, show(block=False)) | `train.py` | Done |

---

## Round 2 — Remaining Fixes

### Fix 1 (Critical): PER buffer shape mismatch — crashes `gather()`
**File:** `code/replay_buffer.py` lines 78-82

PrioritizedReplayBuffer uses `np.stack` on scalar actions/rewards/dones, producing shape `[B]`. But the agent's `gather(1, actions)` requires `[B, 1]`, and `rewards [B] + next_q_values [B,1]` broadcasts to `[B, B]`.

**Fix:** Normalize both buffers to use `np.stack` for states (works for 1D and 3D) and explicit `unsqueeze(1)` for scalar fields:

```python
# Both buffers — sample() method:
states = torch.FloatTensor(np.stack([e.state for e in experiences])).to(self.device)
actions = torch.LongTensor(np.array([e.action for e in experiences])).unsqueeze(1).to(self.device)
rewards = torch.FloatTensor(np.array([e.reward for e in experiences])).unsqueeze(1).to(self.device)
next_states = torch.FloatTensor(np.stack([e.next_state for e in experiences])).to(self.device)
dones = torch.FloatTensor(np.array([e.done for e in experiences])).unsqueeze(1).to(self.device)
```

This makes ReplayBuffer and PrioritizedReplayBuffer produce identical tensor shapes, and fixes the Atari `np.vstack` bug (Fix 6 below) in the same change.

---

### Fix 2 (Critical): `update_priorities` receives wrong shape
**File:** `code/dqn_agent.py` line 117

`td_errors` is `[B, 1]`. Passing `(B, 1)` numpy array to `priorities[indices]` (shape `(B,)`) causes bad assignment.

**Fix:** Squeeze before passing:
```python
self.memory.update_priorities(indices, td_errors.squeeze(1).detach().cpu().numpy())
```

---

### Fix 3 (Critical): No pixel normalization for Atari
**File:** `code/dqn.py` CNNDQN.forward()

Atari frames are `[0, 255]` uint8. Without normalization, activations explode.

**Fix:** Add at start of CNNDQN.forward():
```python
x = x.float() / 255.0
```

---

### Fix 4 (High): Replay buffer OOM on Atari — stores float32
**File:** `code/replay_buffer.py` — both `push()` methods

Each Atari state as float32 = ~113KB. 100K buffer × 2 states = ~22GB. Storing as uint8 = ~5.5GB.

**Fix:** Remove `dtype=np.float32` from push() — let numpy preserve the original dtype (uint8 for Atari, float64 for CartPole). The `torch.FloatTensor()` conversion in `sample()` handles float conversion automatically.

```python
def push(self, state, action, reward, next_state, done):
    state = np.array(state)
    next_state = np.array(next_state)
    ...
```

---

### Fix 5 (High): Missing standard Atari wrappers
**File:** `code/train.py` create_env()

Replace manual ResizeObservation + GrayScaleObservation with gymnasium's `AtariPreprocessing` which bundles:
- Noop reset (random no-ops at start)
- Frame skipping (4) with max-pool over last 2 frames
- Grayscale conversion
- Resize to 84×84

Then wrap with `FrameStack(env, 4)`.

```python
from gymnasium.wrappers import AtariPreprocessing, FrameStack

if is_atari:
    env = AtariPreprocessing(env, noop_max=30, frame_skip=4,
                             screen_size=84, grayscale_obs=True,
                             terminal_on_life_loss=True, scale_obs=False)
    env = FrameStack(env, 4)
    state_dim = (4, 84, 84)
```

Note: `scale_obs=False` because CNNDQN handles normalization internally (Fix 3). `terminal_on_life_loss=True` for faster learning signal.

Also update `test.py` `create_env()` with the same wrappers (but `terminal_on_life_loss=False` for fair evaluation).

---

### Fix 6 (High): `ReplayBuffer.sample()` uses `np.vstack` — breaks for images
**File:** `code/replay_buffer.py` line 23

`np.vstack` on `(4, 84, 84)` arrays produces `(B*4, 84, 84)` instead of `(B, 4, 84, 84)`.

**Fix:** Already covered by Fix 1 — switching to `np.stack` for states/next_states fixes this.

---

### Fix 7 (Medium): No DuelingCNNDQN for Atari
**File:** `code/dqn.py`

`use_dueling=True` is silently ignored for image inputs. Add a `DuelingCNNDQN` class:
- Reuse CNNDQN's conv layers for feature extraction
- Split into value stream (FC→1) and advantage stream (FC→action_dim)
- Combine: `Q = V + A - mean(A)`

```python
class DuelingCNNDQN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        c, h, w = input_shape
        # Same conv backbone as CNNDQN
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        linear_input_size = self._conv_output_size(c, h, w)

        # Value stream
        self.fc_value = nn.Linear(linear_input_size, 512)
        self.value = nn.Linear(512, 1)
        # Advantage stream
        self.fc_adv = nn.Linear(linear_input_size, 512)
        self.advantage = nn.Linear(512, output_dim)
```

Update `dqn_agent.py` to select `DuelingCNNDQN` when `is_image and use_dueling`.

---

### Fix 8 (Medium): train.py hardcodes all variants to False
**File:** `code/train.py`

Add CLI flags and pass them through:
```
--double       Enable Double DQN
--dueling      Enable Dueling DQN
--priority     Enable Prioritized Experience Replay
```

Remove the hardcoded `False` values and use `args.double`, `args.dueling`, `args.priority`.

---

### Fix 9 (Medium): test.py can't load variant models
**File:** `code/test.py`

Add `--double`, `--dueling` CLI flags so the correct network architecture is initialized before loading checkpoints. Also save variant flags in the checkpoint (dqn_agent.py save/load) so they can be auto-detected in the future.

---

### Fix 10: Create README.md
**File:** `Project/README.md` (new)

Contents:
- Project description (DQN variants for CartPole and Atari)
- Installation / environment setup (`conda activate rl_dqn`)
- Training commands for CartPole and Atari (plain DQN, DDQN, Dueling, PER)
- Testing/evaluation commands
- Architecture overview (DQN, CNNDQN, DuelingDQN, DuelingCNNDQN)
- Results summary table
- File structure

---

## Implementation Order

| Step | File(s) | Fix | Why this order |
|------|---------|-----|----------------|
| 1 | `replay_buffer.py` | Fix 1 + 4 + 6: Normalize both buffers (np.stack, unsqueeze, remove float32 forcing) | Foundation — all training paths depend on correct buffer shapes |
| 2 | `dqn_agent.py` | Fix 2: Squeeze td_errors for update_priorities | Pairs with buffer fix |
| 3 | `dqn.py` | Fix 3: Add `/255.0` in CNNDQN.forward() | Atari prerequisite |
| 4 | `dqn.py` | Fix 7: Add DuelingCNNDQN class | New architecture |
| 5 | `dqn_agent.py` | Update network selection for DuelingCNNDQN + save variant flags | Connect new architecture |
| 6 | `train.py` | Fix 5 + 8: AtariPreprocessing wrappers + CLI variant flags | Enable Atari + all variants |
| 7 | `test.py` | Fix 9: Add variant flags + matching Atari wrappers | Evaluation support |
| 8 | `README.md` | Fix 10: Project documentation | Final deliverable |

## Files Modified
- `code/replay_buffer.py` — Normalize shapes, remove float32 forcing (Fix 1, 4, 6)
- `code/dqn.py` — Pixel normalization + DuelingCNNDQN class (Fix 3, 7)
- `code/dqn_agent.py` — td_errors squeeze, DuelingCNNDQN selection, save variant flags (Fix 2, 5)
- `code/train.py` — AtariPreprocessing, CLI variant flags (Fix 5, 8)
- `code/test.py` — Variant flags, Atari wrappers for eval (Fix 9)
- `README.md` — New file (Fix 10)

---

## Verification

### CartPole (plain DQN):
```bash
conda activate rl_dqn && cd /Users/bittu/Desktop/GitHub/nn_class/Project/code
python train.py --env CartPole-v1 --episodes 500 --save-dir ../output/cartpole_dqn
```

### CartPole (all variants):
```bash
python train.py --env CartPole-v1 --episodes 500 --double --save-dir ../output/cartpole_ddqn
python train.py --env CartPole-v1 --episodes 500 --double --dueling --save-dir ../output/cartpole_ddqn_duel
python train.py --env CartPole-v1 --episodes 500 --double --priority --save-dir ../output/cartpole_ddqn_per
python train.py --env CartPole-v1 --episodes 500 --double --dueling --priority --save-dir ../output/cartpole_ddqn_duel_per
```

### Pong (Atari):
```bash
python train.py --env PongNoFrameskip-v4 --episodes 1000 --lr 1e-4 --buffer-size 100000 --batch-size 32 --save-dir ../output/pong_dqn
```

### Evaluation:
```bash
python test.py --env CartPole-v1 --load-path ../output/cartpole_dqn/best_model.pth
python test.py --env PongNoFrameskip-v4 --load-path ../output/pong_dqn/best_model.pth
```
