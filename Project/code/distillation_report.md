# Policy Distillation — Model Merging Report

## Overview

Three independently-trained expert models (Breakout, Pong, Freeway) are merged into a
single student model using **multi-task policy distillation**. The student is trained to
reproduce the Q-value decisions of all three expert teachers simultaneously.

This is **not** the CRL-Atari HTCL/Taylor method. Everything lives in `code/merge_models.py`.

---

## Why not CRL-Atari's HTCL?

CRL-Atari implements **Hierarchical Taylor Consolidation (HTCL)**. Its Taylor update formula is:

```
Δw* = (H + λI)⁻¹ [ λ(w_local - w_global) - g ]
w_global ← w_global + η · Δw*
```

Where:
- `H` = diagonal Hessian (squared gradients) — captures which weights were important
- `λ` = regularisation strength
- `g` = gradient of TD loss on combined transitions
- `η` = step size

**HTCL's fundamental assumption**: all models share the same evolving initialisation.
You train task 1, then continue training for task 2, 3... on a single model. The Hessian
tracks which weights were critical for previous tasks and protects them.

**Why it fails here**: Breakout, Pong, and Freeway models were trained completely
independently from random initialisations. Their weight spaces are unrelated — neuron 47
in the Breakout model encodes something entirely different from neuron 47 in the Pong model.
With large λ, HTCL reduces to linear interpolation of unrelated weights:

```
w_global += η * (w_pong - w_breakout)   →   destroys both policies
```

Result with HTCL: Breakout=+10, Pong=-21 (random/adversarial play).

---

## Policy Distillation — How It Works

### Step 1: Collect states from each game

Each expert model plays its game for N steps. We save the raw pixel frames (observations).
These become the training inputs for the student.

```python
collect_transitions(breakout_model, 'breakout', n_steps=20000)
collect_transitions(pong_model,     'pong',     n_steps=20000)
collect_transitions(freeway_model,  'freeway',  n_steps=20000)
```

### Step 2: Pre-compute teacher Q-values (frozen)

For every saved frame, ask each frozen expert: "what Q-value do you assign to each action?"

```python
teacher_q_breakout = breakout_model(breakout_states)  # [N, 18], no gradient
teacher_q_pong     = pong_model(pong_states)           # [N, 18], no gradient
teacher_q_freeway  = freeway_model(freeway_states)     # [N, 18], no gradient
```

`Q[action]` ≈ expected total future reward if this action is taken now.
These are cached once and never updated. The experts are frozen teachers.

### Step 3: Train the student to copy Q-values

The student (same DuelingCNNDQN architecture) processes each frame and minimises
mean squared error against the teacher's Q-values — **only on valid actions per game**:

```
Breakout frame → student → Q[[0,1,3,4]]      vs  breakout_teacher → Q[[0,1,3,4]]
Pong frame     → student → Q[[0,1,3,4,11,12]] vs  pong_teacher    → Q[[0,1,3,4,11,12]]
Freeway frame  → student → Q[[0,2,5]]         vs  freeway_teacher  → Q[[0,2,5]]
```

Valid actions per game (18-action full Atari space with masking):

| Game     | Valid action indices       | Meaning                          |
|----------|---------------------------|----------------------------------|
| Breakout | [0, 1, 3, 4]              | NOOP, FIRE, RIGHT, LEFT          |
| Pong     | [0, 1, 3, 4, 11, 12]      | NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE |
| Freeway  | [0, 2, 5]                 | NOOP, UP, DOWN                   |

---

## Problems Encountered and Fixes

### Problem 1: Q-value scale imbalance

| Game     | Q-value range    | Span  | Relative MSE weight |
|----------|-----------------|-------|---------------------|
| Breakout | [-2.17, +16.89] | ~19   | ~34×                |
| Pong     | [-0.78, +2.46]  | ~3.2  | 1×                  |
| Freeway  | [-0.1, +0.5]    | ~0.6  | ~0.03×              |

Without correction, Breakout's large Q-values dominate the gradient ~34×. The student
learns Breakout well but barely updates for Pong and Freeway.

**Fix**: Compute separate MSE per game and average them equally:

```python
loss_breakout = MSE(student_q[valid_breakout], teacher_q[valid_breakout])
loss_pong     = MSE(student_q[valid_pong],     teacher_q[valid_pong])

total_loss = (loss_breakout + loss_pong) / 2   # equal weight per game
```

### Problem 2: Overfitting

The DuelingCNNDQN has 3.3M parameters. With only 20K training states per game, the
network memorises the specific training frames rather than learning generalisable features.

Symptom: training loss fell from 0.40 → 0.001 (40× reduction) but Pong score got
*worse* (-18 → -19.4) as the model overfit.

**Fix**: L2 weight decay regularisation (`weight_decay=1e-4`) added to Adam optimiser.
This penalises large weights and prevents the network from memorising training states.

### Problem 3: Wrong model selection criterion

Saving the lowest training-loss checkpoint saved the most overfit model.

**Fix**: Evaluate actual game performance (run N real episodes) every 10 epochs.
Save the checkpoint with the highest **total game score**, not lowest training loss.

```
Epoch  20/60: loss=0.005151 *  [breakout=+17.0  pong=-12.0] <-- best
Epoch  30/60: loss=0.002791 *  [breakout=+16.0  pong=+10.0] <-- best  ← saved this
Epoch  40/60: loss=0.001586 *  [breakout=+14.0  pong=+8.0]            ← overfit
Epoch  60/60: loss=0.000939 *  [breakout=+14.4  pong=-19.4]           ← overfit
```

---

## Architecture

Student and all teachers use the same **DuelingCNNDQN** architecture:

```
Input: (4, 84, 84) — 4 stacked greyscale frames at 84×84

Conv backbone (shared):
  Conv2d(4→32,  kernel=8, stride=4)  → ReLU
  Conv2d(32→64, kernel=4, stride=2)  → ReLU
  Conv2d(64→64, kernel=3, stride=1)  → ReLU
  Flatten → 3136-dim feature vector

Dueling streams:
  Value stream:     Linear(3136→512) → ReLU → Linear(512→1)
  Advantage stream: Linear(3136→512) → ReLU → Linear(512→18)

Output: V(s) + A(s,a) - mean(A(s,:))   shape: [18]   (full Atari action space)

Total parameters: 3,300,019
```

At inference, invalid actions are masked to -∞ before argmax, so only valid actions
are ever selected regardless of their raw Q-values.

---

## Final Results

| Model         | Breakout | Pong    | Notes                    |
|---------------|----------|---------|--------------------------|
| Expert (solo) | +17.6    | +21.0   | Individual specialists   |
| Merged        | +15.8    | +10.2   | Single model, both games |
| Retention     | 90%      | 49%     | vs expert baseline       |

The merged model plays both games from a single set of 3.3M shared parameters with
no game-ID input — it infers which game it is playing purely from visual features.

---

## Training Configuration (best run)

```bash
python merge_models.py \
    --games breakout pong \
    --variant ddqn_duel \
    --method distill \
    --collect-steps 20000 \
    --distill-epochs 60 \
    --distill-lr 3e-4 \
    --distill-wd 1e-4 \
    --distill-init warmstart \
    --eval-every 10 \
    --name breakout_pong_distill_v4
```

| Hyperparameter  | Value   | Reason                                          |
|-----------------|---------|-------------------------------------------------|
| collect_steps   | 20,000  | State coverage; more = better generalisation    |
| distill_epochs  | 60      | Loss still improving at 30; plateau around 40-50|
| distill_lr      | 3e-4    | Standard Adam LR for supervised learning        |
| distill_wd      | 1e-4    | L2 regularisation to prevent overfitting        |
| distill_init    | warmstart | Initialise student from breakout model (faster convergence) |
| eval_every      | 10      | Check game performance every 10 epochs          |
| batch_size      | 256     | Fixed in code                                   |
| LR schedule     | CosineAnnealingLR (η_min = lr × 0.1) | Smooth decay |
