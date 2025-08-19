# Metta Training Equations Reference

This document provides a comprehensive reference of all calculations performed in the Metta training pipeline, starting from configuration parameters and showing how derived values are computed.

## Table of Contents
1. [Input Configuration Parameters](#input-configuration-parameters)
2. [Environment and Worker Calculations](#environment-and-worker-calculations)
3. [Experience Buffer Calculations](#experience-buffer-calculations)
4. [Training Loop Calculations](#training-loop-calculations)
5. [PPO Algorithm Calculations](#ppo-algorithm-calculations)
6. [Prioritized Experience Replay](#prioritized-experience-replay)
7. [Learning Rate Scheduling](#learning-rate-scheduling)

---

## Input Configuration Parameters

### From `trainer.yaml`:
```yaml
# Core parameters
num_workers: 16                         # Number of parallel worker processes
batch_size: 524288                      # Total environment steps before training
minibatch_size: 16384                   # Steps per gradient update
bptt_horizon: 64                        # Sequence length for BPTT
update_epochs: 1                        # Times to iterate over batch
forward_pass_minibatch_target_size: 4096  # Target agents per forward pass
async_factor: 2                         # Double-buffering factor
total_timesteps: 10_000_000_000         # Total training steps

# PPO parameters
gamma: 0.977                            # Discount factor
gae_lambda: 0.916                       # GAE lambda for advantage estimation
clip_coef: 0.1                          # PPO clip coefficient
ent_coef: 0.0021                        # Entropy coefficient
vf_coef: 0.44                           # Value function coefficient

# Prioritized replay
prio_alpha: 0.0                         # Prioritization strength (0 = uniform)
prio_beta0: 0.6                         # Initial importance sampling correction
```

### From Environment Config:
```yaml
game:
  num_agents: 3                          # Agents per environment (varies by task)
  obs_width: 11                          # Observation width
  obs_height: 11                         # Observation height
  max_steps: 1000                        # Max steps per episode
```

---

## Environment and Worker Calculations

### 1. Target Batch Size (environments per forward pass)
```python
target_batch_size = forward_pass_minibatch_target_size // num_agents

# Minimum constraint check
if target_batch_size < max(2, num_workers):
    target_batch_size = num_workers
```

### 2. Actual Batch Size (rounded to worker multiple)
```python
batch_size_envs = (target_batch_size // num_workers) * num_workers
```

### 3. Total Environments Created
```python
num_envs = batch_size_envs * async_factor
```

### 4. Environments Per Worker
```python
envs_per_worker = num_envs // num_workers
```

### Example with 3 agents:
- `target_batch_size = 4096 // 3 = 1365`
- `batch_size_envs = (1365 // 16) * 16 = 1360`
- `num_envs = 1360 * 2 = 2720`
- `envs_per_worker = 2720 // 16 = 170`

---

## Experience Buffer Calculations

### 1. Total Segments (sequences)
```python
segments = batch_size // bptt_horizon
```
- Example: `524288 // 64 = 8192` segments

### 2. Segments Per Minibatch
```python
minibatch_segments = minibatch_size // bptt_horizon
```
- Example: `16384 // 64 = 256` segments

### 3. Number of Minibatches
```python
num_minibatches = segments // minibatch_segments
```
- Example: `8192 // 256 = 32` minibatches

### 4. Buffer Shape
```python
buffer_shape = [segments, bptt_horizon, ...feature_dimensions...]
# Example: [8192, 64, ...obs_shape, actions, rewards, etc...]
```

### 5. Episode Index Assignment
```python
# Each agent gets assigned to a segment
ep_indices[agent_id] = agent_id % segments
free_idx = num_agents % segments
```

### 6. Validation Constraints
```python
# Must have enough segments for all agents
assert segments >= total_agents

# Segments must be divisible by minibatch segments
assert segments % minibatch_segments == 0

# Minibatch size must be divisible by BPTT horizon
assert minibatch_size % bptt_horizon == 0
```

---

## Training Loop Calculations

### 1. Steps Per Environment Collection
```python
steps_per_env = batch_size // num_envs
```
- Example: `524288 // 2720 = 192` steps

### 2. Total Agent Steps Per Batch
```python
agent_steps_per_batch = batch_size * num_agents
```
- Example: `524288 * 3 = 1,572,864` agent steps

### 3. Gradient Updates Per Batch
```python
gradient_updates_per_batch = num_minibatches * update_epochs
```
- Example: `32 * 1 = 32` updates

### 4. Agent Experiences Per Gradient
```python
experiences_per_gradient = agent_steps_per_batch // gradient_updates_per_batch
```
- Example: `1,572,864 // 32 = 49,152` experiences

### 5. Total Training Epochs
```python
total_epochs = total_timesteps // batch_size
```
- Example: `10,000,000,000 // 524,288 = 19,073` epochs

---

## PPO Algorithm Calculations

### 1. Generalized Advantage Estimation (GAE)
```python
# For each timestep t in trajectory:
delta[t] = rho[t] * (rewards[t+1] + gamma * values[t+1] * (1-done[t+1]) - values[t])

# Backward pass to compute advantages
advantages[T-1] = delta[T-1]
for t in range(T-2, -1, -1):
    advantages[t] = delta[t] + gamma * gae_lambda * c[t] * advantages[t+1] * (1-done[t+1])
```

Where:
- `rho = min(vtrace_rho_clip, importance_sampling_ratio)`
- `c = min(vtrace_c_clip, importance_sampling_ratio)`
- Default: `vtrace_rho_clip = 1.0`, `vtrace_c_clip = 1.0`

### 2. Returns Calculation
```python
returns = advantages + values
```

### 3. Advantage Normalization
```python
if norm_adv:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### 4. PPO Clipped Loss
```python
ratio = exp(log_prob - old_log_prob)
clipped_ratio = clip(ratio, 1 - clip_coef, 1 + clip_coef)
policy_loss = -min(ratio * advantages, clipped_ratio * advantages).mean()
```

### 5. Value Function Loss
```python
if clip_vloss:
    v_clipped = old_values + clip(values - old_values, -vf_clip_coef, vf_clip_coef)
    v_loss = max((values - returns)^2, (v_clipped - returns)^2).mean()
else:
    v_loss = ((values - returns)^2).mean()
```

### 6. Entropy Loss
```python
entropy_loss = -entropy.mean()
```

### 7. Total Loss
```python
total_loss = policy_loss + vf_coef * v_loss + ent_coef * entropy_loss
```

---

## Prioritized Experience Replay

### 1. Priority Calculation
```python
# Advantage magnitude as priority
adv_magnitude = abs(advantages).sum(dim=1)  # Sum over time dimension
prio_weights = adv_magnitude ^ prio_alpha
prio_probs = (prio_weights + 1e-6) / (sum(prio_weights) + 1e-6)
```

### 2. Importance Sampling Weights
```python
# Annealed beta calculation
total_epochs = max(1, total_timesteps // batch_size)
anneal_beta = prio_beta0 + (1 - prio_beta0) * prio_alpha * epoch / total_epochs

# IS weights for sampled minibatch
IS_weights = (segments * prio_probs[sampled_indices]) ^ (-anneal_beta)
```

### 3. Weighted Advantage
```python
weighted_advantages = IS_weights * normalized_advantages
```

---

## Learning Rate Scheduling

### 1. Cosine Schedule (default for learning rate)
```python
min_lr = 0.00003
initial_lr = 0.000457
progress = epoch / total_epochs
lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(pi * progress))
```

### 2. Linear Schedule (for entropy coefficient)
```python
min_value = 0.0
initial_value = 0.0021
value = initial_value - (initial_value - min_value) * (epoch / total_epochs)
```

### 3. Logarithmic Schedule (for PPO clip)
```python
min_value = 0.05
decay_rate = 0.1
initial_value = 0.1
value = max(min_value, initial_value * exp(-decay_rate * epoch / total_epochs))
```

---

## Scaling with Different Agent Counts

### Impact of `num_agents` on calculations:

| Parameter | Formula | 3 agents | 24 agents |
|-----------|---------|----------|-----------|
| Target envs per forward | `4096 / num_agents` | 1365 | 170 |
| Actual envs per forward | Rounded to worker multiple | 1360 | 160 |
| Total environments | `actual * async_factor` | 2720 | 320 |
| Envs per worker | `total_envs / num_workers` | 170 | 20 |
| Steps per env | `batch_size / num_envs` | 192 | 1638 |
| Agent steps per batch | `batch_size * num_agents` | 1.57M | 12.6M |
| Agent experiences per gradient | `agent_steps / num_minibatches` | 49K | 393K |

### Key Insights:
- More agents = fewer environments needed
- More agents = more agent experiences per batch
- Same number of gradient updates regardless of agent count
- Worker utilization varies significantly with agent count

---

## Memory Requirements

### Buffer Memory Usage
```python
# Per element in buffer (assuming float32)
element_size = 4  # bytes

# Observation buffer
obs_memory = segments * bptt_horizon * obs_width * obs_height * element_size
# Example: 8192 * 64 * 11 * 11 * 4 = 253MB per observation type

# Total buffer (all features)
total_features = observations + actions + rewards + values + log_probs + dones + ...
buffer_memory = segments * bptt_horizon * total_features * element_size
```

### GPU Memory During Training
```python
# Minibatch memory
minibatch_memory = minibatch_segments * bptt_horizon * total_features * element_size
# Example: 256 * 64 * features * 4

# With double buffering (async_factor=2)
runtime_memory = 2 * minibatch_memory + model_parameters + optimizer_state
```

---

## Common Configuration Adjustments

### For Large Agent Counts (24+):
- Consider increasing `num_workers` for better parallelism
- May reduce `batch_size` due to higher agent experiences
- Could increase `minibatch_size` to process more agents per gradient

### For Small Agent Counts (1-2):
- May need larger `batch_size` for sufficient experiences
- Consider reducing `num_workers` if underutilized
- Might decrease `forward_pass_minibatch_target_size`

### For Memory Constraints:
- Reduce `batch_size` to shrink buffer
- Enable `cpu_offload` to move buffer to CPU
- Reduce `bptt_horizon` for shorter sequences
- Decrease `minibatch_size` for smaller GPU batches

### For Faster Training:
- Enable `compile=true` with `compile_mode="reduce-overhead"`
- Increase `num_workers` (if CPU permits)
- Increase `async_factor` (if memory permits)
- Adjust `minibatch_size` for optimal GPU utilization