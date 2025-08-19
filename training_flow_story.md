# The Metta Training Journey: From Configuration to Gradients

This document tells the story of how a single training iteration flows through the Metta codebase, tracking every calculation from initial configuration to final gradient update.

## 🎬 Scene 1: Configuration Loading
*Where our journey begins with YAML files and Hydra*

### Step 1.1: Loading Base Configuration
**Location**: `metta/rl/trainer.py:119-124`

The trainer loads configuration from `configs/trainer/trainer.yaml`:
```python
num_workers = 16
batch_size = 524,288
forward_pass_minibatch_target_size = 4,096
async_factor = 2
```

### Step 1.2: Environment Configuration
**Location**: `metta/rl/trainer.py:121` (curriculum loading)

From the curriculum, we get the environment config:
```python
num_agents = 3  # From configs/env/mettagrid/simple.yaml:12
obs_width = 11
obs_height = 11
max_steps = 1,000
```

---

## 🏗️ Scene 2: Environment Setup
*Creating the parallel environments for experience collection*

### Step 2.1: Calculate Target Batch Size
**Location**: `metta/utils/batch.py:17-19`

First, we figure out how many environments we want per forward pass:
```python
target_batch_size = forward_pass_minibatch_target_size // num_agents
target_batch_size = 4,096 // 3 = 1,365

# Safety check for minimum batch size
if target_batch_size < max(2, num_workers):  # Line 18-19
    target_batch_size = num_workers
```

### Step 2.2: Round to Worker Multiple
**Location**: `metta/utils/batch.py:21`

Ensure even distribution across workers:
```python
batch_size_envs = (target_batch_size // num_workers) * num_workers
batch_size_envs = (1,365 // 16) * 16 = 1,360
```

### Step 2.3: Calculate Total Environments
**Location**: `metta/utils/batch.py:22`

Apply async factor for double-buffering:
```python
num_envs = batch_size_envs * async_factor
num_envs = 1,360 * 2 = 2,720
```

### Step 2.4: Create Vectorized Environment
**Location**: `metta/rl/trainer.py:132-140`

```python
vecenv = make_vecenv(
    curriculum,
    system_cfg.vectorization,
    num_envs=2,720,        # Total environments
    batch_size=1,360,      # Environments per forward pass
    num_workers=16,        # Parallel workers
    ...
)
```

Each worker gets:
```python
envs_per_worker = num_envs // num_workers = 2,720 // 16 = 170
```

---

## 📦 Scene 3: Experience Buffer Creation
*Setting up the memory structure for trajectory storage*

### Step 3.1: Calculate Segments
**Location**: `metta/rl/experience.py:53`

Divide the batch into BPTT sequences:
```python
segments = batch_size // bptt_horizon
segments = 524,288 // 64 = 8,192
```

### Step 3.2: Validate Segment Count
**Location**: `metta/rl/experience.py:54-61`

Ensure we have enough segments for all agents:
```python
if total_agents > segments:  # 3 > 8,192 ✓
    raise ValueError(...)  # Won't trigger
```

### Step 3.3: Create Buffer Tensor
**Location**: `metta/rl/experience.py:63-64`

Allocate the experience buffer:
```python
spec = experience_spec.expand(segments, bptt_horizon)
# Creates shape: [8,192, 64, ...features...]
buffer = spec.zero()  # Initialize with zeros
```

### Step 3.4: Initialize Episode Tracking
**Location**: `metta/rl/experience.py:67-69`

Each agent gets assigned to a segment:
```python
ep_indices = torch.arange(total_agents) % segments
# Agent 0 → segment 0, Agent 1 → segment 1, Agent 2 → segment 2
ep_lengths = torch.zeros(total_agents)  # Track episode progress
```

### Step 3.5: Configure Minibatches
**Location**: `metta/rl/experience.py:84-94`

Calculate how training will divide the data:
```python
minibatch_segments = minibatch_size // bptt_horizon
minibatch_segments = 16,384 // 64 = 256

num_minibatches = segments // minibatch_segments  
num_minibatches = 8,192 // 256 = 32
```

---

## 🎮 Scene 4: Rollout Collection
*Gathering experience from parallel environments*

### Step 4.1: Reset for New Rollout
**Location**: `metta/rl/trainer.py:310`

```python
experience.reset_for_rollout()  # Reset tracking variables
# Sets: full_rows = 0, free_idx = 3, ep_indices = [0,1,2,...]
```

### Step 4.2: Environment Stepping Loop
**Location**: `metta/rl/trainer.py:313-351`

For each environment step until buffer is full:
```python
while not experience.ready_for_training():  # Until full_rows >= 8,192
    # Step 4.2a: Receive observations
    obs, info, env_id, _ = vecenv.recv()  # Line 324
    
    # Step 4.2b: Policy inference
    with torch.no_grad():
        policy(td)  # Line 338
    
    # Step 4.2c: Store experience
    experience.store(data_td, env_id)  # Line 341
    
    # Step 4.2d: Send actions to environment
    vecenv.send(actions, env_id)  # Line 348
```

### Step 4.3: Experience Storage
**Location**: `metta/rl/experience.py:115-130`

Each step gets stored in the buffer:
```python
# Get current position for this agent
episode_length = ep_lengths[agent_id]  # Current step in sequence
segment_idx = ep_indices[agent_id]     # Which segment this agent uses

# Store at [segment_idx, episode_length]
buffer[segment_idx, episode_length] = new_data

# Update tracking
ep_lengths[agent_id] += 1

# Check if sequence complete
if ep_lengths[agent_id] >= bptt_horizon:  # 64 steps
    # Reset this agent's sequence
    ep_indices[agent_id] = free_idx
    free_idx = (free_idx + 1) % segments
    full_rows += 1  # One more complete sequence
```

### Step 4.4: Collection Statistics
**Location**: `metta/rl/trainer.py:352-353`

After collecting full batch:
```python
steps_collected = batch_size = 524,288
sequences_collected = full_rows = 8,192
time_per_env = steps_collected / num_envs = 192 steps
```

---

## 📊 Scene 5: Advantage Calculation
*Computing the "credit assignment" for each action*

### Step 5.1: GAE Computation
**Location**: `metta/rl/trainer.py:372-382`

Calculate advantages using Generalized Advantage Estimation:
```python
advantages = compute_advantages(
    values=experience.buffer["values"],      # Shape: [8,192, 64]
    rewards=experience.buffer["rewards"],    
    dones=experience.buffer["dones"],
    gamma=0.977,
    gae_lambda=0.916,
    ...
)
```

### Step 5.2: GAE Algorithm (CPU/MPS)
**Location**: `metta/rl/mps.py:28-39`

For each sequence, working backwards:
```python
# Step 1: Calculate TD errors
delta[t] = rewards[t+1] + gamma * values[t+1] * (1-done[t+1]) - values[t]

# Step 2: Accumulate advantages backwards
advantages[T-1] = delta[T-1]
for t in range(T-2, -1, -1):
    advantages[t] = delta[t] + gamma * gae_lambda * advantages[t+1] * (1-done[t+1])
```

---

## 🎯 Scene 6: Training Loop
*Where learning happens through gradient descent*

### Step 6.1: Calculate Prioritized Sampling
**Location**: `metta/rl/trainer.py:362-368`

Compute importance sampling correction:
```python
# From metta/utils/batch.py:35-37
total_epochs = total_timesteps // batch_size = 19,073
anneal_beta = prio_beta0 + (1 - prio_beta0) * prio_alpha * epoch / total_epochs
anneal_beta = 0.6 + 0.4 * 0.0 * epoch / 19,073 = 0.6  # With prio_alpha=0
```

### Step 6.2: Iterate Over Epochs
**Location**: `metta/rl/trainer.py:392`

```python
for update_epoch in range(update_epochs):  # Just 1 epoch by default
```

### Step 6.3: Minibatch Sampling Loop
**Location**: `metta/rl/trainer.py:393-419`

Process each of the 32 minibatches:
```python
for _ in range(num_minibatches):  # 32 iterations
```

### Step 6.4: Sample Minibatch
**Location**: `metta/rl/experience.py:160-175`

```python
# Calculate priorities based on advantage magnitude
adv_magnitude = advantages.abs().sum(dim=1)  # Sum over time
prio_weights = adv_magnitude ** prio_alpha   # With alpha=0, all weights=1
prio_probs = prio_weights / prio_weights.sum()

# Sample 256 segments for this minibatch
idx = torch.multinomial(prio_probs, minibatch_segments)  # 256 segments

# Extract minibatch from buffer
minibatch = buffer[idx]  # Shape: [256, 64, ...features...]

# Calculate importance sampling weights
IS_weights = (segments * prio_probs[idx]) ** -anneal_beta
# With uniform sampling: (8,192 * 1/8,192) ** -0.6 = 1.0
```

---

## 🧮 Scene 7: PPO Loss Calculation
*Computing the losses that drive learning*

### Step 7.1: Forward Pass Through Policy
**Location**: `metta/rl/trainer.py:402-404`

```python
policy_td = policy(minibatch, inference_mode=False)
# Returns new values, log_probs, entropy
```

### Step 7.2: Normalize Advantages
**Location**: `metta/rl/losses.py:117`

```python
adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
adv = IS_weights * adv  # Apply importance sampling
```

### Step 7.3: PPO Clipped Policy Loss
**Location**: `metta/rl/losses.py:121-130`

```python
# Calculate probability ratio
ratio = exp(new_log_prob - old_log_prob)

# Clip the ratio
clipped_ratio = clip(ratio, 1 - clip_coef, 1 + clip_coef)
clipped_ratio = clip(ratio, 0.9, 1.1)  # With clip_coef=0.1

# Policy loss
policy_loss = -min(ratio * advantages, clipped_ratio * advantages).mean()
```

### Step 7.4: Value Function Loss
**Location**: `metta/rl/losses.py:133-142`

```python
# Calculate returns
returns = advantages + old_values

# Clipped value loss
v_clipped = old_values + clip(new_values - old_values, -0.1, 0.1)
v_loss = max((new_values - returns)^2, (v_clipped - returns)^2).mean()
```

### Step 7.5: Entropy Loss
**Location**: `metta/rl/losses.py:144-145`

```python
entropy_loss = -entropy.mean()
```

### Step 7.6: Combined Loss
**Location**: `metta/rl/losses.py:147-151`

```python
total_loss = policy_loss + vf_coef * v_loss + ent_coef * entropy_loss
total_loss = policy_loss + 0.44 * v_loss + 0.0021 * entropy_loss
```

---

## 🎯 Scene 8: Gradient Update
*Applying the learning signal*

### Step 8.1: Backward Pass
**Location**: `metta/rl/trainer.py:417-418`

```python
optimizer.zero_grad()
total_loss.backward()  # Compute gradients
```

### Step 8.2: Gradient Clipping
**Location**: `metta/rl/trainer.py:419` (via ppo_step)

```python
torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm=0.5)
```

### Step 8.3: Optimizer Step
**Location**: `metta/rl/trainer.py:419`

```python
optimizer.step()  # Update parameters
```

---

## 📈 Scene 9: Iteration Summary
*What happened in one complete training iteration*

### Final Statistics for One Batch:
```python
# Collection Phase
environments_used: 2,720
steps_per_environment: 192
total_env_steps: 524,288
total_agent_steps: 1,572,864  # 524,288 * 3 agents

# Buffer Organization
sequences_stored: 8,192
sequence_length: 64
total_timesteps: 524,288  # 8,192 * 64

# Training Phase
minibatches_processed: 32
sequences_per_minibatch: 256
gradient_updates: 32  # 32 minibatches * 1 epoch
agent_experiences_per_gradient: 49,152  # 1,572,864 / 32

# Time Progression
training_epochs_completed: epoch += 1
total_steps_so_far: epoch * 524,288
progress: epoch / 19,073 = 0.0052%  # After first iteration
```

---

## 🔄 The Cycle Continues

After completing these 32 gradient updates, the trainer:

1. **Checkpoints** the policy (if `epoch % checkpoint_interval == 0`)
2. **Logs** metrics to wandb
3. **Resets** the experience buffer (`experience.reset_for_rollout()`)
4. **Returns** to Scene 4 for the next rollout

This cycle repeats for 19,073 epochs until `total_timesteps` (10 billion) is reached.

---

## 💡 Key Insights from the Journey

1. **Parallelism is Key**: 2,720 environments run simultaneously across 16 workers
2. **Sequences Matter**: The BPTT horizon of 64 allows learning temporal dependencies
3. **Batching is Hierarchical**: 524K steps → 8,192 sequences → 32 minibatches
4. **Every Step Counts**: Each environment step contributes to 3 agent experiences
5. **Gradient Efficiency**: 32 gradient updates process 1.57M agent experiences

The entire journey from configuration to gradient takes approximately:
- **Rollout**: ~30-60 seconds (depending on environment complexity)
- **Training**: ~5-10 seconds (for 32 gradient updates)
- **Total**: ~35-70 seconds per iteration