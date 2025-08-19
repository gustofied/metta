# Metta Training Loop Architecture

This document explains the rollout and training loops in Metta, detailing the tensor dimensions and batch sizes at each stage.

## Key Configuration Parameters

From `configs/trainer/trainer.yaml`:
```yaml
batch_size: 524,288              # Total experience buffer size
minibatch_size: 16,384           # Training minibatch size
bptt_horizon: 64                 # Backprop through time sequence length
forward_pass_minibatch_target_size: 4,096  # Target env batch size during rollout
async_factor: 2                  # Async environment multiplier
num_workers: <user_defined>      # Number of parallel env workers
```

## Phase 1: Rollout (Experience Collection)
**Location**: `metta/rl/trainer.py:307-350`

### 1.1 Environment Setup
```python
# Calculate batch sizes (trainer.py:123-128)
target_batch_size = forward_pass_minibatch_target_size // num_agents
# Example: 4,096 // 128 agents = 32

batch_size = (target_batch_size // num_workers) * num_workers  
# Example: (32 // 8 workers) * 8 = 32

num_envs = batch_size * async_factor
# Example: 32 * 2 = 64 environments total
```

### 1.2 Rollout Loop
```python
# trainer.py:316-348
while not experience.ready_for_training:  # Until we have 524,288 steps
    # Get observations from environments
    o, r, d, t, info, training_env_id, _, num_steps = get_observation(vecenv, device, timer)
    
    # Tensor shapes at this point:
    # o (observations): [num_envs * num_agents, observation_dim]
    #                   [64 * 128, obs_dim] = [8,192, obs_dim]
    # r (rewards):      [num_envs * num_agents]
    # d (dones):        [num_envs * num_agents]
    # t (truncated):    [num_envs * num_agents]
    
    # Policy forward pass
    with torch.no_grad():
        policy(td)  # td contains the observations and generates actions
    
    # Store in experience buffer
    experience.store(data_td=td, env_id=training_env_id)
    
    # Send actions back to environments
    send_observation(vecenv, td["actions"], dtype_actions, timer)
```

### 1.3 Experience Buffer Structure
```python
# Experience buffer (experience.py:26-39)
segments = batch_size // bptt_horizon
# 524,288 // 64 = 8,192 segments

buffer_shape = [segments, bptt_horizon, ...]
# [8,192, 64, ...additional_dims]

# Ready when:
experience.ready_for_training = (full_rows >= segments)
# Need all 8,192 segments filled
```

## Phase 2: Training (PPO Updates)
**Location**: `metta/rl/trainer.py:360-442`

### 2.1 Calculate Advantages
```python
# trainer.py:372-386
with timer("_train/adv"):
    advantages, vtrace_adjusted_values = calculate_advantages(
        experience.buffer,
        trainer_cfg.ppo.gamma,      # 0.977
        trainer_cfg.ppo.gae_lambda,  # 0.916
        ...
    )
    
# advantages shape: [segments, bptt_horizon]
#                   [8,192, 64]
```

### 2.2 Training Epochs and Minibatches
```python
# trainer.py:392-436
for _update_epoch in range(trainer_cfg.update_epochs):  # Usually 1 epoch
    
    # Calculate number of minibatches
    num_minibatches = batch_size // minibatch_size
    # 524,288 // 16,384 = 32 minibatches
    
    for _ in range(num_minibatches):  # 32 iterations
        # Sample a minibatch
        minibatch, indices, prio_weights = experience.sample_minibatch(
            advantages=advantages,
            ...
        )
        
        # Minibatch tensor shapes:
        # minibatch_segments = minibatch_size // bptt_horizon
        #                    = 16,384 // 64 = 256 segments
        # 
        # Each tensor in minibatch: [256, 64, ...feature_dims]
        # Total timesteps: 256 * 64 = 16,384
        
        # Process minibatch through policy
        loss = process_minibatch_update(
            policy=policy,
            minibatch=minibatch,
            ...
        )
        
        # Gradient update
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping and optimizer step
        if (minibatch_idx + 1) % accumulate_minibatches == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
```

## Data Flow Summary

### Collection Phase
1. **Environment Batch**: `num_envs * num_agents` = 64 * 128 = 8,192 agents
2. **Per Step**: Collect 8,192 agent observations/actions
3. **Total Steps Needed**: 524,288 / 8,192 = 64 rollout iterations
4. **Storage**: Organized as [8,192 segments, 64 timesteps]

### Training Phase  
1. **Full Buffer**: 524,288 agent-timesteps
2. **Minibatches**: 32 minibatches of 16,384 timesteps each
3. **Per Minibatch Shape**: [256 segments, 64 timesteps]
4. **Gradient Updates**: 32 updates per epoch

## Memory Considerations

### During Rollout
- **Active Tensors**: `num_envs * num_agents * feature_dims`
- **Example**: 64 envs * 128 agents * 1024 features * 4 bytes = ~32 MB per tensor
- **Multiple tensors**: observations, actions, values, etc.

### During Training  
- **Experience Buffer**: 524,288 * total_features * 4 bytes
- **Minibatch**: 16,384 * total_features * 4 bytes
- **Gradients**: Same size as model parameters

## Key Relationships

```python
# Must be satisfied:
assert batch_size % minibatch_size == 0  # For even minibatches
assert batch_size % bptt_horizon == 0    # For segmented storage
assert minibatch_size % bptt_horizon == 0  # For BPTT in minibatches

# Segment calculations:
total_segments = batch_size // bptt_horizon  # 8,192
minibatch_segments = minibatch_size // bptt_horizon  # 256
num_minibatches = total_segments // minibatch_segments  # 32

# Environment calculations:
target_batch_size = forward_pass_minibatch_target_size // num_agents
num_envs = ((target_batch_size // num_workers) * num_workers) * async_factor
```

## Example with Concrete Numbers

Given typical configuration:
- 128 agents per environment
- 8 workers
- forward_pass_minibatch_target_size = 4,096
- batch_size = 524,288
- minibatch_size = 16,384
- bptt_horizon = 64

**Rollout**:
- 32 environments per batch (4,096 / 128)
- 64 total environments (32 * 2 async_factor)
- 8,192 agents total (64 * 128)
- 64 collection steps needed (524,288 / 8,192)

**Training**:
- 8,192 segments in buffer (524,288 / 64)
- 32 minibatches (524,288 / 16,384)
- 256 segments per minibatch (16,384 / 64)
- 32 gradient updates per epoch