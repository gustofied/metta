# Metta Training Pipeline
## From Configuration to Gradients

---

# Slide 1: Overview
## The Journey of One Training Iteration

• **Input**: Configuration files (YAML)
• **Process**: 9 stages of transformation
• **Output**: 32 gradient updates
• **Duration**: ~35-70 seconds
• **Scale**: 1.57M agent experiences processed

---

# Slide 2: Starting Configuration
## Key Parameters

**Trainer Config**
• `num_workers`: 16
• `batch_size`: 524,288 steps
• `forward_pass_minibatch_target_size`: 4,096
• `async_factor`: 2

**Environment Config**
• `num_agents`: 3
• `obs_size`: 11×11
• `max_steps`: 1,000

---

# Slide 3: Environment Setup
## Creating Parallel Worlds

**Calculate Environments Needed**
```
target_batch = 4,096 ÷ 3 agents = 1,365 envs
actual_batch = round_to_workers(1,365) = 1,360 envs
total_envs = 1,360 × 2 (async) = 2,720 envs
```

**Distribution**
• 2,720 total environments
• 170 environments per worker
• 16 parallel workers

---

# Slide 4: Experience Buffer
## Memory Architecture

**Buffer Dimensions**
```
segments = 524,288 ÷ 64 = 8,192 sequences
shape = [8,192 segments, 64 timesteps, ...features]
```

**Minibatch Structure**
```
minibatch_segments = 16,384 ÷ 64 = 256
num_minibatches = 8,192 ÷ 256 = 32
```

---

# Slide 5: Rollout Collection
## Gathering Experience

**Collection Loop**
1. Reset experience buffer
2. While buffer not full:
   • Receive observations from envs
   • Run policy inference
   • Store experience
   • Send actions to envs

**Collection Stats**
• 192 steps per environment
• 524,288 total env steps
• 1,572,864 agent steps (×3 agents)

---

# Slide 6: Advantage Calculation
## GAE (Generalized Advantage Estimation)

**Forward Pass**
```
TD_error[t] = r[t] + γ·V[t+1] - V[t]
```

**Backward Pass**
```
A[t] = TD[t] + γ·λ·A[t+1]
```

**Parameters**
• γ (gamma) = 0.977
• λ (lambda) = 0.916

---

# Slide 7: Training Loop Structure
## 32 Gradient Updates

**Outer Loop**: 1 epoch (configurable)

**Inner Loop**: 32 minibatches
• Sample 256 sequences
• Forward pass through policy
• Calculate losses
• Backward pass
• Update weights

**Per Gradient**
• 49,152 agent experiences
• 256 sequences of length 64

---

# Slide 8: PPO Loss Calculation
## Three Components

**1. Policy Loss**
```
ratio = exp(new_log_prob - old_log_prob)
clipped = clip(ratio, 0.9, 1.1)
loss = -min(ratio·A, clipped·A)
```

**2. Value Loss**
```
v_clipped = old_v + clip(new_v - old_v, -0.1, 0.1)
loss = max((new_v - returns)², (v_clipped - returns)²)
```

**3. Total Loss**
```
total = policy_loss + 0.44·value_loss + 0.0021·entropy
```

---

# Slide 9: Key Metrics
## One Iteration in Numbers

**Collection Phase**
• 2,720 environments
• 524,288 environment steps
• 1,572,864 agent experiences

**Training Phase**
• 32 minibatches
• 32 gradient updates
• 8,192 sequences processed

**Efficiency**
• 49,152 experiences per gradient
• 192 steps per environment
• 170 envs per worker

---

# Slide 10: Scaling Analysis
## Impact of Agent Count

| Agents | Envs | Agent Steps | Per Gradient |
|--------|------|-------------|--------------|
| 3      | 2,720| 1.57M       | 49K          |
| 4      | 2,048| 2.10M       | 66K          |
| 8      | 1,024| 4.19M       | 131K         |
| 24     | 320  | 12.58M      | 393K         |

**Key Insight**: More agents = fewer envs, more data

---

# Slide 11: Memory Usage
## Buffer Requirements

**Observation Buffer**
```
8,192 × 64 × 11 × 11 × 4 bytes = 253 MB
```

**Total Buffer** (all features)
```
~1-2 GB depending on observation types
```

**GPU Memory** (training)
```
Minibatch + Model + Optimizer ≈ 2-4 GB
```

---

# Slide 12: Performance Tips
## Configuration Adjustments

**For Many Agents (24+)**
• ↑ Increase `num_workers`
• ↓ Reduce `batch_size`
• ↑ Increase `minibatch_size`

**For Few Agents (1-2)**
• ↑ Increase `batch_size`
• ↓ Reduce `num_workers`
• ↓ Decrease `forward_pass_target`

**For Speed**
• Enable `compile=true`
• ↑ Increase `async_factor`
• Optimize `minibatch_size` for GPU

---

# Slide 13: File Locations
## Where It Happens

**Core Files**
• `metta/rl/trainer.py` - Main training loop
• `metta/utils/batch.py` - Batch calculations
• `metta/rl/experience.py` - Buffer management
• `metta/rl/losses.py` - PPO losses
• `metta/rl/vecenv.py` - Environment creation

**Config Files**
• `configs/trainer/trainer.yaml`
• `configs/env/mettagrid/*.yaml`

---

# Slide 14: Training Cycle
## The Complete Loop

```
┌─────────────────┐
│ Load Config     │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Setup Envs      │ → 2,720 parallel envs
└────────┬────────┘
         ↓
┌─────────────────┐
│ Collect Rollout │ → 524K steps
└────────┬────────┘
         ↓
┌─────────────────┐
│ Calculate GAE   │ → Advantages
└────────┬────────┘
         ↓
┌─────────────────┐
│ Train (32×)     │ → Gradient updates
└────────┬────────┘
         ↓
┌─────────────────┐
│ Checkpoint      │ → Save progress
└────────┬────────┘
         ↓
    [Repeat]
```

---

# Slide 15: Summary
## Training Pipeline Essentials

**The Magic Numbers**
• 524,288 - Batch size in steps
• 8,192 - Number of sequences
• 2,720 - Parallel environments
• 64 - BPTT sequence length
• 32 - Gradient updates per batch
• 16 - Worker processes

**The Flow**
Config → Envs → Rollout → Buffer → Training → Gradients

**The Result**
1.57M agent experiences → 32 model updates