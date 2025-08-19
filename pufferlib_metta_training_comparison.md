# Running `puffer train metta` vs Metta's Own Training

## If You Run `puffer train metta` in PufferLib

This shows what happens when you use PufferLib's training system with the Metta environment.

### Configuration Used (from pufferlib/config/metta.ini)

```python
# Environment setup
num_envs = 64           # Total environments
num_workers = 16        # Parallel workers
num_agents = 64         # From metta.yaml line 23

# Training parameters
batch_size = auto       # Will be calculated
minibatch_size = 32768  # Fixed size per gradient
bptt_horizon = 64       # Sequence length
update_epochs = 1       # Default from default.ini
```

### Calculated Values

```python
# Step 1: Calculate batch_size (pufferl.py lines 75-78)
total_agents = num_envs * num_agents = 64 * 64 = 4,096
batch_size = total_agents * bptt_horizon = 4,096 * 64 = 262,144

# Step 2: Calculate segments (pufferl.py line 82)
segments = batch_size // bptt_horizon = 262,144 // 64 = 4,096

# Step 3: Calculate minibatch segments (pufferl.py line 129)
minibatch_segments = minibatch_size // bptt_horizon = 32,768 // 64 = 512

# Step 4: Calculate total minibatches (pufferl.py line 128)
total_minibatches = update_epochs * batch_size / minibatch_size
total_minibatches = 1 * 262,144 / 32,768 = 8
```

### Training Loop Structure

```python
# PUFFERLIB'S TRAINING LOOP (pufferl.py)
for epoch in range(total_epochs):  # 300M steps / 262K = ~1,145 epochs
    
    # ROLLOUT PHASE
    while not experience_full:  # Collect 262,144 steps
        # Process 64 environments at once (no async_factor concept)
        obs = vecenv.recv()  # All 64 envs
        actions = policy(obs)
        vecenv.send(actions)
        # Store in segments (4,096 sequences)
    
    # TRAINING PHASE - Just 8 minibatches!
    for mb in range(8):  # total_minibatches = 8
        # Sample 512 sequences (32,768 steps)
        idx = torch.multinomial(prio_probs, 512)  # minibatch_segments
        
        # Forward pass
        mb_obs = observations[idx]  # [512, 64, ...obs_shape]
        logits, values = policy(mb_obs)
        
        # PPO losses
        # ... (similar to Metta)
        
        # Gradient update
        loss.backward()
        optimizer.step()
    
    # Only 8 gradient updates per epoch!
```

---

## Comparison Table: PufferLib Metta vs Native Metta Training

| Parameter | PufferLib `puffer train metta` | Native Metta Training | Notes |
|-----------|--------------------------------|----------------------|-------|
| **Environment Setup** |
| Total environments | 64 | 2,720 | PufferLib uses way fewer |
| Environments per batch | 64 (all) | 1,360 (half) | No async in PufferLib |
| Workers | 16 | 16 | Same |
| Agents per env | 64 | 3 | Metta.yaml specifies 64! |
| **Batch Sizes** |
| batch_size (steps) | 262,144 | 524,288 | PufferLib is 2× smaller |
| minibatch_size (steps) | 32,768 | 16,384 | PufferLib is 2× larger |
| bptt_horizon | 64 | 64 | Same |
| **Calculated Values** |
| Total agents | 4,096 | 8,160 | 64×64 vs 2,720×3 |
| Segments (sequences) | 4,096 | 8,192 | Half as many |
| Minibatch segments | 512 | 256 | Twice as many per batch |
| Num minibatches | 8 | 32 | PufferLib does 4× fewer! |
| **Training Loop** |
| Update epochs | 1 | 1 | Same |
| Gradient updates/batch | 8 | 32 | Big difference! |
| Agent experiences/gradient | 262K | 49K | PufferLib processes more |
| Total training epochs | ~1,145 | ~19,073 | Different batch sizes |

---

## Key Differences When Running PufferLib Metta

### 1. **Environment Count Mismatch**
- PufferLib config: 64 environments with 64 agents each = 4,096 total agents
- Native Metta: 2,720 environments with 3 agents each = 8,160 total agents
- **The metta.yaml says 64 agents but native Metta uses 3!**

### 2. **No Async Double-Buffering**
- PufferLib doesn't use `async_factor`
- All 64 environments step together
- No overlap of compute and stepping

### 3. **Fewer Gradient Updates**
- PufferLib: 8 gradient updates per batch
- Native Metta: 32 gradient updates per batch
- This could significantly impact learning!

### 4. **Different Batch Composition**
- PufferLib: 262K steps from 64 envs = 4,096 steps per env
- Native Metta: 524K steps from 2,720 envs = 192 steps per env
- PufferLib runs each env 21× longer per batch

### 5. **Larger Minibatches**
- PufferLib: 32,768 steps per gradient (512 sequences)
- Native Metta: 16,384 steps per gradient (256 sequences)
- Larger batches = more stable but less frequent updates

---

## Which is Better?

### PufferLib Metta Advantages:
- Simpler (no async complexity)
- Fewer environments to manage
- Larger minibatches (more stable gradients)
- Less memory overhead

### Native Metta Advantages:
- 4× more gradient updates (better learning)
- Async double-buffering (better hardware utilization)
- More diverse experience (2,720 vs 64 environments)
- Battle-tested for this specific environment

### The Verdict:
**Native Metta is likely better for Metta training** because:
1. More gradient updates (32 vs 8) means faster learning
2. More environments (2,720 vs 64) means more diverse experience
3. The system was specifically tuned for Metta's characteristics

However, **PufferLib Metta might be useful for**:
- Quick experiments with fewer resources
- Testing with different agent counts (64 agents!)
- Simpler debugging without async complexity

---

## Configuration Confusion Alert! 🚨

The biggest issue: **metta.yaml says 64 agents per environment**, but native Metta training uses 3!

This suggests either:
1. The PufferLib config is outdated/wrong
2. There are different Metta variants (64-agent vs 3-agent)
3. Someone changed the agent count without updating configs

This needs investigation - training with 64 agents per env is VERY different from 3!