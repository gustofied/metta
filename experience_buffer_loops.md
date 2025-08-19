# Understanding the Experience Buffer and Training Loops
## The Nested Structure of Metta's Training Data

---

## The Big Picture: From Buffer to Gradients

The experience buffer contains **8,192 sequences** of length **64**, and we process it through **32 minibatches**. Here's exactly how this nested structure works:

```
┌─────────────────────────────────────────────────────┐
│           EXPERIENCE BUFFER (Full Batch)            │
│                                                     │
│  Total Size: 524,288 timesteps                     │
│  Structure: 8,192 sequences × 64 steps each        │
│                                                     │
│  ┌─────────────────────────────────────────────┐  │
│  │ Segment 0:    [t0, t1, t2, ... t63]        │  │
│  │ Segment 1:    [t0, t1, t2, ... t63]        │  │
│  │ Segment 2:    [t0, t1, t2, ... t63]        │  │
│  │ ...                                         │  │
│  │ Segment 8191: [t0, t1, t2, ... t63]        │  │
│  └─────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
                           ↓
                    Divided into
                           ↓
┌─────────────────────────────────────────────────────┐
│              32 MINIBATCHES                        │
│         Each with 256 sequences                    │
└─────────────────────────────────────────────────────┘
```

---

## The Nested Loop Structure

### Level 1: Update Epochs (Outermost)
```python
for update_epoch in range(update_epochs):  # Default: 1 epoch
    # Process entire buffer
```

### Level 2: Minibatches (32 iterations)
```python
for minibatch_idx in range(32):
    # Sample and process 256 sequences
```

### Level 3: Sequences in Minibatch (Implicit)
```python
# Each minibatch contains 256 sequences
# Processed in parallel by PyTorch
```

### Level 4: Timesteps in Sequence (Implicit)
```python
# Each sequence has 64 timesteps
# Processed by RNN/LSTM unrolling
```

---

## Visual Breakdown: How 32 Minibatches Divide the Buffer

```
FULL BUFFER: 8,192 sequences total
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Minibatch 0:  [Seq 0-255    ] ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░
Minibatch 1:  [Seq 256-511  ] ░░░░████░░░░░░░░░░░░░░░░░░░░░░░░
Minibatch 2:  [Seq 512-767  ] ░░░░░░░░████░░░░░░░░░░░░░░░░░░░░
Minibatch 3:  [Seq 768-1023 ] ░░░░░░░░░░░░████░░░░░░░░░░░░░░░░
...
Minibatch 30: [Seq 7680-7935] ░░░░░░░░░░░░░░░░░░░░░░░░░░░░████
Minibatch 31: [Seq 7936-8191] ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████

Each █ block = 256 sequences = 16,384 timesteps
```

---

## Detailed Structure of One Minibatch

```
┌──────────────────────────────────────────────────────┐
│              MINIBATCH (1 of 32)                    │
│                                                      │
│  Size: 256 sequences × 64 timesteps = 16,384 steps  │
│                                                      │
│  Shape: [256, 64, ...features...]                   │
│                                                      │
│  ┌────────────────────────────────────────────┐     │
│  │ Seq 0:   [obs, action, reward, value] × 64 │     │
│  │ Seq 1:   [obs, action, reward, value] × 64 │     │
│  │ Seq 2:   [obs, action, reward, value] × 64 │     │
│  │ ...                                        │     │
│  │ Seq 255: [obs, action, reward, value] × 64 │     │
│  └────────────────────────────────────────────┘     │
│                                                      │
│  Processing:                                         │
│  • Forward pass through policy network              │
│  • Calculate PPO losses                             │
│  • Backward pass for gradients                      │
│  • Update model weights                             │
└──────────────────────────────────────────────────────┘
```

---

## The Mathematics of 32 Minibatches

### Why exactly 32?

```
Total Sequences ÷ Sequences per Minibatch = Number of Minibatches
    8,192      ÷         256              =        32

This ensures:
• Every sequence is used exactly once per epoch
• No overlap between minibatches
• Even distribution of data
```

### Data Flow per Minibatch

```python
# For each of the 32 minibatches:
minibatch_data = {
    'sequences': 256,
    'timesteps_per_seq': 64,
    'total_timesteps': 16_384,
    'agent_experiences': 16_384 * 3,  # With 3 agents
    'size_in_mb': ~50  # Approximate, depends on features
}
```

---

## Training Loop Implementation

### Actual Code Flow (Simplified)

```python
# Scene 3 ends with buffer ready
assert experience.ready_for_training()  # full_rows >= 8,192

# Training begins
for update_epoch in range(1):  # Usually just 1 epoch
    
    # These are our 32 minibatches
    for minibatch_num in range(32):
        
        # Sample 256 sequences for this minibatch
        indices = sample_indices(256)  # Random or prioritized
        minibatch = experience.buffer[indices]  # Shape: [256, 64, ...]
        
        # Forward pass (all 256 sequences in parallel)
        predictions = policy(minibatch)
        
        # Calculate losses (vectorized across all sequences)
        policy_loss, value_loss, entropy = calculate_ppo_losses(
            predictions, 
            minibatch,
            advantages[indices]
        )
        
        # Single gradient update
        total_loss = policy_loss + 0.44*value_loss + 0.0021*entropy
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        print(f"Minibatch {minibatch_num}/32 complete")
```

---

## Concrete Example with Real Numbers

Let's trace minibatch #0 through the system:

### Step 1: Sampling
```python
# With uniform sampling (prio_alpha = 0)
indices = [0, 1, 2, ..., 255]  # First 256 sequences
```

### Step 2: Data Extraction
```python
minibatch = experience.buffer[0:256]
# Shape: [256 sequences, 64 timesteps, N features]
# Memory: ~50-100 MB depending on observation size
```

### Step 3: Processing Through RNN
```python
# The RNN processes each sequence
for seq_idx in range(256):  # Parallel in practice
    hidden_state = initial_hidden
    for timestep in range(64):
        output, hidden_state = rnn_cell(
            minibatch[seq_idx, timestep], 
            hidden_state
        )
```

### Step 4: Loss Calculation
```python
# Aggregate losses across all 256 × 64 = 16,384 timesteps
total_timesteps_in_minibatch = 16_384
agent_experiences = 16_384 * 3 = 49_152
```

---

## Why This Structure Matters

### 1. **Memory Efficiency**
```
Full Batch:     524,288 timesteps → ~2 GB memory
Per Minibatch:  16,384 timesteps  → ~64 MB memory

32× smaller memory footprint for gradient calculation!
```

### 2. **Gradient Quality**
```
Each gradient uses: 256 sequences × 64 steps × 3 agents
                  = 49,152 agent experiences

Large enough for stable gradients, small enough to fit on GPU
```

### 3. **Temporal Coherence**
```
Each sequence maintains 64 consecutive timesteps
→ RNN can learn temporal dependencies
→ Agent strategies across time are preserved
```

### 4. **Computational Balance**
```
32 gradient updates provides:
• Enough updates for learning progress
• Not too many to overfit
• Good wall-clock time efficiency
```

---

## Impact of Changing num_minibatches

What if we used different numbers?

### 16 Minibatches (Larger batches)
```
Sequences per minibatch: 8,192 ÷ 16 = 512
Timesteps per minibatch: 512 × 64 = 32,768
Pros: More stable gradients
Cons: 2× memory usage, fewer updates
```

### 64 Minibatches (Smaller batches)
```
Sequences per minibatch: 8,192 ÷ 64 = 128
Timesteps per minibatch: 128 × 64 = 8,192
Pros: More gradient updates, less memory
Cons: Noisier gradients, more overhead
```

### Current Choice (32 minibatches)
```
Sequences per minibatch: 256
Timesteps per minibatch: 16,384
Sweet spot for GPU memory and gradient stability
```

---

## The Complete Picture

```
┌──────────────────────────────────────────────────┐
│            ROLLOUT COLLECTION                    │
│                                                  │
│  Total: 2,720 envs (async_factor=2)             │
│  • Group A: envs 0-1,359 (1,360 envs)          │
│  • Group B: envs 1,360-2,719 (1,360 envs)      │
│                                                  │
│  While Group A computes → Group B steps         │
│  While Group B computes → Group A steps         │
│                                                  │
│  Each env contributes: 192 steps                │
│  Total collected: 524,288 timesteps             │
└────────────────────┬─────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────┐
│            EXPERIENCE BUFFER                     │
│                                                  │
│  Structure: 8,192 sequences × 64 steps          │
│  Total: 524,288 timesteps                       │
│  Shape: [8192, 64, ...features...]              │
└────────────────────┬─────────────────────────────┘
                     ↓
┌──────────────────────────────────────────────────┐
│         TRAINING (32 minibatches)               │
│                                                  │
│  for epoch in range(1):                         │
│    for batch_idx in range(32):  ←─── KEY LOOP   │
│      • Sample 256 sequences                     │
│      • Forward pass (all 256 parallel)          │
│      • Calculate PPO losses                     │
│      • Backward pass                            │
│      • Update weights                           │
│                                                  │
│  Result: 32 gradient updates                    │
│  Experiences per gradient: 49,152               │
└──────────────────────────────────────────────────┘
```

---

## Summary: What num_minibatches=32 Means

1. **The experience buffer is divided into 32 equal parts**
2. **Each part contains 256 sequences (16,384 timesteps)**
3. **We iterate through all 32 parts, updating weights each time**
4. **Total: 32 gradient updates per training iteration**
5. **Every experience is used exactly once**

This is the heartbeat of training: 
- Collect 524K steps → 
- Organize into 8,192 sequences → 
- **Process through 32 minibatches** → 
- Apply 32 gradient updates → 
- Repeat ~19,000 times