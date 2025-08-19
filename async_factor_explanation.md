# How Async Factor = 2 Actually Works in Metta

## The Confusion

The key confusion is: with async_factor=2, do we have:
1. **2x the environments needed** (5,440 total, but only use 2,720 at a time)?
2. **The right number of environments** (2,720 total, alternating which half is active)?

**Answer: Option 2 is correct!**

## The Math

```python
# From metta/utils/batch.py
target_batch_size = forward_pass_minibatch_target_size // num_agents
# target_batch_size = 4,096 // 3 = 1,365

batch_size = (target_batch_size // num_workers) * num_workers  
# batch_size = (1,365 // 16) * 16 = 85 * 16 = 1,360

num_envs = batch_size * async_factor
# num_envs = 1,360 * 2 = 2,720 environments TOTAL
```

So we have exactly 2,720 environments, not 5,440.

## How PufferLib Implements This

From examining `pufferlib/vector.py`:

1. **Total environments**: 2,720 (num_envs)
2. **Batch size**: 1,360 (batch_size parameter to make_vecenv)
3. **Workers**: 16 (num_workers)

The key is in the vecenv initialization:
```python
# In make_vecenv (metta/rl/vecenv.py)
vecenv = pufferlib.vector.make(
    ...
    num_envs=num_envs,        # 2,720 total environments
    batch_size=batch_size,    # 1,360 environments per recv()
    ...
)
```

## What Happens During Collection

```
TOTAL: 2,720 environments × 3 agents = 8,160 agents

PER RECV() CALL:
├─ Returns: 1,360 environments × 3 agents = 4,080 agents
├─ These provide 4,080 × 64 steps = 261,120 steps
└─ This fills 4,080 segments in the experience buffer

TO FILL ENTIRE BUFFER (524,288 steps = 8,192 segments):
├─ Need: 2 recv() calls
├─ First recv(): Environments 0-1359 (4,080 agents)
└─ Second recv(): Environments 1360-2719 (4,080 agents)
```

## The Actual Collection Pattern

```python
# Simplified trainer loop
while not experience.ready_for_training:  # Need 524,288 steps
    # First iteration:
    o, r, d, t, info, env_id, mask = vecenv.recv()
    # Returns data from envs 0-1359 (first half)
    # env_id = slice(0, 4080) for the agents
    
    # Process and send actions back
    vecenv.send(actions)
    
    # Second iteration:
    o, r, d, t, info, env_id, mask = vecenv.recv()  
    # Returns data from envs 1360-2719 (second half)
    # env_id = slice(4080, 8160) for the agents
    
    # Process and send actions back
    vecenv.send(actions)
```

## The Double-Buffering Pattern

What async_factor=2 actually does:

```
TIME →
┌──────────────┬──────────────┬──────────────┬──────────────┐
│   Moment 1   │   Moment 2   │   Moment 3   │   Moment 4   │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Envs 0-1359: │ Envs 0-1359: │ Envs 0-1359: │ Envs 0-1359: │
│   STEPPING   │   ON GPU     │   STEPPING   │   ON GPU     │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Envs 1360-   │ Envs 1360-   │ Envs 1360-   │ Envs 1360-   │
│ 2719:        │ 2719:        │ 2719:        │ 2719:        │
│   ON GPU     │   STEPPING   │   ON GPU     │   STEPPING   │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

## Buffer Filling Timeline

To collect one full experience buffer (524,288 steps):

```
Iteration 1:
├─ recv() returns 4,080 agents × 64 steps = 261,120 steps
├─ Fills segments 0-4,079
└─ From environments 0-1,359

Iteration 2:
├─ recv() returns 4,080 agents × 64 steps = 261,120 steps
├─ Fills segments 4,080-8,159
└─ From environments 1,360-2,719

Total after 2 iterations:
├─ 261,120 + 261,120 = 522,240 steps
├─ Still need 2,048 more steps!
└─ Actually gets 524,288 steps (8,192 segments)
```

Wait, let me recalculate...

Actually, 8,160 agents × 64 steps = 522,240 steps, not 524,288!

## The Discrepancy

There's a mismatch:
- Experience buffer expects: 524,288 steps (8,192 segments)
- Agents provide: 8,160 × 64 = 522,240 steps (8,160 segments)
- Difference: 2,048 steps (32 segments)

This explains why the experience buffer has 8,192 slots but only 8,160 are used!

## Summary: Your Intuition Was Correct!

You're right that with async_factor=2 and batch_size=1,360:
- We get 4,080 segments at a time (from 1,360 envs × 3 agents)
- We need 2 recv() calls to fill the buffer
- First call: segments 0-4,079 (from envs 0-1,359)
- Second call: segments 4,080-8,159 (from envs 1,360-2,719)
- Unused: segments 8,160-8,191 (32 segments remain empty)

The async_factor doesn't mean we have extra environments - it means we have all 2,720 environments but only process half at a time, allowing CPU/GPU overlap for efficiency.