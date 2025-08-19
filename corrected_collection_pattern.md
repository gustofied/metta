# The ACTUAL Collection Pattern with Async Factor = 2

## You Were Right! Each recv() is ONE Step

Each environment steps exactly **ONE timestep** per recv()/send() cycle, not 64. The 64-step segments are built up over 64 iterations.

## The Real Numbers

```python
# Per recv() call:
Environments returned: 1,360 (half of 2,720)
Agents per recv(): 1,360 × 3 = 4,080
Steps per agent: 1 (ONE timestep!)
Total steps per recv(): 4,080 × 1 = 4,080 steps

# To fill one segment (64 steps) for each agent:
Need 64 recv() calls per group
Group A: 64 calls → 64 timesteps for agents 0-4,079
Group B: 64 calls → 64 timesteps for agents 4,080-8,159
```

## The Actual Collection Loop

```python
# Simplified version of what happens
while not experience.ready_for_training:  # Need 8,192 full segments
    
    # Get ONE timestep from half the environments
    o, r, d, t, info, env_id, _, num_steps = vecenv.recv()
    # env_id alternates: slice(0, 4080) or slice(4080, 8160)
    
    # Store ONE timestep per agent
    experience.store(data_td, env_id)
    # This increments ep_lengths[env_id] by 1
    
    # When ep_lengths reaches 64, segment is complete
    # and ep_lengths resets to 0 for next segment
    
    # Send actions back
    vecenv.send(actions)
```

## The Alternating Pattern

With async_factor=2, the pattern is:

```
ITERATION 1:
├─ recv() from Group A (envs 0-1,359)
├─ Store timestep 0 for agents 0-4,079
├─ send() actions back to Group A
└─ Group B is computing on GPU

ITERATION 2:
├─ recv() from Group B (envs 1,360-2,719)
├─ Store timestep 0 for agents 4,080-8,159
├─ send() actions back to Group B
└─ Group A is computing on GPU

ITERATION 3:
├─ recv() from Group A again
├─ Store timestep 1 for agents 0-4,079
├─ send() actions back
└─ Group B is computing

... continues alternating ...

ITERATION 127:
├─ recv() from Group A
├─ Store timestep 63 for agents 0-4,079
├─ SEGMENTS COMPLETE for Group A!
└─ ep_lengths[0:4080] resets to 0

ITERATION 128:
├─ recv() from Group B
├─ Store timestep 63 for agents 4,080-8,159
├─ SEGMENTS COMPLETE for Group B!
└─ ep_lengths[4080:8160] resets to 0
```

## Total Iterations to Fill Buffer

```
Each agent needs: 64 timesteps (1 segment)
Each recv() provides: 1 timestep for 4,080 agents

Group A agents (0-4,079):
- Need 64 recv() calls to complete their segments
- These happen on odd iterations: 1, 3, 5, ..., 127

Group B agents (4,080-8,159):
- Need 64 recv() calls to complete their segments
- These happen on even iterations: 2, 4, 6, ..., 128

TOTAL: 128 recv() calls to fill all 8,160 segments!
```

## The Timeline

```
┌─────────────────────────────────────────────────────┐
│ Iterations 1-2: Both groups have timestep 0         │
│ Iterations 3-4: Both groups have timestep 1         │
│ Iterations 5-6: Both groups have timestep 2         │
│ ...                                                  │
│ Iterations 125-126: Both groups have timestep 62    │
│ Iterations 127-128: Both groups have timestep 63    │
│                                                      │
│ After 128 iterations:                               │
│ - All 8,160 segments complete                       │
│ - Ready for training!                               │
└─────────────────────────────────────────────────────┘
```

## Why This Matters

1. **Much more alternation**: Instead of 2 big chunks, we have 128 alternating recv() calls
2. **Groups stay synchronized**: Both groups are always within 1 timestep of each other
3. **True double-buffering**: While Group A is stepping, Group B is computing, switching every iteration
4. **Gradual segment building**: Segments fill up one timestep at a time over 64 iterations

## The Experience Buffer During Collection

```
After iteration 1:  [Group A: t=0][Group B: empty]
After iteration 2:  [Group A: t=0][Group B: t=0]
After iteration 3:  [Group A: t=0,1][Group B: t=0]
After iteration 4:  [Group A: t=0,1][Group B: t=0,1]
...
After iteration 127: [Group A: t=0-63 COMPLETE][Group B: t=0-62]
After iteration 128: [Group A: t=0-63 COMPLETE][Group B: t=0-63 COMPLETE]
```

## Episode Boundaries Revisited

Since we're collecting one timestep at a time:

```
Episode length: 1,000 steps
Segment size: 64 steps
Segments per episode: 15.625

This means:
- Segments 1-15: Full 64-step segments
- Segment 16: Partial (40 steps from episode 1, 24 from episode 2)

But collected gradually over many iterations:
- Iterations 1-128: First segment for all agents
- Iterations 129-256: Second segment for all agents
- ...
- Iterations 1,921-2,048: Segment 16 (with episode boundary!)
```

## The Key Insight

The async_factor=2 creates a **fine-grained interleaving** pattern, not a coarse-grained one. Every single iteration alternates between the two groups of environments, maximizing CPU/GPU overlap while keeping both groups synchronized within one timestep of each other.

This is much more elegant than collecting 64 steps at once - it ensures continuous work for both CPU and GPU with minimal idle time!