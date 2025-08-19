# Agent Lifecycle in the Experience Buffer

This document traces the journey of individual agents through the Metta environment and shows how their experiences get stored in the experience buffer.

## Episode Structure

Each agent lives in episodes of up to 1,000 timesteps (from metta.yaml config):
- **Episode length**: 1,000 steps (max_steps from config)
- **Segment size**: 64 timesteps (bptt_horizon)
- **Segments per episode**: 1,000 ÷ 64 = 15.625 segments (not evenly divisible!)

## The Journey of Agent #42

Let's follow Agent #42 from Environment #1337 through its lifecycle:

### Episode 1: Birth to Death/Timeout (Steps 0-999)

```
Timeline:
Step 0    ┌─────────────┐
          │ Agent spawns│
          └─────────────┘
               ↓
Step 63   [Segment 1: Steps 0-63]    → Stored at buffer slot
               ↓
Step 127  [Segment 2: Steps 64-127]  → Next batch collection
               ↓
Step 191  [Segment 3: Steps 128-191] → Next batch collection
               ↓
          ... (12 more full segments) ...
               ↓
Step 959  [Segment 15: Steps 896-959] → 15th batch collection
               ↓
Step 999  [Partial Segment 16: Steps 960-999] → Only 40 steps!
          ┌──────────────────┐
          │Episode ends/reset│
          └──────────────────┘
```

### Episode 2: Respawn and Continue

```
Step 0    ┌──────────────┐
          │Agent respawns│
          └──────────────┘
               ↓
          New episode begins...
```

## How Episodes Fill the Buffer

With 2,720 environments × 3 agents = 8,160 total agents, here's how episodes interleave:

### Buffer Allocation Pattern

The experience buffer has 8,192 segments total. During rollout collection:

```
INITIAL SYNCHRONIZED COLLECTION (Batch 1)
┌────────┬──────────────────────────────────────────────┐
│Segment │ Content                                          │
├────────┼──────────────────────────────────────────────┤
│ 0      │ Env#0, Agent#0, Steps 0-63                     │
│ 1      │ Env#0, Agent#1, Steps 0-63                     │
│ 2      │ Env#0, Agent#2, Steps 0-63                     │
│ 3      │ Env#1, Agent#0, Steps 0-63                     │
│ 4      │ Env#1, Agent#1, Steps 0-63                     │
│ 5      │ Env#1, Agent#2, Steps 0-63                     │
│ ...    │ ...                                             │
│ 8159   │ Env#2719, Agent#2, Steps 0-63                  │
│ 8160-  │ (unused - we have 8192 slots but only          │
│ 8191   │  8160 agents, so 32 slots remain empty)        │
└────────┴──────────────────────────────────────────────┘
```

But due to async_factor=2, we collect from half at a time:

```
HOW ASYNC_FACTOR=2 REALLY WORKS (Fine-Grained Alternation)

EACH recv() call returns ONE timestep per agent!

ITERATION 1: recv() from Group A (envs 0-1,359)
├─ Returns: 4,080 agents × 1 step = 4,080 steps
├─ Stores: Timestep 0 for agents 0-4,079
└─ Meanwhile: Group B on GPU

ITERATION 2: recv() from Group B (envs 1,360-2,719)
├─ Returns: 4,080 agents × 1 step = 4,080 steps
├─ Stores: Timestep 0 for agents 4,080-8,159
└─ Meanwhile: Group A on GPU

... alternates 126 more times ...

ITERATION 127: Group A completes segments 0-4,079
ITERATION 128: Group B completes segments 4,080-8,159

TOTAL: 128 recv() calls to fill buffer!
├─ Each group: 64 recv() calls
├─ Segments filled: 8,160
└─ Empty segments: 32 (indices 8,160-8,191)
```

## Correct Buffer Filling Pattern

With 8,192 segments needed and 8,160 agents:

```
ROLLOUT COLLECTION (Target: 524,288 steps)
├─ Actual collection: 8,160 agents × 64 steps = 522,240 steps
├─ Each agent provides: 1 segment built over 64 recv() calls
├─ Total recv() calls needed: 128 (alternating groups)
└─ Buffer mismatch: 32 segments remain empty!

BUFFER AFTER 128 recv() CALLS:
┌────────┬──────────────────────────────────────────────┐
│Segment │ Content (built gradually over 64 iterations)    │
├────────┼──────────────────────────────────────────────┤
│ 0      │ Env#0, Agent#0, Steps 0-63                     │
│ 1      │ Env#0, Agent#1, Steps 0-63                     │
│ 2      │ Env#0, Agent#2, Steps 0-63                     │
│ ...    │ (Group A: built during odd iterations)         │
│ 4079   │ Env#1359, Agent#2, Steps 0-63                  │
│ 4080   │ Env#1360, Agent#0, Steps 0-63                  │
│ ...    │ (Group B: built during even iterations)        │
│ 8159   │ Env#2719, Agent#2, Steps 0-63                  │
│ 8160-  │ (unused - 32 empty segments!)                   │
│ 8191   │                                                 │
└────────┴──────────────────────────────────────────────┘
```

## Episode Boundaries and Segment Collection

Episodes span multiple segment collections, with each segment built gradually:

### Segment 1 Collection (Iterations 1-128)
- All agents gradually build steps 0-63 of Episode 1
- 128 recv() calls total (64 per group)
- Both groups stay synchronized

### Segment 2 Collection (Iterations 129-256)
- All agents gradually build steps 64-127 of Episode 1
- Another 128 recv() calls
- Segments slowly filling, one timestep at a time

### Segment 3 Collection (Iterations 257-384)
- All agents gradually build steps 128-191 of Episode 1
- Pattern continues...

...continuing until...

### Segment 15 Collection (Iterations 1,793-1,920)
- All agents: Steps 896-959 of Episode 1
- Getting close to episode boundary!

### Segment 16 Collection (Iterations 1,921-2,048)
- All agents: Steps 960-999 of Episode 1 (only 40 steps!)
- Episode ends, agents reset
- Same segment continues with steps 0-23 of Episode 2
- Episodes wrap within the same segment!

## Key Insights

### 1. **Initially Synchronous Episodes**
All agents start synchronized:
- Batch 1: All agents at steps 0-63
- Batch 2: All agents at steps 64-127
- Initially no agent gets ahead of others

### 2. **Episode Completion is Messy**
- 1,000 steps per episode ÷ 64 steps per batch = 15.625 batches
- Episodes don't align with segment boundaries!
- Batch 16 contains BOTH end of Episode 1 (steps 960-999) AND start of Episode 2 (steps 0-23)

### 3. **Initial Buffer Homogeneity**
Early in training, the buffer contains segments from the SAME timestep range:
```
After Batch 5 (initially):
┌─────────────────────────────────────┐
│ ALL 8,160 segments contain:        │
│ Steps 256-319 from Episode 1        │
└─────────────────────────────────────┘
```

### 4. **No Temporal Diversity!**
This is actually a limitation - all experiences in the buffer are from the same episode phase:
- Can't learn from early-episode and late-episode simultaneously
- All agents experiencing similar game phases together

## Actual Mixed Episodes Scenario

In reality, episodes don't stay synchronized due to:
1. Different episode lengths (agents die at different times)
2. Environment resets at different times
3. The `desync_episodes: true` flag in configs

Here's what actually happens:

```
REALISTIC BUFFER SNAPSHOT (After many training steps)
┌────────┬──────────────────────────────────────────────┐
│Segment │ Content                                          │
├────────┼──────────────────────────────────────────────┤
│ 0      │ Env#0, Agent#0, Episode#3, Steps 512-575       │
│ 1      │ Env#0, Agent#1, Episode#2, Steps 896-959       │
│ 2      │ Env#0, Agent#2, Episode#4, Steps 64-127        │
│ 3      │ Env#1, Agent#0, Episode#2, Steps 704-767       │
│ 4      │ Env#1, Agent#1, Episode#3, Steps 320-383       │
│ ...    │ (diverse mix of episodes and timesteps)        │
└────────┴──────────────────────────────────────────────┘

TEMPORAL DIVERSITY (Realistic)
┌──────────────────┬────────────┬──────────────────────┐
│ Episode Phase    │ # Segments │ % of Buffer          │
├──────────────────┼────────────┼──────────────────────┤
│ Early (0-255)    │ ~2,048     │ ~25%                 │
│ Mid (256-511)    │ ~2,048     │ ~25%                 │
│ Late (512-767)   │ ~2,048     │ ~25%                 │
│ Final (768-999)  │ ~2,048     │ ~25%                 │
└──────────────────┴────────────┴──────────────────────┘
```

## Death and Respawn Cycle

When an agent dies before 1,000 steps:

```
Agent #42 Death at Step 427:
┌──────────────────────────────────────────────────────┐
│ During Batch 7 collection (steps 384-447):           │
│   - Steps 384-427: Normal play                       │
│   - Step 427: Death occurs, terminal flag set        │
│   - Steps 428-447: Environment resets agent          │
│   - New episode begins immediately                   │
│                                                       │
│ Important: The SAME segment contains:                │
│   - End of Episode N (steps 384-427)                 │
│   - Start of Episode N+1 (steps 428-447)             │
│                                                       │
│ LSTM state gets reset when terminal flag detected    │
└──────────────────────────────────────────────────────┘
```

## Prioritized Sampling Impact

During training, segments are sampled based on advantage:

```
SAMPLING PROBABILITY DISTRIBUTION
┌──────────────────────────────────────────────────────┐
│ High-Advantage Segments (deaths, rewards): 40%       │
│ Medium-Advantage Segments (combat, exploration): 35% │
│ Low-Advantage Segments (idle, wandering): 25%        │
└──────────────────────────────────────────────────────┘

Result: Agent learns more from critical moments!
```

## Summary: The Complete Picture

```
THE AGENT'S JOURNEY THROUGH THE SYSTEM

1. SPAWN
   ↓
2. LIVE (up to 1,000 steps)
   ├─ Each 64 steps → 1 segment
   ├─ Stored in experience buffer
   └─ Episode = 15.625 segments (messy!)
   ↓
3. DIE/TIMEOUT
   ↓
4. RESPAWN (new episode)
   ├─ Can happen mid-segment
   └─ LSTM state resets on terminal flag
   ↓
5. TRAINING SAMPLES SEGMENTS
   ├─ Prioritized by advantage
   ├─ Grouped into 32 minibatches
   └─ 32 gradient updates per buffer
   ↓
6. POLICY IMPROVES
   ↓
7. REPEAT
```

### The Key Corrections:

1. **Each recv() returns ONE timestep per agent, not 64!**
   - Segments are built gradually over 64 recv() calls
   - Total of 128 recv() calls to fill the buffer (64 per group)

2. **Episodes are 1,000 steps** - This means episodes don't align with segment boundaries (15.625 segments per episode)

3. **Async factor = 2 creates fine-grained alternation**:
   - Total: 2,720 environments (8,160 agents)
   - Odd iterations: Group A (envs 0-1,359, agents 0-4,079)
   - Even iterations: Group B (envs 1,360-2,719, agents 4,080-8,159)
   - Groups alternate every single recv() call

4. **Buffer organization**:
   - Segments 0-4,079: Built during odd iterations (Group A)
   - Segments 4,080-8,159: Built during even iterations (Group B)
   - Segments 8,160-8,191: Remain empty (32 unused slots)

5. **Both groups stay synchronized**:
   - After iteration 2: Both groups have timestep 0
   - After iteration 4: Both groups have timestep 1
   - Groups never differ by more than 1 timestep

6. **Segments can span episode boundaries** - Both because episodes are 15.625 segments long and because agents can die/respawn mid-segment

The async_factor=2 creates a beautiful alternating pattern where CPU and GPU work is maximally overlapped, with environments switching between stepping and computing every single iteration!