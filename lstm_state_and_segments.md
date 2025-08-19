# LSTM State Management in Metta Training

This document explains how LSTM hidden states are managed across segment boundaries, how they interact with policy updates, and what happens during the 64-step BPTT segments.

## LSTM Architecture in Metta

From `agent/src/metta/agent/lib/lstm.py`:

```python
class LSTM(LayerBase):
    def __init__(self, **cfg):
        self.hidden_size = cfg["hidden_size"]  # e.g., 256
        self.num_layers = cfg["num_layers"]    # e.g., 1 or 2
        
        # State storage per environment batch
        self.lstm_h: Dict[int, torch.Tensor] = {}  # Hidden states
        self.lstm_c: Dict[int, torch.Tensor] = {}  # Cell states
```

## What Happens at Segment Boundaries (64, 128, etc.)

### The Critical Insight: State Persistence vs. Detachment

At each 64-step boundary, two important things happen:

1. **State is DETACHED from the computation graph**
2. **State VALUES are PRESERVED for the next segment**

```python
# From lstm.py line 128-129
self.lstm_h[training_env_id_start] = h_n.detach()  # Detach but preserve values!
self.lstm_c[training_env_id_start] = c_n.detach()
```

### Timeline of Agent #42's LSTM State

```
Step 0: Episode Start
├─ LSTM state initialized: h_0 = zeros, c_0 = zeros
├─ Forward pass through steps 0-63
└─ Step 63: h_63, c_63 computed and DETACHED

Step 64: Segment Boundary
├─ LSTM receives: h_0 = h_63.detach(), c_0 = c_63.detach()
├─ VALUES preserved but GRADIENT FLOW BROKEN
├─ Forward pass through steps 64-127
└─ Step 127: h_127, c_127 computed and DETACHED

Step 128: Next Segment Boundary
├─ LSTM receives: h_0 = h_127.detach(), c_0 = c_127.detach()
├─ Again, values preserved but no gradient flow from previous segment
└─ Continues...
```

## The Gradient Flow Interruption

### Why Detach at Segment Boundaries?

```
WITHOUT DETACHMENT (Theoretical):
┌────────────────────────────────────────────────────┐
│ Gradient would flow through 1,024 steps!           │
│ Memory requirements: O(episode_length)             │
│ Vanishing/exploding gradients over 1000+ steps     │
└────────────────────────────────────────────────────┘

WITH DETACHMENT (Actual):
┌────────────────────────────────────────────────────┐
│ Gradient only flows through 64 steps (BPTT)        │
│ Memory requirements: O(bptt_horizon) = O(64)       │
│ Stable gradients, manageable memory                │
└────────────────────────────────────────────────────┘
```

### Impact on Learning

The 64-step BPTT means:
- **Short-term dependencies** (< 64 steps): Fully learned via gradients
- **Long-term dependencies** (> 64 steps): Learned indirectly through state values
- **Cross-segment information**: Flows through hidden state VALUES, not gradients

## When Are Policies Updated?

### During Rollout Collection (524,288 steps)

```
ROLLOUT PHASE - NO POLICY UPDATES
├─ Step 0-63: Segment 1 collected
│  └─ LSTM state: h_63 = f(h_0, x_0:63, θ_old)
├─ Step 64-127: Segment 2 collected  
│  └─ LSTM state: h_127 = f(h_63.detach(), x_64:127, θ_old)
├─ Step 128-191: Segment 3 collected
│  └─ LSTM state: h_191 = f(h_127.detach(), x_128:191, θ_old)
└─ ... continues with SAME policy weights θ_old
```

**Key Point**: During rollout, all 8,192 segments use the SAME policy weights!

### During Training Phase (32 minibatches)

```
TRAINING PHASE - POLICY UPDATES HAPPEN
├─ Minibatch 1: Sample 256 segments, compute gradients, UPDATE θ
├─ Minibatch 2: Sample 256 segments, compute gradients, UPDATE θ
├─ Minibatch 3: Sample 256 segments, compute gradients, UPDATE θ
└─ ... 32 total updates

After training: θ_new ≠ θ_old
```

## LSTM State Validity After Policy Updates

### The Critical Question: Does h_63 Still Make Sense After Weights Change?

When we update policy weights from θ_old to θ_new:

```
BEFORE UPDATE:
h_63 = LSTM(x_0:63, h_0, θ_old)  # State computed with old weights

AFTER UPDATE:
h_127 = LSTM(x_64:127, h_63.detach(), θ_new)  # New weights, old state!
         ↑                ↑            ↑
    New computation   Old state   New weights
```

### Why This (Mostly) Works

1. **State as Memory, Not Parameters**
   - Hidden state represents "what the agent remembers"
   - This memory is still valid even if the processing changes slightly
   - Like updating your brain but keeping your memories

2. **Gradual Weight Changes**
   - Learning rate is small (0.018)
   - Weights change incrementally: θ_new ≈ θ_old + ε
   - LSTM can adapt to slightly mismatched states

3. **Detachment Provides Stability**
   - No gradient explosion from mismatched states
   - Each segment starts fresh gradient-wise
   - Old state serves as "initialization" not "truth"

### When It Breaks Down

```
POTENTIAL ISSUES:
1. Large Learning Rate
   └─ θ_new very different from θ_old
   └─ Old states become nonsensical
   
2. Dramatic Policy Shifts
   └─ After many gradient updates
   └─ Hidden states from old policy misleading
   
3. Episode Boundaries Not Respected
   └─ Carrying state across episode resets
   └─ Mixing information from different episodes
```

## Episode Resets and State Management

From `lstm.py` lines 116-121:

```python
# Reset the hidden state if episode is done or truncated
dones = td.get("dones", None)
truncateds = td.get("truncateds", None)
if dones is not None and truncateds is not None:
    reset_mask = (dones.bool() | truncateds.bool()).view(1, -1, 1)
    h_0 = h_0.masked_fill(reset_mask, 0)
    c_0 = c_0.masked_fill(reset_mask, 0)
```

### State Reset Timeline

```
Agent Dies at Step 427:
├─ Step 384-447 (Segment 7)
│  ├─ Step 427: Agent dies, done=True
│  ├─ Step 428-447: Padding/invalid
│  └─ End of segment: State RESET to zeros
│
└─ Step 448-511 (Segment 8)
   ├─ Starts with h_0=0, c_0=0 (fresh state)
   └─ New episode begins cleanly
```

## The Complete LSTM State Lifecycle

```
1. EPISODE START
   ├─ Initialize: h=0, c=0
   └─ Begin with fresh state

2. WITHIN SEGMENT (64 steps)
   ├─ Gradients flow freely
   ├─ BPTT works normally
   └─ State evolves continuously

3. SEGMENT BOUNDARY
   ├─ Detach state from graph
   ├─ Preserve values: h_n, c_n
   └─ Pass to next segment

4. TRAINING PHASE
   ├─ Sample old segments
   ├─ Update policy weights
   └─ θ_old → θ_new

5. NEXT ROLLOUT
   ├─ Use new weights θ_new
   ├─ But segments still connect via state
   └─ Gradual adaptation

6. EPISODE END
   ├─ Reset state to zeros
   └─ Start fresh for next episode
```

## Implications for Learning

### What Gets Learned When

**During Single Segment Training:**
- Associations within 64 steps
- Short-term patterns and reactions
- Immediate cause-effect relationships

**Across Multiple Training Iterations:**
- Long-term strategies emerge
- State learns to carry useful information
- Policy adapts to use persistent state effectively

**The Bootstrap Process:**
1. Early training: States are mostly noise
2. Policy learns to encode useful info in states
3. States become more informative
4. Policy learns to decode and use states better
5. Positive feedback loop develops

## Configuration Impact

### Key Parameters That Affect LSTM State

```python
# From trainer.yaml
bptt_horizon: 64        # Segment size - gradient flow length
batch_size: 524288      # Steps before policy update
minibatch_size: 16384   # Steps per gradient update
update_epochs: 1        # Times through buffer (PPO epochs)
```

### Alternative Configurations and Their Effects

```
SHORTER BPTT (e.g., 16 steps):
✓ More frequent state detachment
✓ Less memory usage
✗ Harder to learn long dependencies
✗ More state "handoffs"

LONGER BPTT (e.g., 256 steps):
✓ Better long-term dependency learning
✓ Fewer state boundaries
✗ Higher memory usage
✗ Potential gradient issues

NO DETACHMENT (Full Episode):
✓ Perfect gradient flow
✗ Massive memory requirements
✗ Gradient explosion/vanishing
✗ Computationally infeasible
```

## Summary: Your Three Key Questions Answered

### 1. What Happens at Steps 64, 128, etc.?

**State is detached but preserved:**
- Gradient flow stops (backprop only through 64 steps)
- State values continue (h_63 becomes h_0 for next segment)
- Information flows forward, gradients don't flow backward

### 2. When Do Policies Get Updated?

**After full rollout collection:**
- Collect all 524,288 steps with θ_old
- Then do 32 minibatch updates
- Next rollout uses θ_new

### 3. Does LSTM State Become Invalid After Updates?

**Mostly no, but with caveats:**
- State represents memory, not parameters
- Small weight changes are tolerated
- System designed for gradual adaptation
- Episode boundaries provide clean resets
- Issues only with dramatic weight changes

The key insight: **Metta's LSTM implementation trades perfect gradient flow for practical trainability**, using state detachment at segment boundaries to maintain manageable memory usage while preserving temporal information through state values.