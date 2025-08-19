# Training Nested Loops Visualization

```python
# ASYNC DOUBLE-BUFFERING: 2,720 envs split into two groups of 1,360
# While Group A computes on GPU, Group B steps on CPU, then they swap

# MAIN LOOP: 19,073 epochs (10B total steps / 524K batch size)
for epoch in range(19_073):
    
    # PHASE 1: ROLLOUT COLLECTION (trainer.py:308-353)
    experience.reset_for_rollout()
    
    while experience.full_rows < 8_192:  # Until 8,192 sequences ready
        # Receive from 1,360 envs (alternating groups due to async_factor=2)
        obs, rewards, dones, truncated, infos, env_ids, mask, num_steps = vecenv.recv()
        training_env_id = slice(env_ids[0], env_ids[-1] + 1)  # e.g., slice(0, 1360)
        
        # Policy inference on 1,360 environments
        with torch.no_grad():
            actions, values, log_probs = policy(obs[training_env_id])
        
        # Store in buffer (experience.py:115-130)
        experience.store(
            data_td={'obs': obs, 'actions': actions, 'rewards': rewards, 
                     'values': values, 'log_probs': log_probs, 'dones': dones},
            env_id=training_env_id
        )
        # Each env fills its segment (0-8191) over 64 timesteps
        # When timestep 64 reached, full_rows += 1
        
        # Send actions back, these envs step while other group computes
        vecenv.send(actions, training_env_id)
    
    # Buffer complete: 8,192 sequences × 64 timesteps = 524,288 steps
    
    
    # PHASE 2: ADVANTAGE CALCULATION (trainer.py:372-382)
    with torch.no_grad():
        advantages = compute_gae(
            values=experience.buffer["values"],      # [8_192, 64]
            rewards=experience.buffer["rewards"],    # [8_192, 64]
            dones=experience.buffer["dones"],        # [8_192, 64]
            gamma=0.977, gae_lambda=0.916
        )  # Returns: [8_192, 64]
    
    
    # PHASE 3: PPO TRAINING (trainer.py:392-419)
    for update_epoch in range(1):  # Usually 1 pass through data
        
        # THE KEY LOOP: 32 MINIBATCHES
        for minibatch_idx in range(32):  # 8_192 seqs / 256 per batch = 32
            
            # Sample 256 sequences (experience.py:153-175)
            if prio_alpha == 0:
                sampled_indices = torch.randperm(8_192)[:256]
            else:
                adv_magnitude = advantages.abs().sum(dim=1)
                probs = adv_magnitude / adv_magnitude.sum()
                sampled_indices = torch.multinomial(probs, 256)
            
            minibatch = experience.buffer[sampled_indices]  # [256, 64, features]
            minibatch_advantages = advantages[sampled_indices]  # [256, 64]
            
            # Forward pass (trainer.py:402-404)
            policy.reset_memory()
            policy_output = policy(minibatch, inference_mode=False)
            # Returns: new_values [256,64], new_log_probs [256,64], entropy [256,64]
            
            # PPO losses (losses.py:80-151)
            # 1. Normalize advantages
            norm_adv = (minibatch_advantages - minibatch_advantages.mean()) / (
                minibatch_advantages.std() + 1e-8)
            
            # 2. Policy loss (clipped)
            ratio = torch.exp(policy_output['new_log_probs'] - minibatch['old_log_probs'])
            clipped_ratio = torch.clamp(ratio, 0.9, 1.1)  # clip_coef=0.1
            policy_loss = -torch.min(
                ratio * norm_adv, clipped_ratio * norm_adv
            ).mean()
            
            # 3. Value loss (clipped)
            returns = minibatch_advantages + minibatch['old_values']
            v_clipped = minibatch['old_values'] + torch.clamp(
                policy_output['new_values'] - minibatch['old_values'], -0.1, 0.1)
            value_loss = torch.max(
                (policy_output['new_values'] - returns) ** 2,
                (v_clipped - returns) ** 2
            ).mean()
            
            # 4. Total loss
            entropy_loss = -policy_output['entropy'].mean()
            total_loss = policy_loss + 0.44 * value_loss + 0.0021 * entropy_loss
            
            # Gradient update (trainer.py:417-419)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Per minibatch: 256 seqs × 64 steps × 3 agents = 49,152 experiences
        
        # 32 gradient updates complete
    
    
    # PHASE 4: LOGGING & CHECKPOINTING (trainer.py:420-450)
    if epoch % 50 == 0:
        save_checkpoint(policy, epoch)
    if epoch % 300 == 0:
        upload_to_wandb(policy, epoch)
    
    log_metrics({
        'epoch': epoch,
        'total_steps': epoch * 524_288,
        'agent_steps': epoch * 524_288 * 3,
        'gradient_updates': epoch * 32
    })

# TRAINING COMPLETE: 19,073 epochs × 32 updates = 610,336 total gradient updates


# LOOP STRUCTURE SUMMARY:
# Level 1: 19,073 epochs
#   Level 2: Rollout (collect 8,192 sequences)
#   Level 3: 1 update epoch  
#     Level 4: 32 minibatches ← THE KEY LOOP
#       - Sample 256 sequences
#       - Forward pass + loss calculation
#       - Gradient update
```

## Key Values

| Variable | Value | Calculation |
|----------|-------|-------------|
| epochs | 19,073 | 10B ÷ 524,288 |
| num_envs | 2,720 | 1,360 × 2 (async) |
| batch_size | 524,288 | Config |
| segments | 8,192 | 524,288 ÷ 64 |
| num_minibatches | 32 | 8,192 ÷ 256 |
| minibatch_size | 16,384 | 256 × 64 |

## Per Iteration Flow
```
Rollout:  2,720 envs (1,360 at a time) → 524,288 steps → 8,192 sequences
Training: 8,192 sequences → 32 minibatches → 32 gradient updates
Result:   1,572,864 agent experiences → 32 model improvements
```