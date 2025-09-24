import os
import copy
from pathlib import Path
import torch
from torch.nn.utils import clip_grad_norm_
from rl_env.reward import calculate_reward

@torch.no_grad()
def evaluate_mean_reward(policy, vec_env, batch_size: int, device: torch.device,mode):

    td = vec_env.generate_batch_td(B=batch_size)
    env_out = vec_env.reset(td)

    out = policy(env_out=env_out, vec_env=vec_env, phase="val")
    sol = out["solution"]
    reward = calculate_reward(sol, vec_env, device=device)  # [B]
    if mode == "model":
        sampling_out =  vec_env.reset(td)
        sam_out = policy(env_out=sampling_out, vec_env=vec_env, phase="train")
        sam_sol = sam_out["solution"]
        sam_reward = calculate_reward(sam_sol, vec_env, device=device)  # [B]
        sampling_mean_reward = sam_reward.mean()
        
        print("sampling_mean", sampling_mean_reward)
        count_unassign(batch_size,vec_env,sam_sol)

    return reward.mean().item(), reward

def reinforce_step(policy, baseline_policy, vec_env, batch_size: int, device: torch.device, optimizer, grad_clip: float):
    td_batch = vec_env.generate_batch_td(B=batch_size)
    env_out = vec_env.reset(td_batch)

    outdict = policy(env_out=env_out, vec_env=vec_env, phase="train")
    sol = outdict["solution"]
    logprobs = outdict["log_likelihood"]  # [B]
    reward = calculate_reward(sol, vec_env, device=device)  # [B]

    with torch.no_grad():
        base_env_out = vec_env.reset(td_batch)
        base_out = baseline_policy(env_out=base_env_out, vec_env=vec_env, phase="val")
        base_sol = base_out["solution"]
        baseline_reward = calculate_reward(base_sol, vec_env, device=device)  # [B]

    advantage = (reward - baseline_reward).detach()
    adv_mean = advantage.mean()
    adv_std = advantage.std(unbiased=False).clamp_min(1e-8)
    advantage = (advantage - adv_mean) / adv_std
  
    loss = -(advantage * logprobs).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()
    return loss.item()

def checkpoint_path(save_root: Path, run_name: str, epoch: int):
    save_root.mkdir(parents=True, exist_ok=True)
    return save_root / run_name / f"{run_name}.epoch{epoch}.pth"

def save_checkpoint(path: Path, epoch: int, policy, baseline_policy, optimizer, cur_mean_reward: float, base_mean_reward: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "policy_state_dict": policy.state_dict(),
        "baseline_state_dict": baseline_policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "cur_mean_reward": cur_mean_reward,
        "base_mean_reward": base_mean_reward,
    }, str(path))


from rl_env.reward import evaluate_solution
# unassigned_countの分布を見る
from collections import Counter

def count_unassign(batch_size,vec_env,sol):
    batch_totals = []
    
    for i in range(batch_size):
        env = vec_env.envs[i]
        s = env.static
        T = s.num_tasks
        C = s.num_crews
        sol_i = sol[i]

        result = evaluate_solution(s, sol_i)
        batch_totals.append({
            "total_work_time": result["total_work_time"],
            "total_hitch_time": result["total_hitch_time"],
            "unassigned_count": result["unassigned_count"],
        })

    
    uc_list = [bt["unassigned_count"] for bt in batch_totals]
    print("unassigned_count",Counter(uc_list))