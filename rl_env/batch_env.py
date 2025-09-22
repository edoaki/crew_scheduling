# vec_env.py（コピー＆ペースト）
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch

from rl_env.state import StaticObs, DynamicObs,  ActionMask,RoundPairBias
from utils.io import  Assignment, ActionProb

from rl_env.env import CrewAREnv
from rl_env.collate_batch import collate_static_obs ,collate_dynamic_obs,collate_action_mask,collate_pair_bais

class VecCrewAREnv:
    def __init__(
        self,
        generator: Any,
        constraints: Dict[str, Any],
        batch_size: int,
        device: Optional[torch.device],
    ) -> None:
        self.batch_size = int(batch_size)
        self.envs = [CrewAREnv(constraints) for _ in range(self.batch_size)]
        self.generator = generator
        self.device = device   

        self.done = torch.zeros(batch_size, dtype=torch.bool)  # [B] 各envのdone状態
 
    @torch.no_grad()
    def reset(
        self,
        td_list: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        statics  = []
        dyns = []
        masks = []
        pairs = []
        self.done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        for i, env in enumerate(self.envs):
            td = td_list[i]
            static_obs, dyn_obs, mask, pair_info  = env.reset(td)

            statics.append(static_obs)
            dyns.append(dyn_obs)
            masks.append(mask)
            pairs.append(pair_info)
       
        # statics のみパディングして返す（mask/dyn/pair は後で）
        static_batch = collate_static_obs(
            statics=statics,
            device=self.device,     # device 指定があればここでまとめて .to(device)
            return_aux=True         # 付帯出力を付ける（False にすれば外す）
        )
        dyn_batch = collate_dynamic_obs(
            dyns=dyns,
            device=self.device,     # device 指定があればここでまとめて .to(device)
            done=self.done,  # done 情報を渡す
            return_aux=True         # 付帯出力を付ける（False にすれば外す）
        )
        mask_batch = collate_action_mask(
            masks=masks,
            device=self.device,     # device 指定があればここでまとめて .to(device)
            done=self.done,  # done 情報を渡す
        )
        pair_batch = collate_pair_bais(
            pairs=pairs,
            device=self.device,     # device 指定があればここでまとめて .to(device)
            done=self.done,  # done 情報を渡す
        )
        done = self.done
        
        return {"statics": static_batch, "dyns": dyn_batch, "masks": mask_batch, 
                "pairs": pair_batch}
    
    @torch.no_grad()
    def step(
        self,
        selected,
        env_out: torch.Tensor,
    ):
        assignments = selected_to_assignments(selected,env_out)

        dyns: List[DynamicObs] = []
        masks: List[ActionMask] = []
        pairs: List[RoundPairBias] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict[str, Any]] = []

        for i, env in enumerate(self.envs):

            if self.done[i]:
                dyns.append(DynamicObs.init_dummy(T=1,C=1))
                masks.append(ActionMask.init_dummy(T_round=1,C=1))
                pairs.append(RoundPairBias.init_dummy(T_round=1,C=1))
                rewards.append(0.0)
                dones.append(True)
                infos.append({"skipped": True})
                continue
            print()
            print(f"Env {i} processing assignment")
            dyn, mask, pair, reward, done, info = env.step(assignments[i])

            if done:
                self.done[i] = True
                dyns.append(DynamicObs.init_dummy(T=1,C=1))
                masks.append(ActionMask.init_dummy(T_round=1,C=1))
                pairs.append(RoundPairBias.init_dummy(T_round=1,C=1))
                rewards.append(float(reward))
                dones.append(True)
                infos.append(info)
                continue

            dyns.append(dyn)
            masks.append(mask)
            pairs.append(pair)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)

        dyn_batch = collate_dynamic_obs(
            dyns=dyns,
            device=self.device,     # device 指定があればここでまとめて .to(device)
            done=self.done,  # done 情報を渡す
            return_aux=True         # 付帯出力を付ける（False にすれば外す）
        )
        mask_batch = collate_action_mask(
            masks=masks,
            device=self.device,     # device 指定があればここでまとめて .to(device)
            done=self.done,  # done 情報を渡す
        )
        pair_batch = collate_pair_bais(
            pairs=pairs,
            device=self.device,     # device 指定があればここでまとめて .to(device)
            done=self.done,  # done 情報を渡す
        )
        return {"dyns": dyn_batch, "masks": mask_batch, "pairs": pair_batch},torch.tensor(rewards, device=self.device), torch.tensor(dones, device=self.device), infos
    
    def return_solution(self):
        solutions: List[Assignment] = []
        for env in self.envs:
            sol = env.dyn.task_assign
            solutions.append(sol)
        return solutions

    def generate_batch_td(
        self,
        B: int,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        
        td_batch = [self.generator.generate() for _ in range(B)]
        return td_batch

import torch
from typing import List
from utils.io import Assignment

import torch
from typing import List
from utils.io import Assignment

def selected_to_assignments(selected: torch.Tensor,
                            env_out: torch.Tensor,
                            ) -> List[Assignment]:
    """
    selected : [B, T]  各列(=タスク列)に対して選ばれた「行インデックス」。-1 は未割当（またはパディング記号）。
    pad_mask: [B, T]  True=実タスク, False=パディング
    local_crew_ids: [B, R]  logitsの行が指すクルーID列（R=そのラウンドで有効なクルー数）
    round_local_task_ids: [B, T]  logitsの列が指すタスクID列（T=そのラウンドのタスク本数）
    返り値 : 長さ B の Assignment リスト
    """
    pad_mask = env_out["dyns"]["tasks"]["round_task_pad"]
    round_local_task_ids = env_out["dyns"]["tasks"]["round_local_task_ids"]
    local_crew_ids = env_out["dyns"]["crews"]["local_crew_ids"]

    assert selected.shape == pad_mask.shape == round_local_task_ids.shape, \
        "selected/pad_mask/round_local_task_ids は同じ [B, T] 形状である必要があります"
    B, T = selected.shape
    device = selected.device

    assignments: List[Assignment] = []

    for b in range(B):
        mask_b = pad_mask[b]                      # [T] True=実タスク
        sel_b = selected[b]                       # [T] 行インデックス or -1
        task_ids_b = round_local_task_ids[b]      # [T] 各列のタスクID
        crew_rows_b = local_crew_ids[b]           # [R] 行インデックス -> クルーID への写像

        # 実タスクのみ対象（パディング列は除外）
        valid_cols = torch.nonzero(mask_b, as_tuple=False).squeeze(-1)   # [T_real]

        if valid_cols.numel() == 0:
            pairs = torch.empty((0, 2), dtype=torch.long, device=device)
            unassigned_task_ids = torch.empty((0,), dtype=torch.long, device=device)
            assignments.append(Assignment(pairs=pairs, unassigned_task_ids=unassigned_task_ids))
            continue

        sel_real = sel_b[valid_cols]                  # [T_real] 行index or -1
        task_real = task_ids_b[valid_cols]            # [T_real] タスクID

        assigned_mask = sel_real >= 0                 # True=割当あり, False=未割当(-1)

        if assigned_mask.any():
            # 行インデックス -> クルーID へ変換
            crew_idx = sel_real[assigned_mask].to(torch.long)        # [N_pair]
            crew_ids = crew_rows_b.index_select(0, crew_idx)         # [N_pair]
            task_ids = task_real[assigned_mask].to(torch.long)       # [N_pair]
            pairs = torch.stack([crew_ids.to(torch.long), task_ids], dim=1)  # [N_pair, 2]
        else:
            pairs = torch.empty((0, 2), dtype=torch.long, device=device)

        if (~assigned_mask).any():
            unassigned_task_ids = task_real[~assigned_mask].to(torch.long)
        else:
            unassigned_task_ids = torch.empty((0,), dtype=torch.long, device=device)

        assignments.append(Assignment(pairs=pairs, unassigned_task_ids=unassigned_task_ids))

    return assignments

