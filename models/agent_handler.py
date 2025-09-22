# env/agent_handler.py

from __future__ import annotations
import torch
from rl_env.state import ActionMask
from utils.io import ActionProb, Assignment

class BaseAgentHandler:
    """
    AgentHandler のベースクラス。
    ActionMask と ActionProb を受け取り、
    各タスクに必ず一人のクルーを割り当てる。
    """
    def __init__(self, fallback_when_all_masked: bool = True):
        self.fallback_when_all_masked = fallback_when_all_masked

    def _check_and_prepare(self, mask: ActionMask, prob: ActionProb):
        if not torch.equal(mask.rows_crew_ids, prob.rows_crew_ids):
            raise ValueError("rows_crew_ids が一致しません。")
        if not torch.equal(mask.cols_task_ids, prob.cols_task_ids):
            raise ValueError("cols_task_ids が一致しません。")
        if mask.matrix.shape != prob.matrix.shape:
            raise ValueError(f"matrix 形状が一致しません: mask={mask.matrix.shape}, prob={prob.matrix.shape}")

        return (
            mask.rows_crew_ids.long(),
            mask.cols_task_ids.long(),
            mask.matrix.bool(),
            prob.matrix.float(),
        )

    def assign(self, mask: ActionMask, prob: ActionProb) -> Assignment:
        crew_ids, task_ids, feasible, score = self._check_and_prepare(mask, prob)
        chosen_rows = self.select(feasible, score)  # shape [T], 各タスクに割り当てるクルーの「行」index
        return Assignment.from_selection(crew_ids=crew_ids, task_ids=task_ids, chosen_rows=chosen_rows)


    def select(self, feasible: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class GreedyAgentHandler(BaseAgentHandler):
    """確率が最大のクルーを選ぶ（全不可は -1 を返す）"""
    def select(self, feasible: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        neg_inf = torch.tensor(float("-inf"), dtype=score.dtype)
        masked_score = torch.where(feasible, score, neg_inf)
        best_rows = torch.argmax(masked_score, dim=0)

        no_feasible = ~feasible.any(dim=0)  # [T]
        best_rows = torch.where(
            no_feasible,
            torch.full_like(best_rows, -1),
            best_rows
        )
        return best_rows


class StochasticAgentHandler(BaseAgentHandler):
    """確率に従ってサンプリング（全不可は -1 を返す）"""
    def select(self, feasible: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        T = score.shape[1]
        selected = []
        for t in range(T):
            valid_mask = feasible[:, t]         # [C]
            col_score = score[:, t]             # [C]
            if valid_mask.any():
                masked_score = col_score.clone()
                masked_score[~valid_mask] = -float("inf")
                probs = torch.softmax(masked_score, dim=0)
                choice = torch.multinomial(probs, 1).item()
            else:
                choice = -1
            selected.append(choice)
        return torch.tensor(selected, dtype=torch.long, device=score.device)

