from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch

@dataclass
class ActionProb:
    """
    行動ごとの確率を格納するクラス。
    """
    rows_crew_ids: torch.Tensor   # shape: [C]
    cols_task_ids: torch.Tensor   # shape: [T]
    matrix: torch.Tensor          # shape: [C, T], dtype=float, 値域 [0,1]

@dataclass
class Assignment(torch.nn.Module):
    """
    今ラウンドにおける「どのタスクに誰を割り当てたか」をタスク起点で保持するクラス。
    - pairs:                 shape [T, 2], (crew_id, task_id) のペア列。割当不能は (-1, task_id)
    - unassigned_task_ids:   shape [U], 割当不能だったタスクID列（U は失敗数）
    """
    pairs: torch.Tensor
    unassigned_task_ids: torch.Tensor

