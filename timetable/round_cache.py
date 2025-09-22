import numpy as np
from typing import Tuple

def build_round_cache   (
    dep_time: np.ndarray,
    train_id: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    N = int(dep_time.shape[0])

    # dep_time 昇順（同時刻はタスクID昇順）で処理するための並び替え
    order = np.lexsort((np.arange(N), dep_time))

    # 各タスクに割り当てるラウンドID（0始まり）
    round_id = np.zeros(N, dtype=np.int32)

    # ラウンド付与：同一ラウンド内で同じ train_id が現れたら切り替える
    current_round = 0
    seen_train_ids = set()  # 現在のラウンドで既出の train_id

    for idx_in_sorted in range(N):
        i = int(order[idx_in_sorted])
        tid = int(train_id[i])

        # 同じラウンドに同じ train_id を入れない
        if tid in seen_train_ids:
            current_round += 1
            seen_train_ids.clear()

        round_id[i] = current_round
        seen_train_ids.add(tid)

    # タスクID列（0始まり）
    task_ids = np.arange(N, dtype=np.int64)

    # 総ラウンド数（0..current_round の個数）
    R = int(current_round + 1) if N > 0 else 0

    # 各ラウンドの「最小IDタスク」を保持
    round_first_task_id = np.full(R, -1, dtype=np.int64)
    for r in range(R):
        mask = (round_id == r)
        if np.any(mask):
            round_first_task_id[r] = int(task_ids[mask].min())

    # 各タスクが属する round の 0始まり index
    round_task_to_round = round_id.astype(np.int64)

    # 各 round の代表時間 = 「最小IDタスク」の dep_time
    round_time = np.empty(R, dtype=dep_time.dtype)
    for r in range(R):
        tid = int(round_first_task_id[r])
        round_time[r] = dep_time[tid] if tid >= 0 else 0

    return {
        "round_first_task_id": round_first_task_id,  # [n_rounds], 0始まり
        "round_task_to_round": round_task_to_round,  # [n_tasks],  0始まり
        "round_time":           round_time,          # [n_rounds]
    }
