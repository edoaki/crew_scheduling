# -*- coding: utf-8 -*-
"""
round（意思決定ラベル）と [round][station] の lock 集合を作るユーティリティ。

前提（最小）:
- 各タスク i について以下の同長 numpy 配列を受け取る:
  dep_station[i] : 出発駅（0..S-1 の整数）
  dep_time[i]    : 出発時刻（分, int）
  arr_station[i] : 到着駅（0..S-1 の整数）
  arr_time[i]    : 到着時刻（分, int）

出力:
- round_id                : 1始まりのラウンドID（shape [N], int32）
- round_lock_task_ids     : flatten されたタスクID列（1始まり）
- round_lock_ptr          : shape [R*S + 1] のポインタ配列
  -> (r, s) へのアクセスは idx = (r-1)*S + s
     tasks = round_lock_task_ids[ round_lock_ptr[idx] : round_lock_ptr[idx+1] ]

注:
- ラウンドは dep_time 昇順で 1パスにより付与する。
- 「今ラウンドで到着が発生した駅 s の最早到着時刻 ea[s] が、
   次タスクの dep_station と一致し、かつ ea[s] <= 次タスクの dep_time 」
  となった瞬間にラウンドを切り替える（earliest-arrival gating）。
- round_end は保持しない。
"""

from __future__ import annotations
from typing import Tuple
import numpy as np

def build_round_cache   (
    dep_station: np.ndarray,
    dep_time: np.ndarray,
    arr_station: np.ndarray,
    arr_time: np.ndarray,
    num_stations: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    N = int(dep_time.shape[0])
    S = int(num_stations)

    # dep_time 昇順で処理するための並び替え
    order = np.lexsort((np.arange(N), dep_time))
    inv_order = np.empty(N, dtype=np.int64)
    inv_order[order] = np.arange(N)

    round_id = np.zeros(N, dtype=np.int32)

    # ラウンド付与
    current_round = 1
    # ea[s] = そのラウンドで観測した駅 s の最早到着
    ea = np.full(S, np.iinfo(np.int32).max, dtype=np.int32)
    has_ea = np.zeros(S, dtype=np.bool_)

    prev_dep_time = None
    for idx_in_sorted in range(N):
        i = order[idx_in_sorted]
        s_dep = int(dep_station[i])
        t_dep = int(dep_time[i])
        s_arr = int(arr_station[i])
        t_arr = int(arr_time[i])

        # ゲーティング判定（このタスクの直前で切る）
        if has_ea[s_dep] and ea[s_dep] <= t_dep:
            current_round += 1
            ea.fill(np.iinfo(np.int32).max)
            has_ea.fill(False)

        round_id[i] = current_round

        # 到着の最早時刻を更新
        if t_arr < ea[s_arr]:
            ea[s_arr] = t_arr
            has_ea[s_arr] = True

        prev_dep_time = t_dep

    R = int(current_round)

    # [round][station] の lock（= その (round, station) で出発するタスク）を構築
    # タスクIDは 1始まりで保存
    task_ids_1based = np.arange(1, N + 1, dtype=np.int32)
    # バケツ: (r-1)*S + s で 0..R*S-1
    buckets = [[] for _ in range(R * S)]
    for i in range(N):
        r = int(round_id[i]) - 1
        s = int(dep_station[i])
        buckets[r * S + s].append(task_ids_1based[i])

    ptr = [0]
    flat = []
    for b in buckets:
        flat.extend(b)
        ptr.append(len(flat))

    round_lock_task_ids = np.asarray(flat, dtype=np.int32)
    round_lock_ptr = np.asarray(ptr, dtype=np.int32)

    return round_id.astype(np.int32), round_lock_task_ids, round_lock_ptr
