from __future__ import annotations
from typing import Tuple
import numpy as np

def build_task_station_cache(
    *,
    dep_station: np.ndarray,   # [N] int
    dep_time: np.ndarray,      # [N] int (minutes)
    arr_station: np.ndarray,   # [N] int
    arr_time: np.ndarray,      # [N] int (minutes)
    board_min: int,
    post_hitch_ready_min: int,
    num_stations: int,
    max_hops: int,
    window_min: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    タスク t を実行できるようにするため、各 station s で「何時までに居ればよいか」を計算。
    - s == dep_station[t] の場合: must_be_by = dep_time[t] - board_min（便乗なし）
    - s != dep_station[t] の場合: 最大 max_hops 本までの「便乗タスク列」を使って dep_station[t] に
      dep_time[t] - board_min - post_hitch_ready_min までに到達できるなら、そのときの「sに居るべき最新時刻」を算出。
      便乗列（task_idの列）は可変長なので、flatten配列＋ポインタで返す。

    返り値:
      cache_task_ptr       : [N+1]        int32   タスクごとの開始位置
      cache_station_ids    : [K]          int32   （task,*) の駅ID
      cache_must_be_by_min : [K]          int32   その駅に「この分までに居れば可」
      cache_is_hitch       : [K]          uint8   0/1
      cache_hops           : [K]          int16   便乗本数
      cache_hitch_minutes  : [K]          int32   便乗合計所要（分）
      cache_path_ptr       : [K+1]        int32   便乗タスク列の開始位置
      cache_path_task_ids  : [P]          int32   便乗タスクID（1始まり）
    """
    N = int(dep_time.shape[0])
    S = int(num_stations)

    ds = np.asarray(dep_station, dtype=np.int32)
    dt = np.asarray(dep_time, dtype=np.int32)
    as_ = np.asarray(arr_station, dtype=np.int32)
    at = np.asarray(arr_time, dtype=np.int32)

    task_ptr = [0]
    st_ids = []
    must_by = []
    is_hitch = []
    hops_arr = []
    hitch_min = []
    path_ptr = [0]
    path_ids = []

    INF_NEG = -10**9

    incoming_by_station = [[] for _ in range(S)]
    for h in range(N):
        incoming_by_station[as_[h]].append(h)

    for t in range(N):
        src = int(ds[t])
        dtime = int(dt[t])
        deadline = dtime - int(board_min) - int(post_hitch_ready_min)

        # 自駅（便乗なし）
        st_ids.append(src)
        must_by.append(dtime - int(board_min))
        is_hitch.append(0)
        hops_arr.append(0)
        hitch_min.append(0)
        path_ptr.append(path_ptr[-1])  # 空列

        if max_hops <= 0 or deadline < 0:
            task_ptr.append(len(st_ids))
            continue

        latest = np.full(S, INF_NEG, dtype=np.int32)
        latest[src] = deadline
        pred_edge = np.full(S, -1, dtype=np.int32)
        pred_next = np.full(S, -1, dtype=np.int32)
        hops_used = np.zeros(S, dtype=np.int16)

        # 逆向き探索（最大 max_hops）
        for _ in range(1, max_hops + 1):
            prev_latest = latest.copy()
            prev_hops = hops_used.copy()
            updated = False
            window_cut = dtime - int(window_min)
            for v in range(S):
                Tv = int(prev_latest[v])
                if Tv <= INF_NEG:
                    continue
                for h in incoming_by_station[v]:
                    if at[h] > Tv:
                        continue
                    if dt[h] < window_cut:
                        continue
                    cand = min(int(dt[h]) - int(board_min), Tv)
                    u = int(ds[h])
                    if cand > latest[u]:
                        latest[u] = cand
                        pred_edge[u] = h
                        pred_next[u] = v
                        hops_used[u] = prev_hops[v] + 1
                        updated = True
            if not updated:
                break

        # 到達可能な駅（src以外）
        reach = np.where(latest > INF_NEG)[0].tolist()
        if src in reach:
            reach.remove(src)

        for s in sorted(reach):
            path = []
            total_min = 0
            k = s
            ok_chain = True
            while k != src:
                e = int(pred_edge[k]); v = int(pred_next[k])
                if e < 0 or v < 0:
                    ok_chain = False; break
                path.append(e)
                total_min += int(at[e]) - int(dt[e])
                k = v
                if len(path) > (max_hops + 2) * 2:
                    ok_chain = False; break
            if not ok_chain:
                continue

            st_ids.append(int(s))
            must_by.append(int(latest[s]))
            is_hitch.append(1 if len(path) > 0 else 0)
            hops_arr.append(len(path))
            hitch_min.append(int(total_min))

            start = path_ptr[-1]
            for e in path:
                path_ids.append(int(e) + 1)  # 1始まり
            path_ptr.append(start + len(path))

        task_ptr.append(len(st_ids))

    cache_task_ptr = np.asarray(task_ptr, dtype=np.int32)
    cache_station_ids = np.asarray(st_ids, dtype=np.int32)
    cache_must_be_by_min = np.asarray(must_by, dtype=np.int32)
    cache_is_hitch = np.asarray(is_hitch, dtype=np.uint8)
    cache_hops = np.asarray(hops_arr, dtype=np.int16)
    cache_hitch_minutes = np.asarray(hitch_min, dtype=np.int32)
    cache_path_ptr = np.asarray(path_ptr, dtype=np.int32)
    cache_path_task_ids = np.asarray(path_ids, dtype=np.int32)

    return (
        cache_task_ptr,
        cache_station_ids,
        cache_must_be_by_min,
        cache_is_hitch,
        cache_hops,
        cache_hitch_minutes,
        cache_path_ptr,
        cache_path_task_ids,
    )
