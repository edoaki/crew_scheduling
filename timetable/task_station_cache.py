from __future__ import annotations

import numpy as np
from typing import Dict

def build_task_station_cache(
    *,
    dep_station: np.ndarray,   # [N] int
    dep_time: np.ndarray,      # [N] int (minutes)
    arr_station: np.ndarray,   # [N] int
    arr_time: np.ndarray,      # [N] int (minutes)
    post_hitch_ready_min: int,
    num_stations: int,
    max_hops: int,
    window_min: int,
) -> Dict[str, np.ndarray]:
    """
    出力:
      - 'station_ids'     : (S,)
      - 'task_ids'        : (T,)
      - 'must_be_by_min'  : (T,S)  到達不可能は -10**9
      - 'is_hitch'        : (T,S)  bool
      - 'hops'            : (T,S)  到達不可能は -1
      - 'hitch_minutes'   : (T,S)  到達不可能は 0
      - 'paths'           : (T,S)  dtype=object, 各要素は np.ndarray[int32]（時間順の便乗 task id 列）
    計算の中身（逆向きDPでの到達可否・締切伝播、同点時は hops が小さい方を優先）は従来方針どおりです。
    """
    N = int(dep_station.shape[0])
    S = int(num_stations)

    ds = dep_station.astype(np.int32, copy=False)
    dt = dep_time.astype(np.int32, copy=False)
    as_ = arr_station.astype(np.int32, copy=False)
    at = arr_time.astype(np.int32, copy=False)

    INF_NEG = -10**9

    # 出力（密）配列
    must_be_by_min = np.full((N, S), INF_NEG, dtype=np.int32)
    is_hitch = np.zeros((N, S), dtype=bool)
    hops = np.full((N, S), -1, dtype=np.int16)
    hitch_minutes = np.zeros((N, S), dtype=np.int32)
    # 各 (t, s) の便乗タスクID配列（object配列; 中身は np.ndarray[int32]）
    paths = np.empty((N, S), dtype=object)
    for t in range(N):
        for s in range(S):
            paths[t, s] = np.empty((0,), dtype=np.int32)

    station_ids = np.arange(S, dtype=np.int32)
    task_ids = np.arange(N, dtype=np.int32)

    # 到着駅ごとの「入ってくる辺」リスト
    incoming_by_station = [[] for _ in range(S)]
    for e in range(N):
        v = int(as_[e])
        if 0 <= v < S:
            incoming_by_station[v].append(e)

    # 各タスク t について、締切を逆伝播
    for t in range(N):
        src = int(ds[t])
        if not (0 <= src < S):
            continue

        dtime = int(dt[t])
        # 便乗なし（出発駅で直接乗る）候補
        mb = dtime 
        must_be_by_min[t, src] = mb
        is_hitch[t, src] = False
        hops[t, src] = 0
        hitch_minutes[t, src] = 0
        paths[t, src] = np.empty((0,), dtype=np.int32)  # 直接乗るので経路は空

        if max_hops <= 0:
            continue

        # 逆向きDP: best_latest[s] = 駅 s に "遅くとも" 何分までにいればよいか
        best_latest = np.full(S, INF_NEG, dtype=np.int32)
        best_hops = np.full(S, 7, dtype=np.int32)         # tie-break 用
        best_hitch_total = np.zeros(S, dtype=np.int32)        # 便乗総分数（参考）
        pred_edge = np.full(S, -1, dtype=np.int32)            # 経路復元用（uで選んだ e）
        pred_next = np.full(S, -1, dtype=np.int32)            # 経路復元用（uの次の駅 v）

        best_latest[src] = mb
        best_hops[src] = 0
        best_hitch_total[src] = 0

        # 最大 max_hops 回だけ逆伝播
        for _hop in range(1, max_hops + 1):
            updated = False

            # それぞれの「到着側」駅 v から、v に入る列車 e をたどって「出発側」駅 u を緩和
            for v in range(S):
                Lv = int(best_latest[v])
                if Lv <= INF_NEG:
                    continue

                # v に到着する列車 e（u->v）を候補に
                for e in incoming_by_station[v]:
                    arr_e = int(at[e])
                    # 到着後の準備時間を確保して v に間に合うか
                    if arr_e + int(post_hitch_ready_min) > Lv:
                        continue
                    # 検索ウィンドウ（不要に古い便を弾く）
                    if arr_e < Lv - int(window_min):
                        continue

                    u = int(ds[e])
                    if not (0 <= u < S):
                        continue
                    dep_e = int(dt[e])

                    # u 側の「遅くとも」= 出発に間に合う + 乗継準備に間に合う の小さい方
                    cand_latest = min(dep_e , Lv - int(post_hitch_ready_min))
                    cand_hops = int(best_hops[v]) + 1
                    if cand_hops > max_hops:
                        continue
                    cand_hitch = best_hitch_total[v] + (arr_e - dep_e)

                    # 改善条件: latest が大きい方優先、同点は hops が小さい方
                    if (cand_latest > best_latest[u]) or (cand_latest == best_latest[u] and cand_hops < best_hops[u]):
                        best_latest[u] = cand_latest
                        best_hops[u] = cand_hops
                        best_hitch_total[u] = cand_hitch
                        pred_edge[u] = e
                        pred_next[u] = v
                        updated = True

            if not updated:
                break

        # 結果を書き戻し（src は既に直接乗車で埋めているので除外）
        for s in range(S):
            if s == src:
                continue
            Ls = int(best_latest[s])
            if Ls <= INF_NEG:
                continue

            must_be_by_min[t, s] = Ls
            is_hitch[t, s] = best_hops[s] > 0
            hops[t, s] = min(int(best_hops[s]), int(np.iinfo(np.int16).max))
            hitch_minutes[t, s] = int(best_hitch_total[s])

            # --- 経路復元（便乗する task の id 配列を作る） ---
            # s から src へ向かう経路を pred_* でたどる
            seq = []
            u = int(s)
            safe = 0
            while u != src:
                e = int(pred_edge[u])
                if e == -1:
                    # 経路が途中で途切れる場合：便乗不可扱い（空配列）
                    seq = []
                    break
                seq.append(e)           # 便乗する task の id（= 辺インデックス）
                u = int(pred_next[u])   # 次の駅へ
                safe += 1
                if safe > S:            # 念のためのループ防止
                    seq = []
                    break

            if seq:
                paths[t, s] = np.asarray(seq, dtype=np.int32)
            else:
                paths[t, s] = np.empty((0,), dtype=np.int32)

    return {
        "station_ids": station_ids,
        "task_ids": task_ids,
        "must_be_by_min": must_be_by_min,
        "is_hitch": is_hitch,
        "hops": hops,
        "hitch_minutes": hitch_minutes,
        "paths": paths,
    }

