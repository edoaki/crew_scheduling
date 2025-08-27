# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from typing import Dict, Any
from timetable.io_npz import load_timetable_bundle
from timetable.generator import generate_and_save

def mm_to_hhmm(m: int) -> str:
    h = int(m) // 60
    mm = int(m) % 60
    return f"{h:02d}:{mm:02d}"

def show_overview(bundle: Dict[str, Any], n_tt: int = 5, n_round: int = 3, n_tasks: int = 2):
    tt = bundle["tt"]
    rd = bundle.get("round", {})
    feat = bundle.get("features", {})
    meta = bundle.get("meta", {})
    labels = meta.get("station_label_vocab", [])

    def lab(i: int) -> str:
        if 0 <= i < len(labels):
            return labels[i]
        return str(i)

    print("== TT rows (head) ==")
    N = tt["depart_time"].shape[0]
    head = min(n_tt, N)
    for i in range(head):
        ds = int(tt['depart_station'][i]); as_ = int(tt['arrive_station'][i])
        print(f"[{i}] train={tt['train_ids'][i]} dir={tt['direction'][i]} "
              f"depS={ds}({lab(ds)}) depT={mm_to_hhmm(tt['depart_time'][i])} "
              f"arrS={as_}({lab(as_)}) arrT={mm_to_hhmm(tt['arrive_time'][i])} "
              f"svc={tt['service'][i]}")

    if rd:
        print("\n== Rounds (head) ==")
        ptr = rd["round_ptr"]; flat = rd["round_tt_idx"]; anchor = rd["round_anchor_min"]
        R = ptr.shape[0] - 1
        for r in range(min(n_round, R)):
            s, e = ptr[r], ptr[r+1]
            idxs = flat[s:e]
            print(f"round_id {r+1}: idx={list(map(int, idxs))} anchor={mm_to_hhmm(anchor[r])}")

    if feat:
        print("\n== feat (first couple of tasks) ==")
        tptr = feat["cache_task_ptr"]
        st = feat["cache_station_ids"]
        must = feat["cache_must_be_by_min"]
        hitch = feat["cache_is_hitch"]
        hops = feat["cache_hops"]
        hmin = feat["cache_hitch_minutes"]
        pptr = feat["cache_path_ptr"]
        pids = feat["cache_path_task_ids"]
        T = tptr.shape[0] - 1
        for t in range(min(n_tasks, T)):
            s, e = tptr[t], tptr[t+1]
            print(f"task {t+1}:")
            for k in range(s, e):
                sid = int(st[k])
                ps, pe = pptr[k], pptr[k+1]
                print(f"  station={sid}({lab(sid)}) must_be_by={mm_to_hhmm(int(must[k]))} "
                      f"hitch={int(hitch[k])} hops={int(hops[k])} hitch_min={int(hmin[k])} "
                      f"path={list(map(int, pids[ps:pe]))}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="config/station.yaml")
    ap.add_argument("--train", default="config/train.yaml")
    ap.add_argument("--out", default="data/timetable.npz")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--board_min", type=int, default=2)
    ap.add_argument("--post_hitch_ready_min", type=int, default=5)
    ap.add_argument("--max_hops", type=int, default=2)
    ap.add_argument("--window_min", type=int, default=120)
    ap.add_argument("--n_tt", type=int, default=5)
    ap.add_argument("--n_round", type=int, default=3)
    ap.add_argument("--n_tasks", type=int, default=2)
    args = ap.parse_args()

    ok, msg = generate_and_save(
        args.station, args.train, args.out,
        board_min=args.board_min,
        post_hitch_ready_min=args.post_hitch_ready_min,
        max_hops=args.max_hops,
        window_min=args.window_min,
        seed=args.seed,
    )
    print(("OK " if ok else "NG ") + msg)
    if ok:
        bundle = load_timetable_bundle(args.out)
        show_overview(bundle, args.n_tt, args.n_round, args.n_tasks)
