from __future__ import annotations
from typing import Dict, Any, List, Tuple, Set, Optional
from datetime import datetime, timedelta
import numpy as np

# ---- 定数（他モジュールからも使う） -----------------------------------------

MS10MIN = 10 * 60 * 1000
MS1HOUR = 60 * 60 * 1000

COLORS = {
    "local_line": "#00C853",   # 緑
    "local_turn": "#AEEA00",   # 黄緑
    "rapid_line": "#E53935",   # 赤
    "rapid_turn": "#FF9800",   # オレンジ
}

# ---- ユーティリティ ---------------------------------------------------------

def hhmm(m: int) -> str:
    m = int(m)
    h = (m // 60) % 24
    mm = m % 60
    return f"{h:02d}:{mm:02d}"

def service_key(x: Any) -> str:
    s = str(x).lower()
    return "rapid" if "rapid" in s else "local"

def norm_dir(v) -> Optional[int]:
    """
    方向を +1(up) / -1(down) / 0 に正規化。未知は None。
    """
    if v is None:
        return None
    try:
        n = int(v)
        return 1 if n > 0 else (-1 if n < 0 else 0)
    except Exception:
        s = str(v).strip().lower()
        m = {
            "up": 1, "u": 1, "inbound": 1, "north": 1, "+1": 1, "1": 1,
            "down": -1, "d": -1, "outbound": -1, "south": -1, "-1": -1,
            "0": 0, "both": 0, "none": 0,
        }
        return m.get(s, None)

def group_by_train(data: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    train_id ごとの行インデックスを depart_time 昇順でまとめる。
    """
    tids = [str(t) for t in data["train_id"]]
    groups: Dict[str, List[int]] = {}
    for i, tid in enumerate(tids):
        groups.setdefault(tid, []).append(i)
    dep_t_all = data["depart_time"]
    for tid in groups:
        groups[tid].sort(key=lambda i: int(dep_t_all[i]))
    return groups

# ---- 駅の縦位置：local の所要時間比で内分（Aが上） -----------------------------

def station_positions_by_local_time(station_order: List[str], data: Dict[str, Any]) -> Dict[str, float]:
    """
    連続する駅ペア（A→B, B→C, ...）ごとに、service==local の所要時間の代表値（中央値）を取り、
    その比で [A..J] を内分して縦位置を決める。データが無いペアは全データ→それも無ければ1でフォールバック。
    返り値は {station_id: y(float)}。Aが0、Jが(駅数-1)になるよう全体をスケール。
    """
    pair_local: Dict[Tuple[str, str], List[float]] = {}
    pair_any: Dict[Tuple[str, str], List[float]] = {}

    dep_st, arr_st = data["depart_station"], data["arrive_station"]
    dep_t, arr_t = data["depart_time"], data["arrive_time"]
    service = data.get("service", None)

    for i in range(len(dep_st)):
        u, v = str(dep_st[i]), str(arr_st[i])
        dur = float(int(arr_t[i]) - int(dep_t[i]))
        pair_any.setdefault((u, v), []).append(dur)
        if service is not None and service_key(service[i]) == "local":
            pair_local.setdefault((u, v), []).append(dur)

    seg = []
    for i in range(len(station_order) - 1):
        u, v = station_order[i], station_order[i + 1]
        cand = pair_local.get((u, v)) or pair_any.get((u, v), [])
        val = float(np.median(cand)) if len(cand) else 1.0
        if val <= 0:
            val = 1.0
        seg.append(val)

    total = sum(seg) if seg else 1.0
    scale = (len(station_order) - 1) / total
    ys = {station_order[0]: 0.0}
    acc = 0.0
    for i, v in enumerate(seg):
        acc += v * scale
        ys[station_order[i + 1]] = acc
    return ys

# ---- 折返し検出 -------------------------------------------------------------

def is_turnback_pair(data: Dict[str, Any], i: int, j: int, st2y: Dict[str, float]) -> bool:
    """
    区間 i の到着と 区間 j の次発が同一駅のとき、
    方向反転（折返し）かどうかを判定して True/False を返す。
    """
    arr_st, dep_st = data["arrive_station"], data["depart_station"]
    if str(arr_st[i]) != str(dep_st[j]):
        return False

    direction = data.get("direction", None)
    if direction is not None:
        da = norm_dir(direction[i])
        db = norm_dir(direction[j])
        if (da is not None) and (db is not None) and da != db:
            return True  # up→down または down→up

    # フォールバック：線分の上下向きで判定（Aが上＝yが小さい）
    s_a = np.sign(st2y[_station_key(arr_st[i])] - st2y[_station_key(dep_st[i])])
    s_b = np.sign(st2y[_station_key(arr_st[j])] - st2y[_station_key(dep_st[j])])
    return (s_a != 0 and s_b != 0 and s_a != s_b)

def detect_turnbacks(
    data: Dict[str, Any], idxs: List[int], st2y: Dict[str, float]
) -> List[Tuple[int, int, str, str, str]]:
    """
    折返し候補を検出。
    戻り値: (arrive_time, next_depart_time, station, orientation, service_after)
      orientation: "up_cap"(up→down=上凸) / "down_cap"(down→up=下凸)
    """
    arr_t, dep_t = data["arrive_time"], data["depart_time"]
    arr_st, dep_st = data["arrive_station"], data["depart_station"]
    direction, service = data.get("direction", None), data.get("service", None)
    out = []
    for k in range(len(idxs) - 1):
        a, b = idxs[k], idxs[k + 1]
        if str(arr_st[a]) != str(dep_st[b]):
            continue

        ori: Optional[str] = None
        if direction is not None:
            da, db = norm_dir(direction[a]), norm_dir(direction[b])
            if (da is not None) and (db is not None) and da != db:
                ori = "up_cap" if (da == 1 and db == -1) else "down_cap"

        if ori is None:
            s_a = np.sign(st2y[str(arr_st[a])] - st2y[str(dep_st[a])])  # <0: up, >0: down
            s_b = np.sign(st2y[str(arr_st[b])] - st2y[str(dep_st[b])])
            if s_a != 0 and s_b != 0 and s_a != s_b:
                ori = "up_cap" if (s_a < 0 and s_b > 0) else "down_cap"

        if ori:
            sv = service_key(service[b]) if service is not None else "local"
            out.append((int(arr_t[a]), int(dep_t[b]), str(arr_st[a]), ori, sv))
    return out

# ---- 折返しの半円（パラボラ）座標をバッファに追加 -----------------------------

def add_cap_arc_buffer(xs: List, ys: List, a_min: float, b_min: float, y: float, ori: str, base: datetime):
    """
    折返し領域を半円風（パラボラ）に近似して、(xs, ys) の末尾へ追記する。
    端は y±gap、頂点は y±h。最後に None を入れてひと区切りする。
    """
    h, gap = 0.35, 0.05
    x0, x1 = a_min, b_min
    n = 21
    for j in range(n):
        s = j / (n - 1)  # 0..1
        x_m = x0 + (x1 - x0) * s
        if ori == "up_cap":
            y_m = (y - gap) - 4.0 * (h - gap) * s * (1.0 - s)
        else:
            y_m = (y + gap) + 4.0 * (h - gap) * s * (1.0 - s)
        xs.append(base + timedelta(minutes=x_m))
        ys.append(y_m)
    xs.append(None); ys.append(None)
