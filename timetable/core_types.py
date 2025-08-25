# core_types.py
# 共通の型とユーティリティ（新形式のdirection/segmentsに対応）

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, List, Optional, Any


class Direction(str, Enum):
    up = "up"
    down = "down"


class Service(str, Enum):
    local = "local"
    rapid = "rapid"


@dataclass
class Station:
    id: str
    express_stop: bool
    depot: bool
    crew_rest: bool
    turnback: bool


@dataclass
class Network:
    stations: Dict[str, Station]
    # ローカルは隣接区間の所要（両方向を格納）
    local_segments: Dict[Tuple[str, str], int]    # (u,v) -> 分
    # ラピッドは「停車駅間」の所要（例: A→E, E→J 両方向を格納）
    rapid_segments: Dict[Tuple[str, str], int]    # (u,v) -> 分
    topology: List[str]                            # 例: ["A","B","C","D","E","F","J"]
    up_toward: str                                 # up が向かう終端（例: "A"）
    down_toward: str                               # down が向かう終端（例: "J")


@dataclass
class Token:
    station: str
    direction: Direction
    service: Service
    release_time: int   # 動けるようになる時刻（分）
    deadline: int       # これを超えたら棄却
    target: Optional[int] = None


@dataclass
class Train:
    id: str
    service: Service
    direction: Direction
    current_station: Optional[str]
    next_station: Optional[str]
    next_action_time: int
    alive: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)  # 例: {"depart_mode":"after_dispatch","deadline":...}


@dataclass
class TimetableRow:
    train_id: str
    depart_station: str
    arrive_station: str
    depart_time: int
    arrive_time: int
    service: Service
    direction: Direction


def parse_time_hhmm(s: str) -> int:
    h, m = s.strip().split(":")
    return int(h) * 60 + int(m)


def parse_mmss_like(x) -> int:
    """
    分を返す汎用パーサ:
      - int / float: 分としてそのまま（floatは四捨五入）
      - "NN": 数字文字列は分として解釈（"3" -> 3）
      - "HH:MM": 時刻っぽい表現は分に換算（"00:30" -> 30）
    """
    # 数値（int/float）は分として扱う
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(round(x))

    # 文字列系
    if isinstance(x, str):
        s = x.strip()
        # "HH:MM" 形式 -> 分に変換
        if ":" in s:
            parts = s.split(":")
            if len(parts) >= 2:
                h = int(parts[0]) if parts[0] else 0
                m = int(parts[1]) if parts[1] else 0
                return h * 60 + m
        # 数字だけの文字列は分
        if s.isdigit():
            return int(s)

    raise ValueError(f"Unsupported time format for minutes: {x!r}")


def format_time_hhmm(m: int) -> str:
    h = m // 60
    mi = m % 60
    return f"{h:02d}:{mi:02d}"


def _step_sign(topology: List[str], up_toward: str, d: Direction) -> int:
    if not topology:
        return +1
    first = topology[0]
    last = topology[-1]
    if d == Direction.up:
        if up_toward == last:
            return +1
        if up_toward == first:
            return -1
        return +1
    else:
        if up_toward == last:
            return -1
        if up_toward == first:
            return +1
        return -1


def next_stop_station(network: Network, station_id: str, direction: Direction, service: Service) -> Optional[str]:
    topo = network.topology
    if station_id not in topo:
        return None
    i = topo.index(station_id)
    step = _step_sign(topo, network.up_toward, direction)
    j = i + step
    if j < 0 or j >= len(topo):
        return None
    if service == Service.local:
        return topo[j]
    # rapid は次の express_stop 駅まで飛ぶ
    while 0 <= j < len(topo):
        st = network.stations[topo[j]]
        if st.express_stop:
            return topo[j]
        j += step
    return None


def _sum_local_between(network: Network, u: str, v: str) -> int:
    topo = network.topology
    iu, iv = topo.index(u), topo.index(v)
    if iu == iv:
        return 0
    step = 1 if iv > iu else -1
    t = 0
    i = iu
    while i != iv:
        a, b = topo[i], topo[i + step]
        t_seg = network.local_segments.get((a, b))
        if t_seg is None:
            raise KeyError(f"local path segment not found: {a}->{b}")
        t += t_seg
        i += step
    return t


def travel_minutes(network: Network, u: str, v: str, service: Service) -> int:
    if service == Service.local:
        t = network.local_segments.get((u, v))
        if t is None:
            # 逆向きが登録されていればそれを流用（片方向しか書かれていない設定の救済）
            t = network.local_segments.get((v, u))
        if t is None:
            raise KeyError(f"local segment not found: {u}->{v}")
        return int(t)
    # rapid: 定義があればそれを使い、無ければローカル合算で近似
    t = network.rapid_segments.get((u, v))
    if t is not None:
        return int(t)
    return _sum_local_between(network, u, v)


def can_stop_here(network: Network, station_id: str, service: Service) -> bool:
    if service == Service.local:
        return True
    return bool(network.stations[station_id].express_stop)


def opposite_direction(d: Direction) -> Direction:
    return Direction.down if d == Direction.up else Direction.up
