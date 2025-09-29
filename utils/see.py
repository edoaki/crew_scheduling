import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import Counter
import numpy as np
from typing import Sequence, Optional, List

def draw_timetable_with_crew(
    depart_station: Sequence[int],
    arrive_station: Sequence[int],
    depart_time: Sequence[float],
    arrive_time: Sequence[float],
    station_names: Optional[List[str]] = None,
    crew_start_station_idx: Optional[Sequence[int]] = None,
    crew_assignable_start_min: Optional[Sequence[float]] = None,
    station_time_from_A: Optional[Sequence[float]] = None,  # ここを追加
    station_spacing: float = 1.0,
    circle_base_size: float = 180.0,
    figsize=(10, 6),
    title: Optional[str] = "timetable with crew",
    vertical_scale: float = 1.0,  # 駅間スケール（分→y座標への倍率）
):
    depart_station = np.asarray(depart_station, dtype=int)
    arrive_station = np.asarray(arrive_station, dtype=int)
    depart_time   = np.asarray(depart_time, dtype=float)
    arrive_time   = np.asarray(arrive_time, dtype=float)
    assert len(depart_station) == len(arrive_station) == len(depart_time) == len(arrive_time)

    # 駅ID（0..S-1 想定）
    station_ids = sorted(set(depart_station.tolist() + arrive_station.tolist()))
    S = len(station_ids)

    # 駅名
    if station_names is None:
        station_names = [str(sid) for sid in station_ids]
    else:
        assert len(station_names) == S

    # 駅の縦位置を決定：station_time_from_A があればそれを使用、無ければ等間隔
    if station_time_from_A is not None:
        sta_time = np.asarray(station_time_from_A, dtype=float)
        assert len(sta_time) >= max(station_ids) + 1, "station_time_from_A の長さが station id をカバーしていません。"
        base = np.min(sta_time[station_ids])
        y_pos = {sid: (sta_time[sid] - base) * vertical_scale for sid in station_ids}
    else:
        y_pos = {sid: i * station_spacing for i, sid in enumerate(station_ids)}

    # 描画範囲（時間）
    if crew_assignable_start_min is not None:
        t_min = min(depart_time.min(), np.min(crew_assignable_start_min))
        t_max = max(arrive_time.max(), np.max(crew_assignable_start_min))
    else:
        t_min, t_max = depart_time.min(), arrive_time.max()
    pad = max(5.0, (t_max - t_min) * 0.05)
    t_min_plot, t_max_plot = t_min - pad, t_max + pad

    fig, ax = plt.subplots(figsize=figsize)

    # 駅の水平ライン
    for i, sid in enumerate(station_ids):
        y = y_pos[sid]
        ax.hlines(y, t_min_plot, t_max_plot, linewidth=1, color="#888888")
        ax.plot([t_min_plot], [y], marker='o', markersize=3, color="#555555")
        ax.text(t_min_plot, y, f"  {station_names[i]}", va='center', ha='left')

    # ダイヤ（緑）
    for ds, as_, dt, at in zip(depart_station, arrive_station, depart_time, arrive_time):
        y0, y1 = y_pos[int(ds)], y_pos[int(as_)]
        ax.plot([dt, at], [y0, y1], linewidth=2, color="green")
        ax.plot([dt], [y0], marker='o', markersize=4, color="green")
        ax.plot([at], [y1], marker='o', markersize=4, color="green")

    # クルー初期配置（同時刻・同駅で集約／人数表示）
    if crew_start_station_idx is not None and crew_assignable_start_min is not None:
        crew_start_station_idx = np.asarray(crew_start_station_idx, dtype=int)
        crew_assignable_start_min = np.asarray(crew_assignable_start_min, dtype=float)
        assert len(crew_start_station_idx) == len(crew_assignable_start_min)
        buckets = Counter()
        for s, t in zip(crew_start_station_idx, crew_assignable_start_min):
            buckets[(float(t), int(s))] += 1
        for (t, s), cnt in buckets.items():
            y = y_pos[s]
            size = circle_base_size * (1.0 + 0.4 * (cnt - 1))
            ax.scatter([t], [y], s=size, marker='o', zorder=5)  # 既定色（オレンジ系にしたければここで color 指定）
            if cnt >= 2:
                ax.text(t, y, str(cnt), ha='center', va='center', fontsize=10, zorder=6)

    # 軸設定
    y_vals = list(y_pos.values())
    ax.set_ylim(min(y_vals) - 0.5, max(y_vals) + 0.5)
    ax.set_xlim(t_min_plot, t_max_plot)
    ax.set_yticks([])

    import matplotlib.ticker as mticker
    def _fmt_min_to_hhmm(x, pos):
        x = int(round(x))
        h, m = divmod(x, 60)
        return f"{h:02d}:{m:02d}"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_min_to_hhmm))

    if title:
        ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("station")

    plt.tight_layout()
    plt.savefig("timetable_with_crew.png", dpi=150)
    plt.show()
