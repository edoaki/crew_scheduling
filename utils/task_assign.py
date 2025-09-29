import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Sequence, Optional, List

def draw_timetable_with_crew_assign(
    depart_station: Sequence[int],
    arrive_station: Sequence[int],
    depart_time: Sequence[float],
    arrive_time: Sequence[float],
    station_names: Optional[List[str]] = None,   # e.g., ["A","B","C",...]
    crew_start_station_idx: Optional[Sequence[int]] = None,  # len = C (crew id = index)
    crew_assignable_start_min: Optional[Sequence[float]] = None,  # len = C
    station_time_from_A: Optional[Sequence[float]] = None,   # e.g., [0.,8.,13.,...]
    solution: Optional[Sequence[int]] = None,    # len = N, value in [0..C-1]
    vertical_scale: float = 1.0,
    figsize=(11, 6),
    circle_size: float = 140.0,
    jitter_sec: float = 1.2,   # crew circles horizontal jitter (minutes)
    title: str = "Timetable",
):
    # numpy arrays
    depart_station = np.asarray(depart_station, dtype=int)
    arrive_station = np.asarray(arrive_station, dtype=int)
    depart_time    = np.asarray(depart_time, dtype=float)
    arrive_time    = np.asarray(arrive_time, dtype=float)

    N = len(depart_station)
    assert len(arrive_station)==N and len(depart_time)==N and len(arrive_time)==N

    # stations
    station_ids = sorted(set(depart_station.tolist() + arrive_station.tolist()))
    S = len(station_ids)

    # station names
    if station_names is None:
        station_names = [str(s) for s in station_ids]
    else:
        assert len(station_names) == S

    # y position of stations
    if station_time_from_A is not None:
        sta_time = np.asarray(station_time_from_A, dtype=float)
        base = np.min(sta_time[station_ids])
        y_pos = {sid: (sta_time[sid] - base) * vertical_scale for sid in station_ids}
    else:
        y_pos = {sid: i*1.0 for i, sid in enumerate(station_ids)}

    # crews
    if crew_start_station_idx is not None:
        crew_start_station_idx = np.asarray(crew_start_station_idx, dtype=int)
        C = len(crew_start_station_idx)
    else:
        C = 0

    if crew_assignable_start_min is not None:
        crew_assignable_start_min = np.asarray(crew_assignable_start_min, dtype=float)
        if C == 0:
            C = len(crew_assignable_start_min)

    # solution -> number of crews implied if not given
    if solution is not None:
        solution = np.asarray(solution, dtype=int)
        assert len(solution) == N
        C = max(C, int(solution.max())+1)

    # colors for crews
    cmap = plt.get_cmap("tab20")
    colors = {cid: cmap(cid % 20) for cid in range(C)}

    # x-range
    t_min = depart_time.min()
    t_max = arrive_time.max()
    if crew_assignable_start_min is not None:
        t_min = min(t_min, crew_assignable_start_min.min())
        t_max = max(t_max, crew_assignable_start_min.max())
    pad = max(5.0, 0.05*(t_max - t_min + 1))
    t_min_plot, t_max_plot = t_min - pad, t_max + pad

    fig, ax = plt.subplots(figsize=figsize)

    # horizontal station lines (gray)
    for i, sid in enumerate(station_ids):
        y = y_pos[sid]
        ax.hlines(y, t_min_plot, t_max_plot, linewidth=1, color="#888888")
        ax.text(t_min_plot, y, f"  {station_names[i]}", va="center", ha="left")

    # tasks (colored by crew id if solution provided; else green)
    for i in range(N):
        ds, as_, dt, at = int(depart_station[i]), int(arrive_station[i]), float(depart_time[i]), float(arrive_time[i])
        y0, y1 = y_pos[ds], y_pos[as_]
        if solution is not None:
            cid = int(solution[i])
            col = colors[cid]
            ax.plot([dt, at], [y0, y1], linewidth=2.5, color=col, zorder=2)
            ax.plot([dt], [y0], marker="o", markersize=4.5, color=col, zorder=3)
            ax.plot([at], [y1], marker="o", markersize=4.5, color=col, zorder=3)
        else:
            ax.plot([dt, at], [y0, y1], linewidth=2.5, color="green", zorder=2)
            ax.plot([dt], [y0], marker="o", markersize=4.5, color="green", zorder=3)
            ax.plot([at], [y1], marker="o", markersize=4.5, color="green", zorder=3)

    # crew initial positions: draw individually with jitter (no aggregation)
    if C > 0 and (crew_start_station_idx is not None) and (crew_assignable_start_min is not None):
        # bucket by (time, station) then assign offsets
        buckets = defaultdict(list)
        for cid, (s, t) in enumerate(zip(crew_start_station_idx, crew_assignable_start_min)):
            buckets[(float(t), int(s))].append(cid)

        for (t, s), crew_list in buckets.items():
            # symmetric jitter around t
            k = len(crew_list)
            offsets = np.linspace(-(k-1)/2.0, (k-1)/2.0, k) * jitter_sec
            for off, cid in zip(offsets, crew_list):
                y = y_pos[s]
                ax.scatter([t+off], [y], s=circle_size, color=colors.get(cid, "black"), zorder=5, edgecolor="white", linewidth=0.8)
                ax.text(t+off, y, str(cid), ha="center", va="center", fontsize=9, color="white", zorder=6)

    # legend (crew color map)
    if C > 0:
        from matplotlib.lines import Line2D
        handles = [Line2D([0],[0], color=colors[cid], lw=4, label=f"C{cid}") for cid in range(C)]
        ax.legend(handles=handles, title="Crew", ncol=min(6, C), loc="upper left", bbox_to_anchor=(1.01, 1.0))

    # axes
    y_vals = list(y_pos.values())
    ax.set_ylim(min(y_vals)-0.5, max(y_vals)+0.5)
    ax.set_xlim(t_min_plot, t_max_plot)
    ax.set_yticks([])

    import matplotlib.ticker as mticker
    def _fmt_min_to_hhmm(x, pos):
        x = int(round(x))
        h, m = divmod(x, 60)
        return f"{h:02d}:{m:02d}"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_min_to_hhmm))

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Station")
    plt.tight_layout()
    plt.savefig("crew_assign.png")
    plt.show()
