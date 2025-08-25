# viz/plot.py
import math
from collections import defaultdict
from typing import Dict, List

import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.models.formatters import CustomJSTickFormatter

from bokeh.plotting import figure

COLOR_BY_SERVICE = {"local": "green", "rapid": "red"}

def build_sources(bundle: Dict[str, np.ndarray], ymap: Dict[str, float]):
    dep_st = bundle["depart_station"]
    arr_st = bundle["arrive_station"]
    dep_t = bundle["depart_time"].astype(int)
    arr_t = bundle["arrive_time"].astype(int)
    svc = bundle["service"]
    direc = bundle["direction"]
    tid = bundle["train_ids"]

    n = len(dep_t)
    x0 = dep_t
    y0 = np.array([ymap.get(s, 0.0) for s in dep_st], dtype=float)
    x1 = arr_t
    y1 = np.array([ymap.get(s, 0.0) for s in arr_st], dtype=float)
    color = np.array([COLOR_BY_SERVICE.get(s, "gray") for s in svc], dtype=object)

    seg_src = ColumnDataSource(dict(
        x0=x0, y0=y0, x1=x1, y1=y1,
        train_id=tid, depart_station=dep_st, arrive_station=arr_st,
        depart_time=dep_t, arrive_time=arr_t,
        service=svc, direction=direc, color=color,
        idx=np.arange(n),
    ))

    by_tid = defaultdict(list)
    for i, tr in enumerate(tid):
        by_tid[tr].append(i)

    first_points = []
    last_points = []
    for tr, idxs in by_tid.items():
        idxs = sorted(idxs, key=lambda i: (dep_t[i], arr_t[i]))
        first_points.append(idxs[0])
        last_points.append(idxs[-1])

    start_src = ColumnDataSource(dict(
        x=dep_t[first_points],
        y=y0[first_points],
        train_id=tid[first_points],
        station=dep_st[first_points],
        time=dep_t[first_points],
        service=svc[first_points],
        direction=direc[first_points],
        color=color[first_points],
    ))
    end_src = ColumnDataSource(dict(
        x=arr_t[last_points],
        y=y1[last_points],
        train_id=tid[last_points],
        station=arr_st[last_points],
        time=arr_t[last_points],
        service=svc[last_points],
        direction=direc[last_points],
        color=color[last_points],
    ))

    arc_xs, arc_ys, arc_col, arc_tid = [], [], [], []
    for tr, idxs in by_tid.items():
        idxs = sorted(idxs, key=lambda i: (dep_t[i], arr_t[i]))
        for a, b in zip(idxs, idxs[1:]):
            if arr_st[a] == dep_st[b] and direc[a] != direc[b]:
                t0 = arr_t[a]
                t1 = dep_t[b]
                y = y1[a]
                steps = max(8, int((t1 - t0) / 2) + 1)
                xs = np.linspace(t0, t1, steps)
                amp = 0.3
                ys = y + amp * np.sin(np.linspace(0, math.pi, steps))
                arc_xs.append(xs.tolist())
                arc_ys.append(ys.tolist())
                arc_col.append(COLOR_BY_SERVICE.get(svc[b], "gray"))
                arc_tid.append(tr)
    arc_src = ColumnDataSource(dict(xs=arc_xs, ys=arc_ys, color=arc_col, train_id=np.array(arc_tid, dtype=str)))

    return seg_src, start_src, end_src, arc_src

def make_figure(topo: List[str], ymap: Dict[str, float], seg_src, start_src, end_src, arc_src):
    all_x = list(seg_src.data["x0"]) + list(seg_src.data["x1"])
    x_min, x_max = (min(all_x), max(all_x)) if all_x else (0, 60)

    yvals = [ymap[s] for s in topo if s in ymap]
    y_min, y_max = (min(yvals), max(yvals)) if yvals else (0.0, 1.0)

    p = figure(
        height=700, sizing_mode="stretch_width",
        x_range=(x_min, x_max + 60),
        y_range=(y_min - 1, y_max + 1),
        tools="xpan,xwheel_zoom,reset,save,tap",
        active_drag="xpan", active_scroll="xwheel_zoom",
        title="ダイヤグラム（local=緑, rapid=赤 / 折返し=点線の弧 / ○=新規発車, ▲=収納候補）"
    )

    start_30 = (x_min // 30) * 30
    end_30 = ((x_max + 29) // 30) * 30
    ticks = list(range(start_30, end_30 + 1, 30))
    p.xaxis.ticker = FixedTicker(ticks=ticks)
    p.xaxis.formatter =CustomJSTickFormatter(code="""
        const m = tick;
        const h = Math.floor(m / 60);
        const mi = m % 60;
        return ('0'+h).slice(-2) + ':' + ('0'+mi).slice(-2);
    """)

    y_ticks = [ymap[s] for s in topo if s in ymap]
    y_labels = {ymap[s]: s for s in topo if s in ymap}
    p.yaxis.ticker = FixedTicker(ticks=y_ticks)
    p.yaxis.formatter =CustomJSTickFormatter(code=f"""
        const labels = {y_labels};
        return (tick in labels) ? labels[tick] : tick.toString();
    """)

    seg_glyph = p.segment("x0", "y0", "x1", "y1", color="color", line_width=2, source=seg_src)
    arc_glyph = p.multi_line(xs="xs", ys="ys", line_color="color", line_dash="dashed", line_width=1.5, source=arc_src)

    # circle()/triangle() は非推奨なので scatter を使用
    start_r = p.scatter(x="x", y="y", size=6, marker="circle", color="color", line_color="black", source=start_src)
    end_r = p.scatter(x="x", y="y", size=8, marker="triangle", color="color", line_color="black", source=end_src)

    return p, {"seg": seg_glyph, "arc": arc_glyph, "start": start_r, "end": end_r}
