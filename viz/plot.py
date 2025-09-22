from __future__ import annotations
from typing import Dict, Any, List, Set
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go

# コア関数/定数を分離
from .plot_core import (
    MS10MIN, MS1HOUR, COLORS,
    hhmm, service_key,
    station_positions_by_local_time, group_by_train,
    detect_turnbacks, add_cap_arc_buffer, is_turnback_pair,
)

def _extend_with_cd(x_points, y_points, cd_tuple, xs, ys, cds, add_none=True):
    # x_points, y_points: 実点（Noneを含めない）
    # cd_tuple: クリック時に拾いたい情報（ここでは tt の 0始まり index など）
    xs.extend(x_points)
    ys.extend(y_points)
    cds.extend([cd_tuple] * len(x_points))
    if add_none:
        xs.append(None); ys.append(None); cds.append(None)


def build_diamond_figure(
    td: Dict[str, Any],
    station_order: List[str],
    visible_trains: Set[str],
) -> go.Figure:
    """
    ・横方向のみズーム
    ・1時間ごとのラベル、10分ごとの縦グリッド
    ・区間ごとに local/rapid を色分け（local=緑、rapid=赤）
    ・停車は横線で接続（ただし折返しの場合は引かない）
    ・折返しは半円（パラボラ）で表示（local=黄緑、rapid=オレンジ）
    ・Aが上（autorange=reversed）
    ・出庫=○、収納=△（黒固定）
    ・ホバーは最も近い1件のみ表示
    """
    def _station_key(x):
        try:
            import torch
            if hasattr(x, "item"):
                x = x.item()
        except Exception:
            pass
        import numpy as _np
        if isinstance(x, (int, _np.integer)):
            return station_order[int(x)]
        return str(x)
    base = datetime(1970, 1, 1)

    dep_times = np.asarray(td["depart_time"]).astype(int, copy=False) if len(td["depart_time"]) else np.array([0])
    arr_times = np.asarray(td["arrive_time"]).astype(int, copy=False) if len(td["arrive_time"]) else np.array([60])
    tmin = int(dep_times.min()) if dep_times.size else 0
    tmax = int(arr_times.max()) if arr_times.size else 60
    x_min = base + timedelta(minutes=tmin - 5)
    x_max = base + timedelta(minutes=tmax + 5)

    # 駅の縦位置（local 所要時間比で内分）
    st2y = station_positions_by_local_time(station_order, td)
    yvals = [st2y[s] for s in station_order]
    y_top, y_bottom = -0.5, max(yvals) + 0.5

    # 列車ごとのインデックス
    groups = group_by_train(td)

    fig = go.Figure()

    # 折返しアークのまとめバッファ（service × 方向）
    cap_x = {"local": {"up_cap": [], "down_cap": []}, "rapid": {"up_cap": [], "down_cap": []}}
    cap_y = {"local": {"up_cap": [], "down_cap": []}, "rapid": {"up_cap": [], "down_cap": []}}

    # 出庫/収納マーカー（黒固定）
    start_x = []; start_y = []
    end_x = []; end_y = []

    for tid, idxs in groups.items():
        if tid not in visible_trains:
            continue

        # 区間ごとに service を見て、local/rapid で別トレース（停車もつなぐ）
        xs_local: List = []; ys_local: List = []; cd_local: List = []
        xs_rapid: List = []; ys_rapid: List = []; cd_rapid: List = []
        xs_dw_local: List = []; ys_dw_local: List = []; cd_dw_local: List = []
        xs_dw_rapid: List = []; ys_dw_rapid: List = []; cd_dw_rapid: List = []

        for k, i in enumerate(idxs):
            dt, at = int(td["depart_time"][i]), int(td["arrive_time"][i])
            u = _station_key(td["depart_station"][i])
            v = _station_key(td["arrive_station"][i])
            sv = service_key(td["service"][i])
            yu, yv = st2y[u], st2y[v]
            mid = dt + (at - dt) / 2.0
            ym = yu + (yv - yu) / 2.0

            # 例: タスク i の (u -> v) 区間（出発 dt 分 -> 到着 at 分）、中点 mid 分
            x_seg = [base + timedelta(minutes=dt),
                    base + timedelta(minutes=mid),
                    base + timedelta(minutes=at)]
            y_seg = [yu, ym, yv]

            cd_seg = [
                int(i),                                  # tt の 0始まり index
                str(tid),                                # train_id は文字列で統一
                str(u), str(v),                          # 駅IDは文字列化しておくと安全
                hhmm(int(dt)), hhmm(int(at)),            # 時刻は "HH:MM" 文字列
                service_key(td["service"][i]),         # "local" or "rapid"
                str(td["direction"][i]) if "direction" in td else "None",
            ]

            if service_key(td["service"][i])   == "local":
                _extend_with_cd(x_seg, y_seg, cd_seg, xs_local, ys_local, cd_local)
            else:
                _extend_with_cd(x_seg, y_seg, cd_seg, xs_rapid, ys_rapid, cd_rapid)


            # 停車横線（次区間が折返しでない場合だけ描く）
            if k < len(idxs) - 1:
                j = idxs[k + 1]
                if _station_key(td["arrive_station"][i]) == _station_key(td["depart_station"][j]) and not is_turnback_pair(td, i, j, st2y):
                    t0, t1 = int(td["arrive_time"][i]), int(td["depart_time"][j])
                    st = _station_key(td["arrive_station"][i])
                    y = st2y[st]

                    sv_next = service_key(td["service"][j]) if "service" in td else "local"
                    x_dw = [base + timedelta(minutes=t0),base + timedelta(minutes=t1)]
                    y_dw = [y, y]

                    cd_dw = (
                        j,     # 次区間の tt index を持たせるのが運用上便利
                        tid, st, st,
                        hhmm(t0), hhmm(t1),
                        service_key(td["service"][j])  ,
                        str(td.get("direction", ["None"])[j]) if "direction" in td else "None",
                    )

                    if service_key(td["service"][j]) == "local":
                        _extend_with_cd(x_dw, y_dw, cd_dw, xs_dw_local, ys_dw_local, cd_dw_local)
                    else:
                        _extend_with_cd(x_dw, y_dw, cd_dw, xs_dw_rapid, ys_dw_rapid, cd_dw_rapid)


        # 出庫（○）・収納（△）マーカー位置（黒固定）
        i0, i1 = idxs[0], idxs[-1]
        start_x.append(base + timedelta(minutes=int(td["depart_time"][i0]))); start_y.append(st2y[_station_key(td["depart_station"][i0])])
        end_x.append(base + timedelta(minutes=int(td["arrive_time"][i1])));   end_y.append(st2y[_station_key(td["arrive_station"][i1])])


        # 折返し（半円）をまとめる
        for a_t, b_t, st, ori, sv in detect_turnbacks(td, idxs, st2y):
            add_cap_arc_buffer(cap_x[sv][ori], cap_y[sv][ori], float(a_t), float(b_t), st2y[st], ori, base)

        # 列車の線を追加（2トレース）
        if xs_local:
            fig.add_trace(go.Scatter(
                x=xs_local, y=ys_local, mode="lines",
                line=dict(width=2, color=COLORS["local_line"]),
                customtd=cd_local,
                hovertemplate=(
                    "task_idx=%{customtd[0]}<br>"
                    "train_id=%{customtd[1]}<br>"
                    "%{customtd[2]}→%{customtd[3]}<br>"
                    "%{customtd[4]}→%{customtd[5]}<br>"
                    "service=%{customtd[6]}<br>"
                    "direction=%{customtd[7]}<extra></extra>"
                ),
                showlegend=False,
            ))


        if xs_rapid:
            fig.add_trace(go.Scatter(
            x=xs_rapid, y=ys_rapid, mode="lines",
            line=dict(width=2, color=COLORS["rapid_line"]),
            customtd=cd_rapid,
            hovertemplate=(
                "task_idx=%{customtd[0]}<br>"
                "train_id=%{customtd[1]}<br>"
                "%{customtd[2]}→%{customtd[3]}<br>"
                "%{customtd[4]}→%{customtd[5]}<br>"
                "service=%{customtd[6]}<br>"
                "direction=%{customtd[7]}<extra></extra>"
            ),
            showlegend=False,
        ))


    # 折返し（半円）を一括追加（点線）
    for sv in ("local", "rapid"):
        for ori, clr in (("up_cap", COLORS["local_turn"] if sv=="local" else COLORS["rapid_turn"]),
                         ("down_cap", COLORS["local_turn"] if sv=="local" else COLORS["rapid_turn"])):
            xs, ys = cap_x[sv][ori], cap_y[sv][ori]
            if xs:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(width=2, dash="dot", color=clr),
                    hovertemplate=f"turnback ({sv})<extra></extra>",
                    showlegend=False,
                ))


    # 出庫/収納マーカー（黒固定）
    if start_x:
        fig.add_trace(go.Scatter(
            x=start_x, y=start_y, mode="markers",
            marker=dict(symbol="circle-open", size=10, line=dict(color="black", width=2)),
            hoverinfo="skip", showlegend=False,
        ))
    if end_x:
        fig.add_trace(go.Scatter(
            x=end_x, y=end_y, mode="markers",
            marker=dict(symbol="triangle-up", size=9, color="black"),
            hoverinfo="skip", showlegend=False,
        ))

    # 軸設定（Aが上、横方向のみズーム）――駅ごとに横線（Yグリッド）も表示
    first_hour_min = (tmin // 60) * 60
    tick0_hour = base + timedelta(minutes=first_hour_min)
    first_10m_min = (tmin // 10) * 10
    tick0_10m = base + timedelta(minutes=first_10m_min)

    fig.update_yaxes(
        tickmode="array",
        tickvals=yvals,
        ticktext=station_order,
        range=[y_top, y_bottom],
        autorange="reversed",
        title_text="駅",
        fixedrange=True,
        showgrid=True,
        gridcolor="#CFCFCF",
        gridwidth=1,
    )
    fig.update_xaxes(
        title_text="時刻",
        tickformat="%H:%M",
        tick0=tick0_hour,
        dtick=MS1HOUR,        # ラベルは 1h 刻み
        range=[x_min, x_max],
        showgrid=True,
        gridwidth=1,
        minor=dict(          # 10分ごとの縦グリッド（ラベル無し）
            tick0=tick0_10m,
            dtick=MS10MIN,
            showgrid=True,
            gridwidth=0.5,
            gridcolor="#DDDDDD",
        ),
    )

    height = 80 * len(station_order) + 120
    width = 2000
    fig.update_layout(
        hovermode="closest",   # 近い1件だけ
        hoverdistance=20,
        dragmode="pan",
        margin=dict(l=60, r=20, t=40, b=60),
        width=width, height=height,
        showlegend=False,
        uirevision=True,
    )
    return fig
    