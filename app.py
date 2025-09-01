import streamlit as st
from pathlib import Path
from timetable.generator import generate_and_save
from viz.plot import build_diamond_figure
from viz.plot_core import hhmm
from utils.io_npz import load_timetable_bundle ,station_order_from_config
from streamlit_plotly_events import plotly_events
import numpy as np

DATA_DIR = Path("data")
CONFIG_DIR = Path("config")

st.set_page_config(page_title="ダイヤ可視化", layout="wide")
st.title("ダイヤ可視化")

with st.sidebar:
    st.header("生成")
    out_name = st.text_input("保存ファイル名（拡張子なし）", value="timetable")
    if st.button("生成", type="primary"):
        station_yaml = str(CONFIG_DIR / "station.yaml")
        train_yaml = str(CONFIG_DIR / "train.yaml")
        constraints_yaml = str(CONFIG_DIR / "constraints.yaml")
        out_path = str(DATA_DIR / f"{out_name}.npz")
        ok, msg = generate_and_save(station_yaml, train_yaml,constraints_yaml,out_path)
        if ok:
            st.success(f"生成しました: {out_path}")
        else:
            st.error(f"生成に失敗: {msg}")

st.sidebar.markdown(
    """
    <style>
    [data-testid="stSidebar"] [data-testid="stMultiSelect"] div[role="listbox"]{
        max-height: 60vh !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

npz_files = sorted(DATA_DIR.glob("*.npz"))
if not npz_files:
    st.info("data フォルダに .npz がありません。左の「生成」を使うか、ファイルを配置してください。")
    st.stop()

sel_name = st.selectbox("表示するNPZを選択", [p.name for p in npz_files], index=0)
tt, meta, task_station_cache,  rounds = load_timetable_bundle(str(DATA_DIR / sel_name))

# 駅の縦順（Aが最上段）
station_order = station_order_from_config(str(CONFIG_DIR / "station.yaml"))

train_ids = [str(t) for t in tt["train_ids"]]
unique_trains = sorted(set(train_ids), key=lambda x: (len(x), x))

with st.sidebar:
    st.header("表示フィルタ")
    selected = st.multiselect("表示する列車（チェック）", options=unique_trains, default=unique_trains)

fig = build_diamond_figure(
    data=tt,
    station_order=station_order,
    visible_trains=set(selected),
)

# ---- 図表示（クリック取得） ----
try:
    clicked = plotly_events(
        fig,
        click_event=True, hover_event=False, select_event=False,
        override_width=2000, override_height=None, key="diamond"
    )
except Exception:
    st.info("クリックを有効にするには 'streamlit-plotly-events' をインストールしてください。")
    st.plotly_chart(
        fig,
        use_container_width=False,
        config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
    )
    clicked = []

# ---- 補助: 時刻表記 ----
def mm_to_hhmm(m: int) -> str:
    h = int(m) // 60
    mm = int(m) % 60
    return f"{h:02d}:{mm:02d}"

# ---- クリックしたタスクの cache を図の下に表示 ----
with st.container():
    st.subheader("クリックしたタスクの cache 情報")

    if clicked:
        print(clicked)
        ev = clicked[0]

        # まずイベントに customdata が来ていればそれを使う
        cd = ev.get("customdata", None)

        # 来ていない場合は、curveNumber / pointNumber から fig 側の customdata を引く
        if cd is None:
            try:
                cn = int(ev.get("curveNumber", -1))
                pn = ev.get("pointNumber", ev.get("pointIndex", -1))
                pn = int(pn) if pn is not None else -1
                if 0 <= cn < len(fig.data):
                    tr = fig.data[cn]
                    cd_arr = getattr(tr, "customdata", None)
                    if cd_arr is not None and 0 <= pn < len(cd_arr):
                        cd = cd_arr[pn]
            except Exception:
                cd = None

        # ここまでで customdata が取れない＝折返しアークや出庫/収納マーカー等をクリック
        if cd is None:
            st.warning("この点には customdata がありません（折返しアーク/始終点マーカー等の可能性）。線分をクリックしてください。")
            st.stop()

        # 形をそろえる
        if isinstance(cd, (tuple, np.ndarray)):
            cd = list(cd)

        # customdata[0] に 0始まりの task_idx が入っている前提（plot.py の cd_seg）
        if len(cd) == 0 or cd[0] is None or not str(cd[0]).isdigit():
            st.warning("クリック点の customdata[0] から task_idx を解釈できません。")
            st.stop()

        task_idx = int(cd[0])  # 0始まり
        st.write(f"task_id: {task_idx + 1}")

        # 以降は既存の cache 表示ロジック（そのまま）
        req_keys = [
            "cache_task_ptr","cache_station_ids","cache_must_be_by_min",
            "cache_is_hitch","cache_hops","cache_hitch_minutes",
            "cache_path_ptr","cache_path_task_ids"
        ]
        missing = [k for k in req_keys if k not in task_station_cache]
        if missing:
            st.warning("必要な cache キーが見つかりません: " + ", ".join(missing))
        else:
            tptr = task_station_cache["cache_task_ptr"].astype(int)       # [N+1]
            st_ids = task_station_cache["cache_station_ids"].astype(int)  # [K]
            must  = task_station_cache["cache_must_be_by_min"].astype(int)# [K]
            hops  = task_station_cache["cache_hops"].astype(int)          # [K]
            hitch = task_station_cache["cache_is_hitch"].astype(int)      # [K]
            hitch_min = task_station_cache["cache_hitch_minutes"].astype(int)  # [K]
            pptr  = task_station_cache["cache_path_ptr"].astype(int)      # [R+1]
            pflat = task_station_cache["cache_path_task_ids"].astype(int) # [M]

            s = int(tptr[task_idx]); e = int(tptr[task_idx + 1])
            st.write({
                "station_ids": st_ids[s:e].tolist(),
                "must_by_min": must[s:e].tolist(),
                "hops": hops[s:e].tolist(),
                "is_hitch": hitch[s:e].tolist(),
                "hitch_minutes": hitch_min[s:e].tolist(),
            })

# ---- Round 指定（左右ボタン＋スライダー）→ タスクID表示 ----
with st.container():
    st.subheader("ラウンド指定")

    if not rounds or "round_ptr" not in rounds or "round_tt_idx" not in rounds:
        st.warning("round_ptr / round_tt_idx が見つからないため、ラウンド表示をスキップします。")
    else:
        r_ptr = rounds["round_ptr"].astype(int)       # [R+1]
        r_flat = rounds["round_tt_idx"].astype(int)   # [*] 0始まりの tt index 列
        r_anchor = rounds.get("round_anchor_min", None)  # [R] （任意）

        R = int(r_ptr.shape[0] - 1)
        if "round_sel" not in st.session_state:
            st.session_state.round_sel = 1

        c1, c2, c3 = st.columns([1, 6, 1])
        with c1:
            if st.button("◀"):
                st.session_state.round_sel = max(1, st.session_state.round_sel - 1)
        with c2:
            st.session_state.round_sel = st.slider("round", 1, R, st.session_state.round_sel, key="round_slider")
        with c3:
            if st.button("▶"):
                st.session_state.round_sel = min(R, st.session_state.round_sel + 1)

        r = int(st.session_state.round_sel)
        s, e = int(r_ptr[r - 1]), int(r_ptr[r])
        idxs = r_flat[s:e]  # 0始まりの tt index
        ids_1based = (idxs + 1).tolist()

        if r_anchor is not None and len(r_anchor) >= r:
            st.write(f"anchor={mm_to_hhmm(int(r_anchor[r - 1]))}  /  round={r}")
        else:
            st.write(f"round={r}")

        st.write(f"タスクID（1始まり）: {', '.join(map(str, ids_1based)) if ids_1based else '(なし)'}")
