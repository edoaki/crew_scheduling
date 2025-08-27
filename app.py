import streamlit as st
from pathlib import Path
from viz.loader import load_npz_bundle_or_legacy, station_order_from_config
from timetable.generator import generate_and_save
from viz.plot import build_diamond_figure

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
        out_path = str(DATA_DIR / f"{out_name}.npz")
        ok, msg = generate_and_save(station_yaml, train_yaml, out_path)
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
data = load_npz_bundle_or_legacy(str(DATA_DIR / sel_name))

# 駅の縦順（Aが最上段）
if data["topology"] is None:
    station_order = station_order_from_config(str(CONFIG_DIR / "station.yaml"))
else:
    station_order = [s if isinstance(s, str) else s.decode("utf-8", "ignore") for s in data["topology"]]

train_ids = [str(t) for t in data["train_ids"]]
unique_trains = sorted(set(train_ids), key=lambda x: (len(x), x))

with st.sidebar:
    st.header("表示フィルタ")
    selected = st.multiselect("表示する列車（チェック）", options=unique_trains, default=unique_trains)

fig = build_diamond_figure(
    data=data,
    station_order=station_order,
    visible_trains=set(selected),
)

st.plotly_chart(
    fig,
    use_container_width=False,
    config={"scrollZoom": True, "displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
)
