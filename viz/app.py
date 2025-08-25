# viz/app.py
import glob
import os

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import (
    Button, ColumnDataSource, CDSView, BooleanFilter,
    Div, HoverTool, MultiSelect, Select, TextInput
)

from utils.time import hhmm_label
from utils.npz import load_npz_any, derive_topology_and_y
from viz.plot import build_sources, make_figure

DATA_TIMETABLE_DIR = os.path.join("data", "timetable")
CONFIG_DIR = "config"
STATION_YAML = os.path.join(CONFIG_DIR, "station.yaml")
TRAIN_YAML = os.path.join(CONFIG_DIR, "train.yaml")

HAVE_GENERATOR = True
try:
    from timetable.simulator import generate_timetable
except Exception:
    HAVE_GENERATOR = False

def list_npz_files():
    return sorted(glob.glob(os.path.join(DATA_TIMETABLE_DIR, "*.npz")))

file_select = Select(title="NPZを選択 (data/timetable)", value="", options=list_npz_files())
npz_name_input = TextInput(title="保存するNPZ名（拡張子不要）", value="run1")
gen_btn = Button(label="生成（config を再読込して data/timetable/{名前}.npz に保存）", button_type="success", disabled=(not HAVE_GENERATOR))
train_select = MultiSelect(title="列車IDで絞り込み（複数選択可）", size=8)
clear_btn = Button(label="絞り込み解除", button_type="default")
info_div = Div(text="区間をクリックすると詳細を表示。", height=110)

state = {
    "bundle": None,
    "topo": [],
    "ymap": {},
    "sources": None,
    "renderers": None,
    "fig": None,
    "selected_trains": [],
}

def apply_train_filter(selected_trains):
    if state["sources"] is None or state["renderers"] is None:
        return

    seg_src, start_src, end_src, arc_src = state["sources"]
    seg_r = state["renderers"]["seg"]
    start_r = state["renderers"]["start"]
    end_r = state["renderers"]["end"]
    arc_r = state["renderers"]["arc"]

    # マスク作成（選択なし＝全True）
    if not selected_trains:
        seg_mask = [True] * len(seg_src.data["train_id"])
        start_mask = [True] * len(start_src.data["train_id"])
        end_mask = [True] * len(end_src.data["train_id"])
        arc_mask = [True] * len(arc_src.data.get("train_id", []))
    else:
        seg_mask = [t in selected_trains for t in seg_src.data["train_id"]]
        start_mask = [t in selected_trains for t in start_src.data["train_id"]]
        end_mask = [t in selected_trains for t in end_src.data["train_id"]]
        arc_mask = [t in selected_trains for t in arc_src.data.get("train_id", [])]

    # Bokeh 3.x: CDSView(filter=...) を必ずセット（Noneは不可）
    seg_r.view = CDSView(filter=BooleanFilter(seg_mask))
    start_r.view = CDSView(filter=BooleanFilter(start_mask))
    end_r.view = CDSView(filter=BooleanFilter(end_mask))
    if len(arc_mask) == len(arc_src.data.get("train_id", [])):
        arc_r.view = CDSView(filter=BooleanFilter(arc_mask))


def refresh_plot(npz_path: str):
    if not npz_path or not os.path.exists(npz_path):
        info_div.text = "<b>NPZが見つかりません。</b>"
        return

    bundle = load_npz_any(npz_path)
    topo, ymap = derive_topology_and_y(bundle, STATION_YAML)

    seg_src, start_src, end_src, arc_src = build_sources(bundle, ymap)
    tids = sorted(list(set(seg_src.data["train_id"])))
    train_select.options = tids

    p, renders = make_figure(topo, ymap, seg_src, start_src, end_src, arc_src)

    hover = HoverTool(renderers=[renders["seg"]], tooltips=[
        ("train", "@train_id"),
        ("区間", "@depart_station → @arrive_station"),
        ("出発", "@depart_time{0}"),
        ("到着", "@arrive_time{0}"),
        ("種別/方向", "@service / @direction"),
    ])
    p.add_tools(hover)

    def on_select(attr, old, new):
        inds = seg_src.selected.indices
        if not inds:
            info_div.text = "区間をクリックすると詳細を表示。"
            return
        i = inds[0]
        d = {k: seg_src.data[k][i] for k in ["train_id","depart_station","arrive_station","depart_time","arrive_time","service","direction"]}
        info_div.text = (
            f"<b>Train:</b> {d['train_id']}<br>"
            f"<b>区間:</b> {d['depart_station']} → {d['arrive_station']}<br>"
            f"<b>出発:</b> {hhmm_label(int(d['depart_time']))} / "
            f"<b>到着:</b> {hhmm_label(int(d['arrive_time']))}<br>"
            f"<b>種別/方向:</b> {d['service']} / {d['direction']}"
        )
    seg_src.selected.on_change("indices", on_select)

    state["bundle"] = bundle
    state["topo"] = topo
    state["ymap"] = ymap
    state["sources"] = (seg_src, start_src, end_src, arc_src)
    state["renderers"] = renders
    state["fig"] = p
    right.children = [p]

    apply_train_filter(state["selected_trains"])

def on_file_change(attr, old, new):
    refresh_plot(file_select.value)

def on_clear():
    train_select.value = []
    state["selected_trains"] = []
    apply_train_filter([])

def on_train_select(attr, old, new):
    state["selected_trains"] = list(train_select.value)
    apply_train_filter(state["selected_trains"])

def on_generate():
    name = npz_name_input.value.strip()
    if not name:
        info_div.text = "<b>NPZ名を入力してください。</b>"
        return
    if not HAVE_GENERATOR:
        info_div.text = "<b>generate_timetable が使用できません。構成を確認してください。</b>"
        return
    os.makedirs(DATA_TIMETABLE_DIR, exist_ok=True)
    save_path = os.path.join(DATA_TIMETABLE_DIR, f"{name}.npz")
    try:
        _ = generate_timetable(STATION_YAML, TRAIN_YAML, save_path=save_path)
        file_select.options = list_npz_files()
        file_select.value = save_path
        info_div.text = f"<b>生成完了:</b> {save_path}"
    except Exception as e:
        info_div.text = f"<b>生成に失敗:</b> {e}"

file_select.on_change("value", on_file_change)
train_select.on_change("value", on_train_select)
clear_btn.on_click(on_clear)
gen_btn.on_click(on_generate)

left = column(
    file_select,
    npz_name_input,
    gen_btn,
    train_select,
    clear_btn,
    info_div,
    width=380,
    height=720,
    sizing_mode="fixed",
)
    
right = column(sizing_mode="stretch_both")
layout = row(left, right, sizing_mode="stretch_both")
curdoc().add_root(layout)

files = list_npz_files()
if files:
    file_select.value = files[0]
else:
    info_div.text = "data/timetable にNPZがありません。生成してください。"
