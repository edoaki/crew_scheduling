# round_view.py
# 同じNPZから round と feat を読んで、ラウンドごとのタスク/便乗情報を可視化する簡易UI
import streamlit as st
from pathlib import Path
import numpy as np

# io_npz 読み込み（プロジェクト構成に応じて両対応）

from timetable.io_npz import load_timetable_bundle

DATA_DIR = Path("data")

st.set_page_config(page_title="ラウンド検査ビュー", layout="wide")
st.title("ラウンド検査ビュー（NPZ直読み）")

# NPZ選択
npz_files = sorted(DATA_DIR.glob("*.npz"))
if not npz_files:
    st.info("data フォルダに .npz がありません。生成後に再実行してください。")
    st.stop()

sel_name = st.selectbox("表示するNPZを選択", [p.name for p in npz_files], index=0)
bundle = load_timetable_bundle(str(DATA_DIR / sel_name))

tt = bundle["tt"]
rd = bundle.get("round", {})
feat = bundle.get("features", {})
meta = bundle.get("meta", {})

if not rd or "round_ptr" not in rd or "round_tt_idx" not in rd:
    st.error("このNPZには round 情報が含まれていません（round_ptr / round_tt_idx が無い）。")
    st.stop()

# 補助関数
def mm_to_hhmm(m: int) -> str:
    h = int(m) // 60
    mm = int(m) % 60
    return f"{h:02d}:{mm:02d}"

labels = meta.get("station_label_vocab", None)  # 数値ID→ラベル
def lab(i: int) -> str:
    if labels is not None and 0 <= i < len(labels):
        return labels[i]
    return str(i)

# Round 基本配列
r_ptr = rd["round_ptr"]           # [R+1]
r_flat = rd["round_tt_idx"]       # [*] 0始まりのttインデックス
r_anchor = rd.get("round_anchor_min", None)  # [R] minutes

R = int(r_ptr.shape[0] - 1)
st.write(f"ラウンド数: **{R}**")

# セッション状態：現在のラウンド（1始まりでUI表示）
if "round_id" not in st.session_state:
    st.session_state.round_id = 1

# 左右ボタン + スライダー
col_btn_l, col_slider, col_btn_r = st.columns([1, 6, 1])
with col_btn_l:
    if st.button("← 前のラウンド", use_container_width=True) and st.session_state.round_id > 1:
        st.session_state.round_id -= 1
with col_slider:
    st.session_state.round_id = st.slider("ラウンドを選択", 1, R, st.session_state.round_id, key="round_slider")
with col_btn_r:
    if st.button("次のラウンド →", use_container_width=True) and st.session_state.round_id < R:
        st.session_state.round_id += 1

r = st.session_state.round_id  # 1..R
s, e = int(r_ptr[r-1]), int(r_ptr[r])
idxs = r_flat[s:e].astype(int)  # 0-based タスク行インデックス
anchor_txt = mm_to_hhmm(int(r_anchor[r-1])) if r_anchor is not None and r-1 < len(r_anchor) else "N/A"

st.subheader(f"ラウンド {r} の概要")
st.write(f"- anchor時刻: **{anchor_txt}**")
st.write(f"- タスク数: **{len(idxs)}**")
st.write(f"- タスクID（1始まり表示）: {list((idxs + 1).tolist())}")

# タスクの頭数行を一覧（任意）
with st.expander("このラウンドのタスク（ダイヤ基本情報）", expanded=False):
    # depart/arrive の表示は tt に依存（tt がラベルかIDかは生成側次第）
    # ここでは配列がある前提で安全に取り出す
    def safe(arr, i):
        try:
            return arr[i]
        except Exception:
            return ""
    rows = []
    for t0 in idxs:
        depS = safe(tt.get("depart_station", []), t0)
        arrS = safe(tt.get("arrive_station", []), t0)
        depT = safe(tt.get("depart_time", []), t0)
        arrT = safe(tt.get("arrive_time", []), t0)
        train = safe(tt.get("train_ids", []), t0)
        rows.append(dict(
            task_id=int(t0)+1,
            train_id=str(train),
            depart_station=str(depS),
            depart_time=mm_to_hhmm(int(depT)) if str(depT).isdigit() else str(depT),
            arrive_station=str(arrS),
            arrive_time=mm_to_hhmm(int(arrT)) if str(arrT).isdigit() else str(arrT),
        ))
    st.dataframe(rows, use_container_width=True, hide_index=True)

# feat から task×station の便乗情報を復元・表示
req_keys = [
    "cache_task_ptr","cache_station_ids","cache_must_be_by_min",
    "cache_is_hitch","cache_hops","cache_hitch_minutes",
    "cache_path_ptr","cache_path_task_ids"
]
missing = [k for k in req_keys if k not in feat]
if missing:
    st.warning(f"feat に必要なキーが見つかりません: {missing}")
else:
    tptr = feat["cache_task_ptr"].astype(int)       # [N+1]
    st_ids = feat["cache_station_ids"].astype(int)  # [K]
    must = feat["cache_must_be_by_min"].astype(int) # [K]
    hitch = feat["cache_is_hitch"].astype(int)      # [K]
    hops = feat["cache_hops"].astype(int)           # [K]
    hmin = feat["cache_hitch_minutes"].astype(int)  # [K]
    pptr = feat["cache_path_ptr"].astype(int)       # [K+1]
    pids = feat["cache_path_task_ids"].astype(int)  # [P] 1始まりの便乗タスクID

    st.subheader("タスク×駅ごとの便乗情報")
    for t0 in idxs:  # 0-based
        s_k, e_k = int(tptr[t0]), int(tptr[t0+1])
        st.markdown(f"**Task {int(t0)+1}**")
        if s_k == e_k:
            st.write("- 登録された駅はありません。")
            continue

        # 表形式で出す
        rows = []
        for k in range(s_k, e_k):
            ps, pe = int(pptr[k]), int(pptr[k+1])
            rows.append(dict(
                station_id=int(st_ids[k]),
                station_label=lab(int(st_ids[k])),
                must_be_by=mm_to_hhmm(int(must[k])),
                is_hitch=bool(hitch[k]),
                hops=int(hops[k]),
                hitch_minutes=int(hmin[k]),
                hitch_path=list(map(int, pids[ps:pe].tolist())),
            ))
        st.dataframe(rows, use_container_width=True, hide_index=True)
