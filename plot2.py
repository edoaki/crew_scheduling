import pandas as pd
import matplotlib.pyplot as plt

# ログファイルのパス
LOG_PATH = "checkpoints/test2_config/training_log.csv"

# CSV を読み込み
df = pd.read_csv(LOG_PATH)

# 同一 epoch に複数行がある場合は「最後の行」を採用（baseline_mean は除外）
metrics = [
    "cur_mean",
    "sampling_mean",
    "cost",
    "sampling_cost",
]
agg_dict = {m: "last" for m in metrics}
df_epoch = df.groupby("epoch", as_index=False).agg(agg_dict)

# 表示用に cost / sampling_cost にマイナスを掛ける（保存済みの元値は変更しない）
df_plot = df_epoch.copy()
df_plot["cost"] = -df_plot["cost"]
df_plot["sampling_cost"] = -df_plot["sampling_cost"]

# 可視化（6系列まとめて表示）
plt.figure()
for m in metrics:
    plt.plot(df_plot["epoch"], df_plot[m], label=m)

plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Metrics Transition (per epoch, last record) - cost flipped sign")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("metrics_curve_no_baseline_cost_negated.png", dpi=150, bbox_inches="tight")
plt.show()
