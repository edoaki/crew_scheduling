import pandas as pd
import matplotlib.pyplot as plt

# ログファイルのパス
LOG_PATH = "checkpoints/test2_config/training_log.csv"  # 例: "runs/myrun/training_log.csv"
# CSV を読み込み
df = pd.read_csv(LOG_PATH)

# 同一 epoch に複数行がある場合は「最後の行」を採用
df_epoch = df.groupby("epoch", as_index=False).agg({
    "cur_mean": "last",
    "sampling_mean": "last"
})

# 可視化
plt.figure()
plt.plot(df_epoch["epoch"], df_epoch["cur_mean"], label="cur_mean")
plt.plot(df_epoch["epoch"], df_epoch["sampling_mean"], label="sampling_mean")
plt.xlabel("Epoch")
plt.ylabel("Mean Reward")
plt.title("Reward Transition (per epoch, last record)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig("reward_curve.png", dpi=150, bbox_inches="tight")
plt.show()
