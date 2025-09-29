import os
from pathlib import Path
import time
import csv
import torch
from rl_core.build import load_configs, build_generator, build_env, build_embeddings, build_policy
from rl_core.train_loop import reinforce_step, evaluate_mean_reward, save_checkpoint, checkpoint_path

# ====== 設定（必要に応じて run.yaml から読み込むように変えてもOK） ======
DATA_DIR = Path("data")
CONFIG_DIR = Path("test2_config")  # ここが run_name の元になる
RUN_NAME = CONFIG_DIR.name         # pth 名に反映
SAVE_ROOT = Path("checkpoints")
BATCH_SIZE = 100

NUM_EPOCHS = 100
UPDATES_PER_EPOCH = 1
LR = 1e-4
GRAD_CLIP = 1.0
IMPROVE_PATIENCE = 100

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = load_configs(CONFIG_DIR)
    paths = cfg["paths"]
    constraints = cfg["constraints"]

    generator = build_generator(paths)
    station_emb, time_emb = build_embeddings(paths["station_yaml"], paths["encoding_yaml"])
    vec_env = build_env(generator, constraints, batch_size=BATCH_SIZE, device=device)

    policy = build_policy(station_emb, time_emb).to(device)
    baseline_policy = build_policy(station_emb, time_emb).to(device)
    baseline_policy.load_state_dict(policy.state_dict())

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    epochs_since_improve = 0
    one_improve =0
    
    # --- ログ保存設定 ---
    run_dir = Path(SAVE_ROOT) / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    # 現在時刻
    log_path = run_dir / f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    # 既存ファイルがなければヘッダを書く
    if not log_path.exists():
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "cur_mean", "sampling_mean", "baseline_mean","env_reward","sampling_env_reward","cost","sampling_cost"])

    start_time = time.time()

    print("Starting training...")
    for epoch in range(1, NUM_EPOCHS + 1):
        policy.train()
        for _ in range(UPDATES_PER_EPOCH):
            loss_val = reinforce_step(
                policy=policy,
                baseline_policy=baseline_policy,
                vec_env=vec_env,
                batch_size=BATCH_SIZE,
                device=device,
                optimizer=optimizer,
                grad_clip=GRAD_CLIP,
            )

        policy.eval()
        with torch.no_grad():
            cur_mean, sampling_mean,sol,reward_dict= evaluate_mean_reward(policy, vec_env, BATCH_SIZE, device,"model",return_env_reward=True)
            # print("cur mean",cur_mean)
            base_mean, base_sampling_mean,_,_ = evaluate_mean_reward(baseline_policy, vec_env, BATCH_SIZE, device,"model")
            # print("base mean",base_mean)
        # --- CSVに1エポック分を追記 ---
        env_reward = reward_dict["env_reward"]
        sam_env_reward = reward_dict["sampling_env_reward"]
        cost = reward_dict["cost"]
        sam_cost = reward_dict["sampling_cost"]
        # 小数点以下3桁までに丸めて保存
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            # writer.writerow([epoch, float(loss_val), float(cur_mean), float(sampling_mean), float(base_mean),float(env_reward),float(sam_env_reward),float(cost),float(sam_cost)])
            writer.writerow([epoch, round(float(loss_val),6), round(float(cur_mean),3), round(float(sampling_mean),3), round(float(base_mean),3),round(float(env_reward),3),round(float(sam_env_reward),3),round(float(cost),3),round(float(sam_cost),3)])
        improved = (cur_mean > base_mean)
        # improved = (sampling_mean > base_sampling_mean)
        if improved:
            one_improve +=1
            baseline_policy.load_state_dict(policy.state_dict())
            epochs_since_improve = 0
            ckpt = checkpoint_path(SAVE_ROOT, RUN_NAME, epoch)
            save_checkpoint(ckpt, epoch, policy, baseline_policy, optimizer, cur_mean, base_mean)
        else:
            epochs_since_improve += 1

        elapsed = time.time() - start_time
        print(f"****[Epoch {epoch:03d}] "
              f"train_loss={loss_val:.6f}  "
              f"cur_mean_reward={cur_mean:.6f}  "
              f"baseline_mean_reward={base_mean:.6f}  "
              f"improved={improved}  "
              f"no_improve_for={epochs_since_improve}"
                f"  elapsed_time={elapsed}min"
                )

        if epochs_since_improve >= IMPROVE_PATIENCE and one_improve>0:
            print(f"Early stop at epoch {epoch} (no improvement for {IMPROVE_PATIENCE} epochs).")
            break

    # 最終モデルを保存 (base_line ではなく policy の方)
    torch.save(policy.state_dict(), str(SAVE_ROOT / RUN_NAME / f"{RUN_NAME}.final.pth"))
    print("Training completed.")
    print(f"Total elapsed time:{ (time.time() - start_time) } ")

if __name__ == "__main__":
    main()
