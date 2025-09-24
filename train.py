import os
from pathlib import Path
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
IMPROVE_PATIENCE = 10

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
            cur_mean, _ = evaluate_mean_reward(policy, vec_env, BATCH_SIZE, device,"model")
            print("cur mean",cur_mean)
            base_mean, _ = evaluate_mean_reward(baseline_policy, vec_env, BATCH_SIZE, device,"base")
            print("base mean",base_mean)

        improved = (cur_mean > base_mean)
        if improved:
            one_improve +=1
            baseline_policy.load_state_dict(policy.state_dict())
            epochs_since_improve = 0
            ckpt = checkpoint_path(SAVE_ROOT, RUN_NAME, epoch)
            save_checkpoint(ckpt, epoch, policy, baseline_policy, optimizer, cur_mean, base_mean)
        else:
            epochs_since_improve += 1

        print(f"****[Epoch {epoch:03d}] "
              f"train_loss={loss_val:.6f}  "
              f"cur_mean_reward={cur_mean:.6f}  "
              f"baseline_mean_reward={base_mean:.6f}  "
              f"improved={improved}  "
              f"no_improve_for={epochs_since_improve}")

        if epochs_since_improve >= IMPROVE_PATIENCE and one_improve>0:
            print(f"Early stop at epoch {epoch} (no improvement for {IMPROVE_PATIENCE} epochs).")
            break

if __name__ == "__main__":
    main()
