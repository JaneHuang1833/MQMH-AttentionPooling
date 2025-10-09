import json, argparse, numpy as np, matplotlib.pyplot as plt, torch
from attention_test import SER_From_Emb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npy",  required=True)
    ap.add_argument("--out_png", default="attn.png")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    id2label = {int(k):v for k,v in ckpt["labels"].items()}
    in_dim = ckpt["in_dim"]; cfg = ckpt["args"]
    model = SER_From_Emb(in_dim=in_dim, num_labels=len(id2label),
                         pooling=cfg["pooling"], attn_hidden=cfg["attn_hidden"],
                         mq_q=cfg["mq_q"], mq_h=cfg["mq_h"], clf_hidden=cfg["clf_hidden"])
    model.load_state_dict(ckpt["model"]); model.eval()

    x = np.load(args.npy).astype(np.float32)        # (T,D)
    T = x.shape[0]
    X = torch.from_numpy(x).unsqueeze(0)            # (1,T,D)
    M = torch.ones(1, T, dtype=torch.long)
    with torch.no_grad():
        logits, W = model(X, M, return_weights=True)
        prob = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        pred = id2label[int(prob.argmax())]

    plt.figure(figsize=(10,3))

    # MQMHA: (B,Q,H,T) 展平为 (Q*H, T) 的热力图
    W = W.squeeze(0).numpy()                    # (Q,H,T)
    Q,H,T = W.shape
    WH = W.reshape(Q*H, T)

    # 鲁棒归一化：避免极少数峰值把整图“压黑”
    vmax = float(np.quantile(WH, 0.995))
    vmin = 0.0

    fig, ax = plt.subplots(figsize=(12, 3.6), dpi=200)
    im = ax.imshow(
        WH,
        origin="lower", aspect="auto",
        interpolation="nearest",
        vmin=vmin, vmax=vmax,
    )

    # X 轴：时间刻度
    xticks = np.linspace(0, T - 1, num=min(10, T), dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(int(t)) for t in xticks])
    ax.set_xlabel("Time index")

    # Y 轴：按 Query 分段，居中标注
    centers = [q * H + (H - 1) / 2 for q in range(Q)]
    ax.set_yticks(centers)
    ax.set_yticklabels([f"Query {q}" for q in range(Q)])
    ax.set_ylabel("Queries × Heads")

    # 在 Query 分界处画细分割线
    for q in range(1, Q):
        ax.axhline(q * H - 0.5, lw=0.8, alpha=0.6)

    # 去掉多余边框，紧凑布局
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Attention weight", rotation=90)

    ax.set_title(f"MQMHA Attention Heatmap | Pred: {pred}")
    fig.tight_layout()
    fig.savefig(args.out_png, bbox_inches="tight", dpi=300)




if __name__ == "__main__":
    main()