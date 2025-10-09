import os, csv, json, math, argparse, random
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


class EmbCsvDataset(Dataset):  # database
    def __init__(self, csv_path, label2id):
        self.items = []
        with open(csv_path) as f:
            r = csv.DictReader(f)
            for row in r:
                self.items.append((row["path"], row["label"]))
        self.label2id = label2id

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        path, lab = self.items[i]
        x = np.load(path)
        x = x.astype(np.float32)
        return {"x": x, "y": self.label2id[lab], "path": path}


class Collate:  # 动态padding到同一T_max（用补零＋掩码来处理可变长度序列）
    def __call__(self, batch):
        lens = [b["x"].shape[0] for b in batch]
        Tmax = max(lens)
        D = batch[0]["x"].shape[1]  # feature dim
        X = np.zeros((len(batch), Tmax, D), dtype=np.float32)  # 初始化零数组
        M = np.zeros((len(batch), Tmax), dtype=np.int64)  # mask
        Y = np.array([b["y"] for b in batch], dtype=np.int64)  # 标签整数向量
        P = [b["path"] for b in batch]
        for i, b in enumerate(batch):
            t = b["x"].shape[0]  # 每个batch的序列长度
            X[i, :t, :] = b["x"]  # 将每个样本的前t帧拷入X，形成长度相同（Tmax，缺失位置为0）的batch
            M[i, :t] = 1
        return {
            "x": torch.from_numpy(X),  # (B,T,D)
            "mask": torch.from_numpy(M),  # (B,T)
            "y": torch.from_numpy(Y),
            "paths": P
        }


class MQMHA_Pooling(nn.Module):

    def __init__(self, in_dim, num_queries=2, num_heads=4, mlp_hidden=128):
        super().__init__()
        assert in_dim % num_heads == 0, "D 必须能被H整除"
        self.Q, self.H = num_queries, num_heads
        self.Dh = in_dim // num_heads

        # 为每个 (q,h) 定义一个小打分网络 F^{(q,h)}
        self.scorers = nn.ModuleList(
            nn.ModuleList(
                [nn.Sequential(nn.Linear(self.Dh, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))
                 for _ in range(self.H)]
            ) for _ in range(self.Q)
        )

    def forward(self, x, mask):
        # x: (B,T,D) -> split heads
        B, T, D = x.shape
        x = x.view(B, T, self.H, self.Dh)  # (B,T,H,Dh)
        outs_mu, outs_std, all_w = [], [], []

        for q in range(self.Q):
            mu_heads, std_heads = [], []
            w_heads = []
            for h in range(self.H):
                e = self.scorers[q][h](x[:, :, h, :]).squeeze(-1)  # (B,T)
                e = e.masked_fill(mask == 0, -1e9)
                w = torch.softmax(e, dim=-1)  # (B,T)
                w_ = w.unsqueeze(1)
                mu = torch.bmm(w_, x[:, :, h, :]).squeeze(1)  # (B,Dh)
                m2 = torch.bmm(w_, x[:, :, h, :] ** 2).squeeze(1)
                var = torch.clamp(m2 - mu * mu, min=1e-12)
                std = torch.sqrt(var)
                mu_heads.append(mu)
                std_heads.append(std)
                w_heads.append(w)

            mu_q = torch.cat(mu_heads, dim=-1)  # (B,D)
            std_q = torch.cat(std_heads, dim=-1)  # (B,D)
            outs_mu.append(mu_q);
            outs_std.append(std_q)
            all_w.append(torch.stack(w_heads, dim=1))  # (B,H,T)

            # 拼 Q 个查询： (B, 2*Q*D)
        out = torch.cat([torch.cat(outs_mu, dim=-1), torch.cat(outs_std, dim=-1)], dim=-1)

        weights = torch.stack(all_w, dim=1)
        return out, weights


class SER_From_Emb(nn.Module):
    def __init__(self, in_dim, num_labels, pooling="mq_mha",
                 attn_hidden=128, mq_q=2, mq_h=4, clf_hidden=None, dropout=0.1):
        super().__init__()
        self.pooling_name = pooling
        self.pooling = MQMHA_Pooling(in_dim, num_queries=mq_q, num_heads=mq_h, mlp_hidden=attn_hidden)
        feat_dim = 2 * in_dim * mq_q
        if clf_hidden:
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Dropout(dropout),
                nn.Linear(feat_dim, clf_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(clf_hidden, num_labels),
            )
        else:
            self.head = nn.Linear(feat_dim, num_labels)

    def forward(self, x, mask, return_weights=False):
        pooled, w = self.pooling(x, mask)  # ASP: w=(B,T)；MQMHA: w=(B,Q,H,T)
        logits = self.head(pooled)
        if return_weights:
            return logits, w
        return logits


def train(args):
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels = args.labels.split(",")
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2, ensure_ascii=False)
        # 估计 in_dim（读一条样本）
    import numpy as np, csv as _csv
    with open(args.train_csv) as f:
        r = _csv.DictReader(f);
        first = next(iter(r))
    in_dim = np.load(first["path"]).shape[1]
    train_ds = EmbCsvDataset(args.train_csv, label2id)
    val_ds = EmbCsvDataset(args.val_csv, label2id)
    collate = Collate()
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, num_workers=2, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False, num_workers=2, collate_fn=collate)
    model = SER_From_Emb(
        in_dim=in_dim, num_labels=len(labels),
        pooling=args.pooling, attn_hidden=args.attn_hidden,
        mq_q=args.mq_q, mq_h=args.mq_h, clf_hidden=args.clf_hidden
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    best_acc, best_path = -1.0, os.path.join(args.out_dir, "best.pt")

    for ep in range(1, args.epochs + 1):
        model.train();
        total = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)  # (B,T,D)
            m = batch["mask"].to(device)  # (B,T)
            y = batch["y"].to(device)
            logits = model(x, m)
            loss = nn.CrossEntropyLoss()(logits, y)
            opt.zero_grad();
            loss.backward();
            opt.step()
            total += loss.item()

        model.eval();
        preds = [];
        gts = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device);
                m = batch["mask"].to(device)
                y = batch["y"].to(device)
                p = model(x, m)
                preds += p.argmax(-1).cpu().tolist()
                gts += y.cpu().tolist()
        acc = accuracy_score(gts, preds)
        print(f"[Epoch {ep}] train_loss={total / len(train_loader):.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "in_dim": in_dim, "labels": id2label, "args": vars(args)},
                       best_path)
            print("  -> best saved:", best_path)
    print("Done. best val_acc=", best_acc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--labels", required=True,
                    help="逗号分隔，如: angry, disgusted, fearful, happy, other, sad, surprised")
    ap.add_argument("--out_dir", default="ser_from_emb_out")
    ap.add_argument("--pooling", default="mqmha", choices=["asp", "mqmha"])
    ap.add_argument("--attn_hidden", type=int, default=128)
    ap.add_argument("--mq_q", type=int, default=2)
    ap.add_argument("--mq_h", type=int, default=4)
    ap.add_argument("--clf_hidden", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    train(args)
