"""GPU jobs on Modal: embeddings for the kNN baseline and ModernBERT fine-tuning.

One-time setup (from the project folder):
    pip install modal
    modal volume create k8s-router
    modal volume put k8s-router data/processed/dataset.jsonl /dataset.jsonl
Run:
    modal run modal_app.py::embed            # -> /emb/<model>.npy on the volume
    modal run --detach modal_app.py::train   # -> /runs/<name>/{val,test}_logits.npy, classes.json
Fetch results:
    modal volume get k8s-router /runs/modernbert-base data/runs/ --force
"""
import json
import os
import time

import modal

app = modal.App("k8s-issue-router")
vol = modal.Volume.from_name("k8s-router", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.5.1", "transformers==4.48.3", "accelerate==1.3.0",
                      "sentence-transformers==3.4.1", "numpy<2.2", "scikit-learn==1.6.1"))
MNT = "/vol"


def load_rows():
    return [json.loads(l) for l in open(f"{MNT}/dataset.jsonl")]


@app.function(image=image, gpu="T4", volumes={MNT: vol}, timeout=1800)
def embed(model_name: str = "BAAI/bge-base-en-v1.5"):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    rows = load_rows()
    m = SentenceTransformer(model_name, device="cuda")
    m.max_seq_length = 512
    t = time.time()
    E = m.encode([r["text"] for r in rows], batch_size=64, normalize_embeddings=True,
                 show_progress_bar=False, convert_to_numpy=True)
    os.makedirs(f"{MNT}/emb", exist_ok=True)
    np.save(f"{MNT}/emb/{model_name.replace('/', '__')}.npy", E.astype("float32"))
    vol.commit()
    print(E.shape, f"{time.time() - t:.0f}s")


@app.function(image=image, gpu="L4", volumes={MNT: vol}, timeout=3 * 3600)
def train(model_name: str = "answerdotai/ModernBERT-base", run: str = "modernbert-base",
          max_len: int = 512, epochs: int = 4, lr: float = 3e-5, batch: int = 16, seed: int = 0, save_model: bool = True):
    """Multi-label classifier over SIGs. Trains on `train`, keeps the epoch with the best
    val top-1, writes raw logits for val and test (calibration and thresholds happen offline)."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    rows = load_rows()
    classes = sorted({s for r in rows if r["gold"] for s in r["gold"]})
    idx = {s: i for i, s in enumerate(classes)}
    by_split = {sp: [r for r in rows if r["split"] == sp] for sp in ("train", "val", "test")}
    train_rows = [r for r in by_split["train"] if r["gold"]]

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(classes), problem_type="multi_label_classification").cuda()

    def enc(rs):
        return tok([r["text"] for r in rs], truncation=True, max_length=max_len, padding=True,
                   return_tensors="pt")

    def target(rs):
        y = torch.zeros(len(rs), len(classes))
        for i, r in enumerate(rs):
            for s in r["gold"]:
                y[i, idx[s]] = 1
        return y

    # class imbalance: weight positives by sqrt(neg/pos), capped
    Y = target(train_rows)
    pos = Y.sum(0).clamp(min=1)
    pos_weight = ((len(Y) - pos) / pos).sqrt().clamp(max=10).cuda()
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loader = DataLoader(list(range(len(train_rows))), batch_size=batch, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    @torch.no_grad()
    def logits_for(rs):
        model.eval()
        out = []
        for k in range(0, len(rs), 64):
            b = {kk: v.cuda() for kk, v in enc(rs[k:k + 64]).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(**b).logits.float().cpu())
        model.train()
        return torch.cat(out).numpy()

    def val_top1(L):
        lab = [(i, r) for i, r in enumerate(by_split["val"]) if r["gold"]]
        return np.mean([classes[L[i].argmax()] in r["gold"] for i, r in lab])

    best, best_logits = -1, None
    t0 = time.time()
    for ep in range(epochs):
        for step, ids in enumerate(loader):
            rs = [train_rows[i] for i in ids.tolist()]
            b = {k: v.cuda() for k, v in enc(rs).items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**b).logits
            loss = loss_fn(logits.float(), target(rs).cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{len(loader)} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        Lv = logits_for(by_split["val"])
        acc = val_top1(Lv)
        print(f"epoch {ep}: val top1 {acc:.3f}", flush=True)
        if acc > best:
            best, best_logits = acc, (Lv, logits_for(by_split["test"]))
            if save_model:  # keep the weights that produced the reported logits
                os.makedirs(f"{MNT}/runs/{run}/model", exist_ok=True)
                model.save_pretrained(f"{MNT}/runs/{run}/model")
                tok.save_pretrained(f"{MNT}/runs/{run}/model")

    out = f"{MNT}/runs/{run}"
    os.makedirs(out, exist_ok=True)

    np.save(f"{out}/val_logits.npy", best_logits[0])
    np.save(f"{out}/test_logits.npy", best_logits[1])
    json.dump({"classes": classes,
               "val_numbers": [r["number"] for r in by_split["val"]],
               "test_numbers": [r["number"] for r in by_split["test"]],
               "best_val_top1": best, "train_seconds": time.time() - t0,
               "config": dict(model=model_name, max_len=max_len, epochs=epochs, lr=lr, batch=batch, seed=seed)},
              open(f"{out}/meta.json", "w"), indent=1)
    vol.commit()
    print("best val top1", best)


@app.local_entrypoint()
def seeds():
    """modal run --detach modal_app.py::seeds  -> 3 runs in parallel to measure seed variance."""
    calls = [train.spawn(run=f"modernbert-s{s}", seed=s) for s in (0, 1, 2)]
    for c in calls:
        c.get()
