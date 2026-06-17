import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from foldvision import FoldVisionEncoder
from foldvision.finetune import FoldVisionPredictor
from foldvision.dataloader import loader_function, get_labels_dict, get_full_bounding_boxes


def assert_binary_labels(df: pd.DataFrame, keep_ids, label_col: str = "label"):
    # Only check labels for proteins that will actually be used
    sub = df[df["Protein ID"].astype(str).isin(keep_ids)]
    vals = pd.unique(sub[label_col].dropna())
    bad = [v for v in vals if v not in (0, 1)]
    if bad:
        raise ValueError(f"Binary task requires labels in {{0,1}}. Found invalid values: {bad}")

    # Optional: small warning if split is degenerate
    if len(vals) == 1:
        print(f"[WARN] Only one unique label in this split: {vals[0]} (training/metrics may be degenerate).")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True, help="Folder containing bounding_boxes.npy and numpy_3D_point_lists/")
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", default=None)
    p.add_argument("--task", choices=["regression", "binary"], default="regression")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument("--encoder_id", default="AlexanderKroll/foldvision-encoder")
    p.add_argument("--task_name", required=True, help="Name for this prediction task (used in checkpoint filenames)")
    p.add_argument("--amp", action="store_true", help="Enable AMP (autocast + GradScaler) on CUDA")
    p.add_argument("--resume", default=None, help="Path to checkpoint (.pt) to resume training from")
    p.add_argument("--pos_class_weight", type=float, default=None,
               help="(binary only) Positive class weight for BCEWithLogitsLoss. Example: 5.0")

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(args.amp and device == "cuda")

    print("============================================================")
    print("FoldVision finetune")
    print(f" Device        : {device}")
    print(f" AMP enabled   : {use_amp}")
    print(f" Task          : {args.task}")
    print(f" Task name     : {args.task_name}")
    print(f" Epochs        : {args.epochs}")
    print(f" Batch size    : {args.batch_size}")
    print(f" LR            : {args.lr}")
    print(f" Train CSV     : {args.train_csv}")
    print(f" Val CSV       : {args.val_csv}")
    print(f" Data path     : {args.data_path}")
    print(f" Output dir    : {args.out_dir}")
    print("============================================================")

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv) if args.val_csv is not None else None

    print(f"[INFO] Train rows: {len(train_df)}")
    if val_df is not None:
        print(f"[INFO] Val rows  : {len(val_df)}")

    boxes_path = os.path.join(args.data_path, "bounding_boxes.npy")
    if not os.path.exists(boxes_path):
        raise FileNotFoundError(f"Could not find bounding_boxes.npy at: {boxes_path}")
    bounding_boxes = np.load(boxes_path, allow_pickle=True).item()
    print(f"[INFO] Loaded bounding_boxes entries: {len(bounding_boxes)}")

    labels_dict, train_names, val_names = get_labels_dict(train_df, val_df, bounding_boxes)

    print(f"[INFO] Usable train proteins (exist in bounding_boxes): {len(train_names)}")
    print(f"[INFO] Usable val proteins   (exist in bounding_boxes): {len(val_names)}")

    if len(train_names) == 0:
        raise RuntimeError("No training proteins found that exist in bounding_boxes.npy")

    if args.task == "binary":
        assert_binary_labels(train_df, set(train_names))
        if val_df is not None and len(val_names) > 0:
            assert_binary_labels(val_df, set(val_names))

        tr_labels = [labels_dict[p] for p in train_names if p in labels_dict]
        if len(tr_labels) > 0:
            tr_labels = np.array(tr_labels)
            print(f"[INFO] Train label balance: mean={tr_labels.mean():.4f} (fraction of 1s if labels are 0/1)")

    split_files_with_boxes = get_full_bounding_boxes(bounding_boxes, train_names + val_names)

    print("[INFO] Building loaders...")
    train_loader = loader_function(
        data_path=args.data_path,
        filenames=train_names,
        split_files_with_boxes=split_files_with_boxes,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        n_gpus=1,
        rank=0,
        seed=args.seed,
        labels_dict=labels_dict,
        augment=True, 
    )

    val_loader = None
    if len(val_names) > 0:
        val_loader = loader_function(
            data_path=args.data_path,
            filenames=val_names,
            split_files_with_boxes=split_files_with_boxes,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            n_gpus=1,
            rank=0,
            seed=args.seed,
            labels_dict=labels_dict,
            augment=False,  # val aug OFF
        )

    # --- model ---
    print(f"[INFO] Loading encoder from: {args.encoder_id}")
    encoder = FoldVisionEncoder.from_pretrained(args.encoder_id)
    model = FoldVisionPredictor(encoder=encoder, hidden_dim=256, output_dim=1).to(device)


    if args.task == "regression":
        criterion = nn.MSELoss()
    else:
        if args.pos_class_weight is not None:
            pos_weight = torch.tensor([args.pos_class_weight]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            print(f"[INFO] Using BCEWithLogitsLoss(pos_weight={args.pos_class_weight})")
        else:
            criterion = nn.BCEWithLogitsLoss()


    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    best_epoch = None


    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {args.resume}")

        print(f"[RESUME] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")

        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])

        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)


    best_path = os.path.join(args.out_dir, f"best_{args.task_name}_{args.task}.pt")
    last_path = os.path.join(args.out_dir, f"last_{args.task_name}_{args.task}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for X, y, _ in pbar:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    pred = model(X)  # (B,)
                    loss = criterion(pred, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(X)
                loss = criterion(pred, y)
                loss.backward()
                opt.step()

            losses.append(loss.item())
            pbar.set_postfix(loss= float(np.mean(losses)))

        train_loss = float(np.mean(losses)) if len(losses) > 0 else float("nan")

        # Validation
        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            vloss = []
            correct = 0
            total = 0

            vbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False)
            with torch.no_grad():
                for X, y, _ in vbar:
                    X = X.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True).float()

                    if use_amp:
                        with torch.amp.autocast(device_type="cuda"):
                            pred = model(X)
                            loss = criterion(pred, y)
                    else:
                        pred = model(X)
                        loss = criterion(pred, y)

                    vloss.append(loss.item())
                    vbar.set_postfix(loss=float(loss.item()))

                    if args.task == "binary":
                        probs = torch.sigmoid(pred)
                        preds01 = (probs >= 0.5).float()
                        correct += (preds01 == y).sum().item()
                        total += y.numel()

            val_loss = float(np.mean(vloss)) if len(vloss) > 0 else float("nan")
            if args.task == "binary" and total > 0:
                val_acc = correct / total

        msg = f"Epoch {epoch:03d} | train loss {train_loss:.4f}"
        if val_loss is not None:
            msg += f" | val loss {val_loss:.4f}"
            if val_acc is not None:
                msg += f" | val acc {val_acc:.4f}"
        print(msg)

        # Save ONLY if val improves (or if no val split: save last each epoch OR just final once)
        if val_loss is not None:
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch

                ckpt = {
                    "epoch": epoch,
                    "task": args.task,
                    "task_name": args.task_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
                torch.save(ckpt, best_path)
                print(f"[SAVE] Improved val -> saved: {best_path} (best epoch = {best_epoch}, best val = {best_val:.4f})")
        else:
            # No validation: save the last checkpoint each epoch (your previous behavior was saving always anyway)
            ckpt = {
                "epoch": epoch,
                "task": args.task,
                "task_name": args.task_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "train_loss": train_loss,
                "val_loss": None,
            }
            torch.save(ckpt, last_path)
            print(f"[SAVE] No val split -> saved: {last_path}")

    print("============================================================")
    print("[DONE]")
    if best_epoch is not None:
        print(f" Best epoch: {best_epoch} | best val: {best_val:.4f}")
        print(f" Best ckpt : {best_path}")
    else:
        print(f" Last ckpt : {last_path}")
    print("============================================================")


if __name__ == "__main__":
    main()
