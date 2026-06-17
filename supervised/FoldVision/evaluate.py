import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, matthews_corrcoef, roc_auc_score

from foldvision import FoldVisionEncoder
from foldvision.finetune import FoldVisionPredictor
from foldvision.dataloader import loader_function, get_labels_dict, get_full_bounding_boxes


@torch.no_grad()
def run_eval_with_augmentations(
    model,
    loader,
    device: str,
    n_preds: int = 10,
    base_seed: int = 123,
):
    """
    Returns:
      names: list[str] length N
      preds_all: np.ndarray shape (n_preds, N)  (raw predictions per augmentation run)
      labels: np.ndarray shape (N,) or None
    """
    model.eval()

    names_ref = None
    labels_ref = None

    preds_all = []

    for r in range(n_preds):
        # Ensure each run has different random augmentations but reproducible overall
        np.random.seed(base_seed + r)
        torch.manual_seed(base_seed + r)

        run_preds = []
        run_names = []
        run_labels = []

        pbar = tqdm(loader, desc=f"Eval aug run {r+1}/{n_preds}", leave=False)
        for X, y, names in pbar:
            X = X.to(device, non_blocking=True)

            pred = model(X)  # (B,)
            pred = pred.detach().cpu().numpy().astype(np.float32)

            run_preds.extend(pred.tolist())
            run_names.extend(list(names))

            if y is not None:
                run_labels.extend(y.detach().cpu().numpy().astype(np.float32).tolist())

        if names_ref is None:
            names_ref = run_names
            labels_ref = np.array(run_labels, dtype=np.float32) if len(run_labels) > 0 else None
        else:
            # Sanity: same order every run
            if run_names != names_ref:
                raise RuntimeError(
                    "Order of proteins changed between augmentation runs. "
                    "Make sure evaluation loader uses shuffle=False and deterministic sampling."
                )

        preds_all.append(np.array(run_preds, dtype=np.float32))

    preds_all = np.stack(preds_all, axis=0)  # (n_preds, N)
    return names_ref, preds_all, labels_ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--csv", required=True, help="CSV with columns: Protein ID, label (label optional)")
    p.add_argument("--task", choices=["regression", "binary"], default="regression")
    p.add_argument("--task_name", required=True)
    p.add_argument("--checkpoint", required=True, help="Path to best_*.pt checkpoint")
    p.add_argument("--encoder_id", default="AlexanderKroll/foldvision-encoder")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--n_preds", type=int, default=10, help="Number of augmented predictions per protein")
    p.add_argument("--base_seed", type=int, default=123)
    p.add_argument("--out_dir", default="eval_outputs")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Task:", args.task)
    print("n_preds:", args.n_preds)

    df = pd.read_csv(args.csv)
    has_labels = "label" in df.columns

    boxes_path = os.path.join(args.data_path, "bounding_boxes.npy")
    bounding_boxes = np.load(boxes_path, allow_pickle=True).item()

    # Use helper when labels are present; otherwise keep label-free mode working.
    if has_labels:
        # We pass val_df as df and train_df as empty-ish.
        empty_train = df.iloc[0:0].copy()
        labels_dict, _, eval_names = get_labels_dict(empty_train, df, bounding_boxes)
    else:
        labels_dict = None
        boxes_keys = set(bounding_boxes.keys())
        eval_names = [
            str(pid) for pid in df["Protein ID"].astype(str).tolist()
            if str(pid) + ".npz" in boxes_keys
        ]

    print(f"[INFO] Usable proteins for eval: {len(eval_names)}")
    if len(eval_names) == 0:
        raise RuntimeError("No proteins from csv found in bounding_boxes.npy")

    split_files_with_boxes = get_full_bounding_boxes(bounding_boxes, eval_names)

    # shuffle=False so order stays stable, augment=True because we want random rotations at eval
    eval_loader = loader_function(
        data_path=args.data_path,
        filenames=eval_names,
        split_files_with_boxes=split_files_with_boxes,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        n_gpus=1,
        rank=0,
        seed=args.base_seed,
        labels_dict=labels_dict,
        augment=True, 
    )

    encoder = FoldVisionEncoder.from_pretrained(args.encoder_id)
    model = FoldVisionPredictor(encoder=encoder, hidden_dim=256, output_dim=1).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    print("[INFO] Loaded checkpoint:", args.checkpoint)

    # Run augmented evaluation
    names, preds_all, labels = run_eval_with_augmentations(
        model=model,
        loader=eval_loader,
        device=device,
        n_preds=args.n_preds,
        base_seed=args.base_seed,
    )

    pred_mean = preds_all.mean(axis=0)  # (N,)
    pred_std = preds_all.std(axis=0)    # (N,)
    pred_proba_mean = None
    if args.task == "binary":
        pred_proba_mean = 1.0 / (1.0 + np.exp(-pred_mean))

    # Save outputs
    out_npz = os.path.join(args.out_dir, f"preds_{args.task_name}_{args.task}_n{args.n_preds}.npz")
    np.savez_compressed(
        out_npz,
        protein_ids=np.array(names),
        pred_mean=pred_mean.astype(np.float32),
        pred_std=pred_std.astype(np.float32),
        pred_proba_mean=(pred_proba_mean.astype(np.float32) if pred_proba_mean is not None else None),
        preds_all=preds_all.astype(np.float32),
        labels=(labels.astype(np.float32) if labels is not None else None),
    )
    print("[SAVE]", out_npz)

    # Print simple metrics if labels exist
    if labels is not None:
        y_true = labels.astype(np.float64)
        y_pred = pred_mean.astype(np.float64)  # logits for binary, raw regression prediction for regression

        if args.task == "regression":
            # Spearman / Pearson
            sp = spearmanr(y_true, y_pred).correlation
            pr = pearsonr(y_true, y_pred)[0]

            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))

            # Optional: still useful
            mse = float(np.mean((y_pred - y_true) ** 2))
            rmse = float(np.sqrt(mse))

            print(f"[METRIC] Spearman r : {sp:.6f}")
            print(f"[METRIC] Pearson r  : {pr:.6f}")
            print(f"[METRIC] MAE        : {mae:.6f}")
            print(f"[METRIC] R2         : {r2:.6f}")
            print(f"[METRIC] RMSE       : {rmse:.6f}")

        else:
            # Binary classification: y_pred are logits
            probs = pred_proba_mean.astype(np.float64)
            y_hat = (probs >= 0.5).astype(int)
            y_true_int = y_true.astype(int)

            acc = float(accuracy_score(y_true_int, y_hat))
            mcc = float(matthews_corrcoef(y_true_int, y_hat))

            # ROC-AUC requires both classes present
            if len(np.unique(y_true_int)) < 2:
                roc_auc = None
                print("[WARN] ROC-AUC undefined: only one class present in y_true.")
            else:
                roc_auc = float(roc_auc_score(y_true_int, probs))

            print(f"[METRIC] Accuracy   : {acc:.6f}")
            print(f"[METRIC] MCC        : {mcc:.6f}")
            if roc_auc is not None:
                print(f"[METRIC] ROC-AUC    : {roc_auc:.6f}")


    # Show a few example rows
    print("\nExamples:")
    for i in range(min(5, len(names))):
        if args.task == "binary":
            print(f"  {names[i]}  logit_mean={pred_mean[i]:.4f}  prob_mean={pred_proba_mean[i]:.4f}  logit_std={pred_std[i]:.4f}")
        else:
            print(f"  {names[i]}  mean={pred_mean[i]:.4f}  std={pred_std[i]:.4f}")


if __name__ == "__main__":
    main()
