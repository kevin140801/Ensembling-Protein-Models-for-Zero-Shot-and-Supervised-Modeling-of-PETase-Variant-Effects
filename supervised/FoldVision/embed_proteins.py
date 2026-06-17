# scripts/embed_proteins.py

import os
from os.path import join
import numpy as np
import torch
from tqdm import tqdm

from foldvision import FoldVisionEncoder
from foldvision.dataloader import loader_function, get_full_bounding_boxes


def _load_encoder_from_checkpoint(encoder: FoldVisionEncoder, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)
        return "encoder_state_dict"

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd_full = ckpt["model_state_dict"]
        candidate_prefixes = ["encoder.", "prot_encoder.", "model.encoder."]

        for pref in candidate_prefixes:
            enc_sd = {k[len(pref):]: v for k, v in sd_full.items() if k.startswith(pref)}
            if len(enc_sd) > 0:
                encoder.load_state_dict(enc_sd, strict=True)
                return f"model_state_dict (prefix='{pref}')"

        # fallback: try treating full dict as encoder state_dict
        encoder.load_state_dict(sd_full, strict=True)
        return "model_state_dict (treated as encoder sd)"

    if isinstance(ckpt, dict):
        encoder.load_state_dict(ckpt, strict=True)
        return "raw state_dict"

    raise RuntimeError(f"Unsupported checkpoint format at {ckpt_path} (type={type(ckpt)})")


@torch.no_grad()
def embed_pdb_directory(
    work_dir: str,
    out_dir: str,
    encoder_id: str = "AlexanderKroll/foldvision-encoder",
    checkpoint: str = None,
    n_runs: int = 10,
    base_seed: int = 123,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str = None,
    save_per_run: bool = True,
):
    """
    Generate FoldVision encoder embeddings for all *already-preprocessed* proteins.

    Requires:
      work_dir/bounding_boxes.npy
      work_dir/numpy_3D_point_lists/*.npz

    Uses your dataloader to enable batching and workers.

    Saves:
      - embeddings_mean.npz (always)
      - optionally embeddings_runXX.npz per run
    """
    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)

    boxes_path = join(work_dir, "bounding_boxes.npy")
    if not os.path.exists(boxes_path):
        raise FileNotFoundError(f"Missing {boxes_path}. Preprocess first and put results in work_dir.")

    bounding_boxes = np.load(boxes_path, allow_pickle=True).item()

    protein_ids = sorted([k.replace(".npz", "") for k in bounding_boxes.keys() if k.endswith(".npz")])
    if len(protein_ids) == 0:
        raise RuntimeError("No proteins found in bounding_boxes.npy")
    name_to_idx = {pid: i for i, pid in enumerate(protein_ids)}

    split_files_with_boxes = get_full_bounding_boxes(bounding_boxes, protein_ids)

    labels_dict = {pid: 0.0 for pid in protein_ids}


    encoder = FoldVisionEncoder.from_pretrained(encoder_id).to(device)
    ckpt_info = None
    if checkpoint is not None:
        ckpt_info = _load_encoder_from_checkpoint(encoder, checkpoint)
        print(f"[INFO] Loaded encoder weights from checkpoint: {checkpoint}")
        print(f"[INFO] Checkpoint interpretation: {ckpt_info}")
    encoder.eval()

    all_emb = np.zeros((n_runs, len(protein_ids), 1024), dtype=np.float32)
    saved = {}

    print("============================================================")
    print("FoldVision embedding generation")
    print(f" Device      : {device}")
    print(f" Encoder ID  : {encoder_id}")
    print(f" Checkpoint  : {checkpoint}")
    print(f" Proteins    : {len(protein_ids)}")
    print(f" Runs        : {n_runs}")
    print(f" Batch size  : {batch_size}")
    print(f" Workers     : {num_workers}")
    print("============================================================")

    for r in range(n_runs):
        # Make reproducible, but different across runs (controls your augment randomness)
        np.random.seed(base_seed + r)
        torch.manual_seed(base_seed + r)

        # Build loader for this run so worker RNG is re-seeded via 'seed' argument
        loader = loader_function(
            data_path=work_dir,
            filenames=protein_ids,
            split_files_with_boxes=split_files_with_boxes,
            batch_size=batch_size,
            shuffle=False,          
            num_workers=num_workers,
            n_gpus=1,
            rank=0,
            seed=base_seed + r,     
            labels_dict=labels_dict,
            augment=True,           
        )

        run_emb = np.zeros((len(protein_ids), 1024), dtype=np.float32)
        seen = np.zeros(len(protein_ids), dtype=bool)
        pbar = tqdm(loader, desc=f"Embedding run {r+1}/{n_runs}", total=len(loader))
        for X, _, names in pbar:
            X = X.to(device, non_blocking=True)

            z = encoder(X)  # (B,1024)
            z = z.detach().cpu().numpy().astype(np.float32)

            for j, name in enumerate(names):
                name = str(name)
                if name not in name_to_idx:
                    raise RuntimeError(f"Unexpected protein id from loader: {name}")
                i = name_to_idx[name]
                run_emb[i] = z[j]
                seen[i] = True

        if not np.all(seen):
            missing = [protein_ids[i] for i in np.where(~seen)[0].tolist()]
            raise RuntimeError(f"Missing embeddings for proteins: {missing}")

        all_emb[r] = run_emb


        if save_per_run:
            out_path = join(out_dir, f"embeddings_run{r:02d}.npz")
            np.savez_compressed(
                out_path,
                protein_ids=np.array(protein_ids, dtype=object),
                embeddings=run_emb,
                run=int(r),
                seed=int(base_seed + r),
                encoder_id=str(encoder_id),
                checkpoint=str(checkpoint) if checkpoint is not None else "",
                checkpoint_info=str(ckpt_info) if ckpt_info is not None else "",
            )
            saved[f"run_{r:02d}"] = out_path

    emb_mean = all_emb.mean(axis=0)
    emb_std = all_emb.std(axis=0)

    out_mean = join(out_dir, "embeddings_mean.npz")
    np.savez_compressed(
        out_mean,
        protein_ids=np.array(protein_ids, dtype=object),
        embeddings_mean=emb_mean,
        embeddings_std=emb_std,
        embeddings_all=all_emb,
        n_runs=int(n_runs),
        base_seed=int(base_seed),
        encoder_id=str(encoder_id),
        checkpoint=str(checkpoint) if checkpoint is not None else "",
        checkpoint_info=str(ckpt_info) if ckpt_info is not None else "",
    )
    saved["mean"] = out_mean

    print("Saved embeddings to:", out_dir)
    return saved


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True, help="Folder containing bounding_boxes.npy and numpy_3D_point_lists/")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--encoder_id", default="AlexanderKroll/foldvision-encoder")
    p.add_argument("--checkpoint", default=None, help="Optional finetuned checkpoint (.pt) to load encoder weights from")
    p.add_argument("--n_runs", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=123)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--no_save_per_run", action="store_true")
    args = p.parse_args()

    embed_pdb_directory(
        work_dir=args.work_dir,
        out_dir=args.out_dir,
        encoder_id=args.encoder_id,
        checkpoint=args.checkpoint,
        n_runs=args.n_runs,
        base_seed=args.base_seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        save_per_run=(not args.no_save_per_run),
    )
