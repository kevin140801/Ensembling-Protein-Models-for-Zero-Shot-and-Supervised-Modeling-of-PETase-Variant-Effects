import os
from os.path import join
import numpy as np
from tqdm import tqdm

# Import YOUR existing preprocessing helpers (no behavior change)
from foldvision.preprocess.pdb_to_points import (
    pdb_to_point_dict,
    point_dict_to_npy,
    bounding_boxes as DEFAULT_BOUNDING_BOXES,
    dic as DEFAULT_DIC,
    resolution as DEFAULT_RESOLUTION,
    sigma_sqd as DEFAULT_SIGMA_SQD,
)


def preprocess_pdb_directory(
    pdb_dir: str,
    out_dir: str,
    continue_run: bool = True,
    save_every: int = 25,
    max_input_size: int = 160,
    resolution: float = DEFAULT_RESOLUTION,
    sigma_sqd: float = DEFAULT_SIGMA_SQD,
    dic: dict = None,
    bounding_boxes=None,
):
    """
    Preprocess all PDBs in `pdb_dir` into compressed NPZ point lists and a bounding_boxes.npy dict.

    Outputs:
      out_dir/
        bounding_boxes.npy
        numpy_3D_point_lists/*.npz  (each contains key: point_list)

    Returns:
      boxes_dict (dict): maps "<protein>.npz" -> "[X Y Z]" string
      boxes_path (str): path to bounding_boxes.npy
    """
    os.makedirs(out_dir, exist_ok=True)
    npz_dir = join(out_dir, "numpy_3D_point_lists")
    os.makedirs(npz_dir, exist_ok=True)

    if dic is None:
        dic = DEFAULT_DIC
    if bounding_boxes is None:
        bounding_boxes = DEFAULT_BOUNDING_BOXES

    boxes_path = join(out_dir, "bounding_boxes.npy")

    # collect PDB ids
    pdbs = [f for f in os.listdir(pdb_dir) if f.endswith(".pdb")]
    pdbs.sort()
    pdb_ids = [f.replace(".pdb", "") for f in pdbs]

    # continue logic (robust):
    # redo a protein if:
    #   - its npz file is missing, OR
    #   - its key is missing from boxes_dict
    if continue_run:
        if os.path.exists(boxes_path):
            boxes_dict = np.load(boxes_path, allow_pickle=True).item()
        else:
            boxes_dict = {}

        def needs_redo(pid: str) -> bool:
            key = pid + ".npz"
            npz_path = join(npz_dir, key)
            return (not os.path.exists(npz_path)) or (key not in boxes_dict)

        todo = [pid for pid in pdb_ids if needs_redo(pid)]

        if len(todo) == 0:
            print("All PDBs already processed")
            return boxes_dict, boxes_path

        print(f"Continuing: processing {len(todo)} / {len(pdb_ids)} PDBs (others already done).")
        pdb_ids = todo
    else:
        boxes_dict = {}

    # main loop
    for k, pid in tqdm(list(enumerate(pdb_ids)), total=len(pdb_ids), desc="Preprocessing PDBs"):
        name = pid + ".npz"
        try:
            point_dict = pdb_to_point_dict(
                filename=join(pdb_dir, pid + ".pdb"),
                max_input_size=max_input_size,
                resolution=resolution,
                sigma_sqd=sigma_sqd,
                dic=dic,
            )
            point_list = point_dict_to_npy(point_dict)
            point_list[:, 4] = np.round(point_list[:, 4], 4)

            # storage-efficient saving
            np.savez_compressed(join(npz_dir, name), point_list=point_list)

            # bounding box selection (same as your logic)
            indices = point_list[:, :3].astype(int)
            r = [np.max(indices[:, 0]), np.max(indices[:, 1]), np.max(indices[:, 2])]
            for box in bounding_boxes:
                if r[0] < box[0] and r[1] < box[1] and r[2] < box[2]:
                    boxes_dict[name] = str(box)
                    break

            # periodic save so interruptions don't lose progress
            if save_every is not None and save_every > 0:
                if (k + 1) % int(save_every) == 0:
                    np.save(boxes_path, boxes_dict)

        except Exception as e:
            print(f"Error with {pid}: {e}")

    np.save(boxes_path, boxes_dict)
    print(f"Processed {len(pdb_ids)} PDBs")
    print("Saved:", boxes_path)
    return boxes_dict, boxes_path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--pdb_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--continue_run", action="store_true", help="Skip already done proteins (robust).")
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--max_input_size", type=int, default=160)
    args = p.parse_args()

    preprocess_pdb_directory(
        pdb_dir=args.pdb_dir,
        out_dir=args.out_dir,
        continue_run=args.continue_run,
        save_every=args.save_every,
        max_input_size=args.max_input_size,
    )
