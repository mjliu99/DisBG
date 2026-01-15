# src/dataset_loader.py
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch_geometric.utils import dense_to_sparse, remove_self_loops
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, StratifiedKFold

try:
    from torch_geometric.loader import DataLoader  # PyG >= 2.0
except Exception:
    from torch_geometric.data import DataLoader    # PyG < 2.0

# =========================
# Path
# =========================
DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets"


# =========================
# Utils
# =========================
def load_csv(file_path, header=None):
    return pd.read_csv(file_path, sep=",", header=header, engine="python")


# =========================
# Sparse graph construction (TOP-K)
# =========================
def corr_to_edge_index_topk(corr, k=10, abs_val=True):
    """
    corr: numpy [N, N]
    k:    number of neighbors per node
    """
    A = torch.from_numpy(corr).float()
    N = A.size(0)

    # remove self-loops
    A.fill_diagonal_(0)

    score = A.abs() if abs_val else A
    mask = torch.zeros_like(A)

    # NOTE: deterministic given A (no randomness here)
    for i in range(N):
        idx = torch.topk(score[i], k=k).indices
        mask[i, idx] = 1.0

    # symmetrize
    mask = torch.maximum(mask, mask.T)
    A = A * mask

    edge_index, edge_weight = dense_to_sparse(A)
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
    return edge_index, edge_weight


# =========================
# Load raw dataset
# =========================
def load_dataset(dataset_name, use_cache=True, topk=10):
    cache_path = DATASET_PATH / f"{dataset_name}_cached_topk{topk}.pt"

    if use_cache and cache_path.exists():
        print(f"[CACHE] Loading cached {dataset_name} (topk={topk})")
        return torch.load(cache_path)

    print(f"[LOAD] Loading raw {dataset_name} dataset...")

    if dataset_name == "ADHD":
        dataset_folder = DATASET_PATH / "ADHD200_packed"
        meta_df = load_csv(DATASET_PATH / "ADHD.csv", header=0).set_index("ScanDir ID")
        sex_key, age_key = "Gender", "Age"
    elif dataset_name == "ABIDE":
        dataset_folder = DATASET_PATH / "ABIDE_packed"
        meta_df = load_csv(DATASET_PATH / "ABIDE.csv", header=0).set_index("SUB_ID")
        sex_key, age_key = "SEX", "AGE_AT_SCAN"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    x, y, y_a, y_s = [], [], [], []
    edge_index, edge_weight = [], []
    timeseries = []

    for folder in sorted(dataset_folder.iterdir()):
        if not folder.is_dir():
            continue

        sid = int(folder.name)

        corr = load_csv(folder / "corr.csv").to_numpy()
        ts = load_csv(folder / "timeseries.csv").to_numpy()
        label = load_csv(folder / "label.txt").to_numpy().reshape(-1)

        # ---- labels ----
        if dataset_name == "ADHD":
            label = 1 if label > 0 else 0
            sex = meta_df.loc[sid][sex_key]
            if pd.isna(sex):
                continue
            sex = int(sex)
        else:
            label = int(label) - 1
            sex = int(meta_df.loc[sid][sex_key]) - 1

        age = meta_df.loc[sid][age_key]

        ei, ew = corr_to_edge_index_topk(corr, k=topk)

        x.append(corr)
        y.append(int(label))
        y_s.append(int(sex))
        y_a.append(float(age))
        edge_index.append(ei)
        edge_weight.append(ew)
        timeseries.append(ts)

    # ---- age binning (4 classes) ----
    y_a = np.asarray(y_a, dtype=np.float32)
    bins = np.percentile(y_a, [25, 50, 75])
    y_a = np.digitize(y_a, bins).astype(np.int64)

    data = dict(
        x=x,
        y=y,
        y_a=y_a.tolist(),
        y_s=y_s,
        edge_index=edge_index,
        edge_weight=edge_weight,
        timeseries=timeseries,
    )

    if use_cache:
        torch.save(data, cache_path)

    return data


# =========================
# Build PyG Data objects
# =========================
def build_pyg_data(data_dict):
    return [
        Data(
            x=torch.from_numpy(xi).float(),
            edge_index=ei,
            edge_attr=ew,
            y=torch.tensor(yi, dtype=torch.long),
            y_a=torch.tensor(yai, dtype=torch.long),
            y_s=torch.tensor(ysi, dtype=torch.long),
            timeseries=torch.from_numpy(ti).float(),
        )
        for xi, yi, yai, ysi, ei, ew, ti in zip(
            data_dict["x"],
            data_dict["y"],
            data_dict["y_a"],
            data_dict["y_s"],
            data_dict["edge_index"],
            data_dict["edge_weight"],
            data_dict["timeseries"],
        )
    ]


# =========================
# Create DataLoader (STABLE)
# =========================
def create_dataloader(
    dataset_name: str,
    batch_size: int = 8,
    train_seed: int = 0,   # controls training-time randomness (e.g., shuffle order)
    split_seed: int = 0,   # controls data split only (MUST be fixed across folds)
    use_cache: bool = True,
    topk: int = 10,
    use_5fold: bool = False,
    fold_id: int = 0,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    """
    Stability rules:
    - split_seed is used ONLY for data splitting (single split or CV).
      For a proper 5-fold CV, split_seed must be constant across folds.
    - train_seed is used for DataLoader shuffle generator (train loader).
    - When num_workers>0, worker_init_fn seeds NumPy and Torch per worker.
    """

    data = load_dataset(dataset_name, use_cache=use_cache, topk=topk)
    labels = np.array(data["y"], dtype=np.int64)

    if not use_5fold:
        # ===== Single split (70/20/10) =====
        idx = np.arange(len(labels))
        idx_tr, idx_tmp = train_test_split(
            idx, test_size=0.3, random_state=split_seed, stratify=labels
        )
        idx_dev, idx_te = train_test_split(
            idx_tmp,
            test_size=1 / 3,
            random_state=split_seed,
            stratify=labels[idx_tmp],
        )
    else:
        # ===== 5-fold stratified CV (MUST depend on split_seed only) =====
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
        splits = list(skf.split(np.zeros(len(labels)), labels))
        idx_tr, idx_te = splits[int(fold_id)]

        # dev split from train (also split_seed only)
        idx_tr, idx_dev = train_test_split(
            idx_tr,
            test_size=0.1,
            random_state=split_seed,
            stratify=labels[idx_tr],
        )

    def select(idx):
        # idx could be np array
        idx = list(map(int, idx))
        return {k: [v[i] for i in idx] for k, v in data.items()}

    train_data = build_pyg_data(select(idx_tr))
    dev_data   = build_pyg_data(select(idx_dev))
    test_data  = build_pyg_data(select(idx_te))

    print(f"[DATA] Train/Dev/Test = {len(train_data)}/{len(dev_data)}/{len(test_data)}")
    print(f"[SEED] split_seed={int(split_seed)} train_seed={int(train_seed)} fold_id={int(fold_id)}")

    # ---- stable shuffle (train loader only) ----
    g = torch.Generator()
    g.manual_seed(int(train_seed))

    def _worker_init_fn(worker_id: int):
        # Make each worker deterministic
        base = int(train_seed) + int(worker_id)
        np.random.seed(base)
        torch.manual_seed(base)

    common = dict(
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory) if torch.cuda.is_available() else False,
        worker_init_fn=_worker_init_fn if int(num_workers) > 0 else None,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=int(batch_size),
        shuffle=True,
        generator=g,
        **common,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=int(batch_size),
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=int(batch_size),
        shuffle=False,
        **common,
    )

    return train_loader, dev_loader, test_loader
