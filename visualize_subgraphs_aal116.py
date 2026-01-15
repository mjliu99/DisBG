# visualize_subgraphs_aal116.py
# ------------------------------------------------------------
# DisBG mask visualization for AAL116:
#   (0) Single-subject (as before, now with label in title/filename)
#   (1) PAIR: pick 1 Control + 1 ADHD and plot 2x2 (CTRL/ADHD × CAUSAL/BIAS)
#   (2) AVG : group-average adjacency within CTRL and ADHD (2x2)
#   (3) DIFF: group-difference maps (ADHD - CTRL), split into POS/NEG for both CAUSAL and BIAS (2x2)
#
# Additionally:
#   - Save "Top-50 edges" edge lists (ROI_u, ROI_v, weight, ROI labels) for:
#       * PAIR: CTRL/ADHD × CAUSAL/BIAS  (based on masks, top-50 by mask weight)
#       * AVG : CTRL/ADHD × CAUSAL/BIAS  (top-50 by mean adjacency)
#       * DIFF: CAUSAL_POS/CAUSAL_NEG/BIAS_POS/BIAS_NEG (top-50 by magnitude in each map)
#
# Example:
#   python visualize_subgraphs_aal116.py --split test --pair --avg --diff --topk 50 --max_batches 999999 --save_npz
#
# Notes:
# - Your data.y is graph-level labels of shape (B,), already verified.
# - Default label mapping: CTRL=0, ADHD=1 (change via --label_ctrl/--label_adhd).
# - AVG is computed by averaging per-subject TOP-K adjacency (matches your visualization style).
# - DIFF is computed on those averages, then split into POS/NEG to preserve direction.
# ------------------------------------------------------------

import os
import argparse
import numpy as np
import torch
from nilearn import datasets, plotting

from src.dataset_loader import create_dataloader
from nets.disbg import DisBGModel

try:
    from src.utils import set_seed
except Exception:
    def set_seed(seed: int):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# -------------------------
# AAL helpers
# -------------------------
def get_aal116_coords():
    aal = datasets.fetch_atlas_aal(version="SPM12")
    coords = plotting.find_parcellation_cut_coords(labels_img=aal.maps)
    if coords.shape != (116, 3):
        raise RuntimeError(f"Expected (116,3), got {coords.shape}")
    return coords, aal.labels  # labels[0] is Background


# -------------------------
# CKPT helpers (infer dims)
# -------------------------
def load_state_dict_from_ckpt(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError("Unknown checkpoint format.")


def infer_encoder_dims_from_state(state_dict: dict):
    """
    This project ckpt uses encoder_c / encoder_b (not encoder).
    Also, conv weights are stored as [in_dim, out_dim] in this codebase ckpt.
    We'll infer dims from encoder_c.* keys.
    """
    conv_keys = []
    for k in state_dict.keys():
        if k.startswith("encoder_c.convs.") and k.endswith(".weight"):
            parts = k.split(".")
            if len(parts) >= 4 and parts[2].isdigit():
                conv_keys.append((int(parts[2]), k))

    if not conv_keys:
        enc_like = [k for k in state_dict.keys() if "encoder" in k.lower()]
        raise RuntimeError(
            "Cannot infer encoder dims: no keys like encoder_c.convs.X.weight in checkpoint.\n"
            "Example encoder-like keys:\n" + "\n".join(enc_like[:50])
        )

    conv_keys.sort(key=lambda x: x[0])
    k0 = conv_keys[0][1]
    kL = conv_keys[-1][1]
    w0 = state_dict[k0]
    wL = state_dict[kL]

    # bias length corresponds to out_dim of that layer
    b0 = state_dict.get(k0.replace(".weight", ".bias"), None)
    bL = state_dict.get(kL.replace(".weight", ".bias"), None)

    # In this ckpt, weight is [in, out] (confirmed by your mismatch earlier)
    num_feats  = int(w0.shape[0])
    hidden_dim = int(b0.shape[0]) if b0 is not None else int(w0.shape[1])
    out_dim    = int(bL.shape[0]) if bL is not None else int(wL.shape[1])
    num_layers = int(conv_keys[-1][0] + 1)

    return num_feats, hidden_dim, out_dim, num_layers



# -------------------------
# Graph helpers
# -------------------------
def topk_adjacency(edge_index, edge_weight, num_nodes=116, topk=50):
    """
    Build dense symmetric adjacency from top-k edges by weight.
    edge_index: (2,E)
    edge_weight: (E,)
    """
    E = int(edge_weight.numel())
    if E == 0:
        return np.zeros((num_nodes, num_nodes), dtype=np.float32)

    k = min(int(topk), E)
    idx = torch.topk(edge_weight, k=k, largest=True).indices

    ei = edge_index[:, idx].numpy()
    ew = edge_weight[idx].numpy()

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for (u, v), w in zip(ei.T, ew):
        u = int(u)
        v = int(v)
        if u == v:
            continue
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        w = float(w)
        if w > A[u, v]:
            A[u, v] = w
        if w > A[v, u]:
            A[v, u] = w
    return A


def edge_set(edge_index):
    ei = edge_index.numpy()
    s = set()
    for u, v in ei.T:
        u = int(u)
        v = int(v)
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        s.add((a, b))
    return s


def print_top_hubs(deg, aal_labels, name, topn=10):
    topn = int(topn)
    order = np.argsort(-deg)[:topn]
    print(f"\n[HUB-{name}] Top-{topn} weighted-degree ROIs:")
    for r, roi0 in enumerate(order, 1):
        lab = aal_labels[roi0 + 1] if (roi0 + 1) < len(aal_labels) else "UNK"
        print(f"  #{r:02d} ROI {roi0+1:03d} | degree={deg[roi0]:.4f} | {lab}")


def extract_graph_edges_and_masks(edge_index_cpu, batch_cpu, m_c_cpu, m_b_cpu, g: int):
    """
    Return (ei_g, mc_g, mb_g) for graph g in current batch.
    edge_index_cpu: (2,E_total) on CPU, with block-diagonal node ids
    batch_cpu: (N_total,) node->graph mapping
    m_c_cpu, m_b_cpu: (E_total,) masks aligned with edge_index_cpu
    """
    src = edge_index_cpu[0]
    edge_batch = batch_cpu[src]
    idx_g = (edge_batch == g)

    ei_g = edge_index_cpu[:, idx_g]
    mc_g = m_c_cpu[idx_g]
    mb_g = m_b_cpu[idx_g]
    if ei_g.numel() == 0:
        return None, None, None

    # restore node ids to 0..115
    offset = int(ei_g.min().item())
    ei_g = ei_g - offset
    return ei_g, mc_g, mb_g


# -------------------------
# Edge list saving (Top-50) for medical analysis
# -------------------------
def save_top_edges_from_edgeindex(ei_g, w_g, aal_labels, out_path, topk=50, name="MASK"):
    """
    Save top-k edges directly from (edge_index, weight) (used for single subject / PAIR masks).
    """
    E = int(w_g.numel())
    k = min(int(topk), E)
    idx = torch.topk(w_g, k=k, largest=True).indices
    ei = ei_g[:, idx].numpy()
    ww = w_g[idx].numpy()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("rank,u_roi,u_label,v_roi,v_label,weight\n")
        for r, ((u, v), w) in enumerate(zip(ei.T, ww), 1):
            u = int(u); v = int(v)
            if u == v:
                continue
            u_lab = aal_labels[u + 1] if (u + 1) < len(aal_labels) else "UNK"
            v_lab = aal_labels[v + 1] if (v + 1) < len(aal_labels) else "UNK"
            f.write(f"{r},{u+1},{u_lab},{v+1},{v_lab},{float(w):.8f}\n")
    print(f"[OK] Saved top-{k} edges (edge_index): {out_path}")


def save_top_edges_from_matrix(A, aal_labels, out_path, topk=50, mode="largest"):
    """
    Save top-k edges from a dense symmetric matrix A (used for AVG/DIFF outputs).

    mode:
      - "largest": pick top by A[u,v] (assumes nonnegative)
      - "abs":     pick top by abs(A[u,v]) but store signed weight
    """
    A = np.asarray(A)
    assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be square"
    N = A.shape[0]

    # upper triangle indices
    iu, iv = np.triu_indices(N, k=1)
    vals = A[iu, iv]

    if mode == "abs":
        score = np.abs(vals)
    else:
        score = vals

    # filter out zeros to avoid dumping empty edges
    keep = score > 0
    iu, iv, vals, score = iu[keep], iv[keep], vals[keep], score[keep]

    if vals.size == 0:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("rank,u_roi,u_label,v_roi,v_label,weight\n")
        print(f"[WARN] Matrix has no positive entries. Saved empty edge list: {out_path}")
        return

    k = min(int(topk), vals.size)
    order = np.argsort(-score)[:k]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("rank,u_roi,u_label,v_roi,v_label,weight\n")
        for r, idx in enumerate(order, 1):
            u = int(iu[idx]); v = int(iv[idx])
            u_lab = aal_labels[u + 1] if (u + 1) < len(aal_labels) else "UNK"
            v_lab = aal_labels[v + 1] if (v + 1) < len(aal_labels) else "UNK"
            f.write(f"{r},{u+1},{u_lab},{v+1},{v_lab},{float(vals[idx]):.8f}\n")

    print(f"[OK] Saved top-{k} edges (matrix, mode={mode}): {out_path}")


# -------------------------
# Plot helpers
# -------------------------
def plot_single_connectome(A, coords, title, out_png):
    disp = plotting.plot_connectome(A, coords, title=title, node_size=10, edge_threshold="99%")
    disp.savefig(out_png, dpi=200)
    disp.close()
    print(f"[OK] Saved: {out_png}")


def plot_four_panels(A11, A12, A21, A22, coords, out_png, titles):
    """
    2x2 panel plotting using nilearn's plot_connectome with matplotlib axes.
    titles: (t11, t12, t21, t22)
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    plotting.plot_connectome(A11, coords, title=titles[0], axes=ax1, node_size=10, edge_threshold="99%")
    ax2 = fig.add_subplot(2, 2, 2)
    plotting.plot_connectome(A12, coords, title=titles[1], axes=ax2, node_size=10, edge_threshold="99%")
    ax3 = fig.add_subplot(2, 2, 3)
    plotting.plot_connectome(A21, coords, title=titles[2], axes=ax3, node_size=10, edge_threshold="99%")
    ax4 = fig.add_subplot(2, 2, 4)
    plotting.plot_connectome(A22, coords, title=titles[3], axes=ax4, node_size=10, edge_threshold="99%")

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")


@torch.no_grad()
def main():
    # --------------------------
    # Args
    # --------------------------
    p = argparse.ArgumentParser()
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--subject_id_in_batch", type=int, default=0)
    p.add_argument("--outdir", type=str, default="vis_subgraphs")
    p.add_argument("--ckpt", type=str, default="", help="checkpoint path (.pt). default: ./checkpoints/best.pt")
    p.add_argument("--hub_topn", type=int, default=10)
    p.add_argument("--use_abs", action="store_true",
                   help="Use abs(mask) for topk/degree if masks can be negative.")
    p.add_argument("--dataset", type=str, default="ADHD", choices=["ADHD", "ABIDE"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    # Multi-view outputs
    p.add_argument("--pair", action="store_true",
                   help="PAIR: plot one ADHD vs one Control (2x2).")
    p.add_argument("--avg", action="store_true",
                   help="AVG : plot group-mean CTRL vs ADHD (2x2).")
    p.add_argument("--diff", action="store_true",
                   help="DIFF: plot group-difference (ADHD-CTRL) with POS/NEG split (2x2 for causal and bias).")
    p.add_argument("--max_batches", type=int, default=999999,
                   help="How many batches to scan for pair/avg/diff. Use smaller for quick debug.")
    p.add_argument("--label_adhd", type=int, default=1,
                   help="Which label value indicates ADHD (default=1).")
    p.add_argument("--label_ctrl", type=int, default=0,
                   help="Which label value indicates Control (default=0).")

    # Save artifacts for medical analysis
    p.add_argument("--save_npz", action="store_true",
                   help="Save AVG and DIFF matrices to npz.")
    p.add_argument("--save_top_edges", action="store_true",
                   help="Save Top-50 edge lists to CSV for PAIR/AVG/DIFF.")
    p.add_argument("--edges_topk", type=int, default=50,
                   help="Top-K edges to export (default=50).")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(int(args.seed))

    coords, aal_labels = get_aal116_coords()

    # --------------------------
    # Find ckpt and infer dims
    # --------------------------
    ckpt_path = args.ckpt.strip()
    if not ckpt_path:
        candidate = os.path.join(os.getcwd(), "checkpoints", "best.pt")
        if os.path.exists(candidate):
            ckpt_path = candidate
    if not ckpt_path or (not os.path.exists(ckpt_path)):
        raise FileNotFoundError("Checkpoint not found. Provide --ckpt or put ./checkpoints/best.pt")

    state = load_state_dict_from_ckpt(ckpt_path)
    num_feats, hidden_dim, out_dim, num_layers = infer_encoder_dims_from_state(state)
    print(f"[CKPT] inferred num_feats={num_feats}, gnn_hidden_dim={hidden_dim}, "
          f"gnn_out_dim={out_dim}, num_gnn_layers={num_layers}")

    # --------------------------
    # Build minimal args for DisBGModel
    # --------------------------
    class A:
        pass

    train_args = A()
    train_args.dataset = args.dataset
    train_args.seed = args.seed
    train_args.batch_size = args.batch_size

    train_args.num_feats = num_feats
    train_args.num_classes = 2
    train_args.num_sex_classes = 2
    train_args.num_age_classes = 4

    train_args.gnn_hidden_dim = hidden_dim
    train_args.gnn_out_dim = out_dim
    train_args.num_gnn_layers = num_layers
    train_args.dropout = 0.1

    # mask-related
    train_args.mask_temperature = 1.0
    train_args.mask_hidden_dim = hidden_dim
    train_args.bias_scale = 0.5
    train_args.use_mask_node_proj = False
    train_args.mask_node_proj_dim = hidden_dim

    # --------------------------
    # Load data
    # --------------------------
    train_loader, val_loader, test_loader = create_dataloader(train_args.dataset, batch_size=train_args.batch_size)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    # --------------------------
    # Build model and load ckpt
    # --------------------------
    model = DisBGModel(train_args).to(device)
    model.eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[OK] Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")

    # --------------------------
    # (0) Single-subject output (kept)
    # --------------------------
    data = next(iter(loader)).to(device)
    x, edge_index, batch = data.x, data.edge_index, data.batch
    _ = model(x, edge_index, batch)

    m_c = getattr(getattr(model, "mask_gen_c", None), "last_mask", None)
    m_b = getattr(getattr(model, "mask_gen_b", None), "last_mask", None)
    if m_c is None or m_b is None:
        raise RuntimeError("Cannot read model.mask_gen_*.last_mask. Check your mask_generator.")

    mc_full = m_c.view(-1)
    mb_full = m_b.view(-1)
    if args.use_abs:
        mc_full = mc_full.abs()
        mb_full = mb_full.abs()

    ei = edge_index.detach().cpu()
    mc = mc_full.detach().cpu()
    mb = mb_full.detach().cpu()
    batch_cpu = batch.detach().cpu()

    B = int(batch_cpu.max().item()) + 1
    g = int(args.subject_id_in_batch)
    if g < 0 or g >= B:
        raise ValueError(f"subject_id_in_batch={g} out of range. This batch has B={B} graphs.")

    if (not hasattr(data, "y")) or data.y is None or data.y.dim() != 1 or data.y.numel() != B:
        raise RuntimeError(f"Expected graph-level data.y shape (B,), got {None if (not hasattr(data,'y')) else tuple(data.y.shape)}")

    y_g = int(data.y[g].item())

    src = ei[0]
    edge_batch = batch_cpu[src]
    idx_g = (edge_batch == g)

    ei_g = ei[:, idx_g]
    mc_g = mc[idx_g]
    mb_g = mb[idx_g]
    if ei_g.numel() == 0:
        raise RuntimeError("E_g=0, cannot visualize.")

    # restore node ids to 0..115
    offset = int(ei_g.min().item())
    ei_g = ei_g - offset

    print(f"[INFO] Single subject: split={args.split} g={g} y={y_g} E_g={mc_g.numel()}")

    A_c = topk_adjacency(ei_g, mc_g, num_nodes=116, topk=args.topk)
    A_b = topk_adjacency(ei_g, mb_g, num_nodes=116, topk=args.topk)

    out_c = os.path.join(args.outdir, f"{train_args.dataset}_{args.split}_SINGLE_g{g}_y{y_g}_causal_top{args.topk}.png")
    out_b = os.path.join(args.outdir, f"{train_args.dataset}_{args.split}_SINGLE_g{g}_y{y_g}_bias_top{args.topk}.png")
    plot_single_connectome(A_c, coords, f"{train_args.dataset} {args.split} SINGLE g={g} y={y_g} CAUSAL top{args.topk}", out_c)
    plot_single_connectome(A_b, coords, f"{train_args.dataset} {args.split} SINGLE g={g} y={y_g} BIAS   top{args.topk}", out_b)

    if args.save_top_edges:
        edges_dir = os.path.join(args.outdir, "edge_lists_top50")
        save_top_edges_from_edgeindex(
            ei_g, mc_g, aal_labels,
            os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_SINGLE_g{g}_y{y_g}_CAUSAL_top{args.edges_topk}.csv"),
            topk=args.edges_topk, name="SINGLE_CAUSAL"
        )
        save_top_edges_from_edgeindex(
            ei_g, mb_g, aal_labels,
            os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_SINGLE_g{g}_y{y_g}_BIAS_top{args.edges_topk}.csv"),
            topk=args.edges_topk, name="SINGLE_BIAS"
        )

    # --------------------------
    # Shared scanning helper for PAIR/AVG/DIFF
    # --------------------------
    def scan_masks_over_loader(for_avg: bool):
        """
        Iterate loader, run model, and yield per-batch CPU tensors:
          ei_cpu, batch_cpu, mc_cpu, mb_cpu, y_cpu (graph-level), B
        """
        scanned = 0
        for dataX in loader:
            scanned += 1
            if scanned > args.max_batches:
                break
            dataX = dataX.to(device)
            _ = model(dataX.x, dataX.edge_index, dataX.batch)
            mcX = model.mask_gen_c.last_mask.view(-1)
            mbX = model.mask_gen_b.last_mask.view(-1)
            if args.use_abs:
                mcX = mcX.abs()
                mbX = mbX.abs()

            ei_cpuX = dataX.edge_index.detach().cpu()
            batch_cpuX = dataX.batch.detach().cpu()
            mc_cpuX = mcX.detach().cpu()
            mb_cpuX = mbX.detach().cpu()

            if dataX.y is None or dataX.y.dim() != 1:
                raise RuntimeError(f"Expected graph-level labels data.y (B,), got {None if dataX.y is None else tuple(dataX.y.shape)}")
            y_cpuX = dataX.y.detach().cpu()

            BX = int(batch_cpuX.max().item()) + 1
            if y_cpuX.numel() != BX:
                raise RuntimeError(f"data.y numel={y_cpuX.numel()} mismatch B={BX}")
            yield scanned, ei_cpuX, batch_cpuX, mc_cpuX, mb_cpuX, y_cpuX, BX

    # --------------------------
    # (1) PAIR: one CTRL + one ADHD (2x2) + export top-50 edges
    # --------------------------
    if args.pair:
        print("\n[PAIR] Searching one CTRL and one ADHD sample...")
        found_ctrl = None  # (ei_g, mc_g, mb_g, y, batch_idx, g_in_batch)
        found_adhd = None

        for scanned, ei_cpu2, batch_cpu2, mc_cpu2, mb_cpu2, y_cpu2, B2 in scan_masks_over_loader(for_avg=False):
            for gg in range(B2):
                ygg = int(y_cpu2[gg].item())

                if (found_ctrl is None) and (ygg == args.label_ctrl):
                    ei_g2, mc_g2, mb_g2 = extract_graph_edges_and_masks(ei_cpu2, batch_cpu2, mc_cpu2, mb_cpu2, gg)
                    if ei_g2 is not None:
                        found_ctrl = (ei_g2, mc_g2, mb_g2, ygg, scanned, gg)

                if (found_adhd is None) and (ygg == args.label_adhd):
                    ei_g2, mc_g2, mb_g2 = extract_graph_edges_and_masks(ei_cpu2, batch_cpu2, mc_cpu2, mb_cpu2, gg)
                    if ei_g2 is not None:
                        found_adhd = (ei_g2, mc_g2, mb_g2, ygg, scanned, gg)

                if found_ctrl is not None and found_adhd is not None:
                    break
            if found_ctrl is not None and found_adhd is not None:
                break

        if found_ctrl is None or found_adhd is None:
            raise RuntimeError(f"[PAIR] Not found both classes within max_batches={args.max_batches}. "
                               f"Try increasing --max_batches or check label mapping.")

        ei_ctrl, mc_ctrl, mb_ctrl, y_ctrl, b_idx_ctrl, g_ctrl = found_ctrl
        ei_adhd, mc_adhd, mb_adhd, y_adhd, b_idx_adhd, g_adhd = found_adhd
        print(f"[PAIR] CTRL found at batch#{b_idx_ctrl} g={g_ctrl} y={y_ctrl}; "
              f"ADHD found at batch#{b_idx_adhd} g={g_adhd} y={y_adhd}")

        Ac_ctrl = topk_adjacency(ei_ctrl, mc_ctrl, num_nodes=116, topk=args.topk)
        Ab_ctrl = topk_adjacency(ei_ctrl, mb_ctrl, num_nodes=116, topk=args.topk)
        Ac_adhd = topk_adjacency(ei_adhd, mc_adhd, num_nodes=116, topk=args.topk)
        Ab_adhd = topk_adjacency(ei_adhd, mb_adhd, num_nodes=116, topk=args.topk)

        out_pair = os.path.join(args.outdir, f"{train_args.dataset}_{args.split}_PAIR_top{args.topk}.png")
        plot_four_panels(
            Ac_ctrl, Ab_ctrl, Ac_adhd, Ab_adhd, coords, out_pair,
            titles=(
                f"{train_args.dataset} {args.split} PAIR CTRL CAUSAL top{args.topk}",
                f"{train_args.dataset} {args.split} PAIR CTRL BIAS   top{args.topk}",
                f"{train_args.dataset} {args.split} PAIR ADHD CAUSAL top{args.topk}",
                f"{train_args.dataset} {args.split} PAIR ADHD BIAS   top{args.topk}",
            )
        )

        if args.save_top_edges:
            edges_dir = os.path.join(args.outdir, "edge_lists_top50")
            save_top_edges_from_edgeindex(
                ei_ctrl, mc_ctrl, aal_labels,
                os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_PAIR_CTRL_CAUSAL_top{args.edges_topk}.csv"),
                topk=args.edges_topk, name="PAIR_CTRL_CAUSAL"
            )
            save_top_edges_from_edgeindex(
                ei_ctrl, mb_ctrl, aal_labels,
                os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_PAIR_CTRL_BIAS_top{args.edges_topk}.csv"),
                topk=args.edges_topk, name="PAIR_CTRL_BIAS"
            )
            save_top_edges_from_edgeindex(
                ei_adhd, mc_adhd, aal_labels,
                os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_PAIR_ADHD_CAUSAL_top{args.edges_topk}.csv"),
                topk=args.edges_topk, name="PAIR_ADHD_CAUSAL"
            )
            save_top_edges_from_edgeindex(
                ei_adhd, mb_adhd, aal_labels,
                os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_PAIR_ADHD_BIAS_top{args.edges_topk}.csv"),
                topk=args.edges_topk, name="PAIR_ADHD_BIAS"
            )

    # --------------------------
    # (2) AVG: group mean CTRL vs ADHD (2x2) + export top-50 edges
    # --------------------------
    Ac_ctrl_mean = Ab_ctrl_mean = Ac_adhd_mean = Ab_adhd_mean = None
    n_ctrl = n_adhd = 0

    if args.avg or args.diff:
        # DIFF depends on AVG, so compute AVG when either flag is set.
        print("\n[AVG] Computing group-average adjacency (CTRL vs ADHD)...")
        sum_Ac_ctrl = np.zeros((116, 116), dtype=np.float64)
        sum_Ab_ctrl = np.zeros((116, 116), dtype=np.float64)
        sum_Ac_adhd = np.zeros((116, 116), dtype=np.float64)
        sum_Ab_adhd = np.zeros((116, 116), dtype=np.float64)
        n_ctrl = 0
        n_adhd = 0

        for scanned, ei_cpu3, batch_cpu3, mc_cpu3, mb_cpu3, y_cpu3, B3 in scan_masks_over_loader(for_avg=True):
            for gg in range(B3):
                ygg = int(y_cpu3[gg].item())
                ei_g3, mc_g3, mb_g3 = extract_graph_edges_and_masks(ei_cpu3, batch_cpu3, mc_cpu3, mb_cpu3, gg)
                if ei_g3 is None:
                    continue

                # average TOP-K adjacency (matches your visual style)
                Ac = topk_adjacency(ei_g3, mc_g3, num_nodes=116, topk=args.topk)
                Ab = topk_adjacency(ei_g3, mb_g3, num_nodes=116, topk=args.topk)

                if ygg == args.label_ctrl:
                    sum_Ac_ctrl += Ac
                    sum_Ab_ctrl += Ab
                    n_ctrl += 1
                elif ygg == args.label_adhd:
                    sum_Ac_adhd += Ac
                    sum_Ab_adhd += Ab
                    n_adhd += 1

        if n_ctrl == 0 or n_adhd == 0:
            raise RuntimeError(f"[AVG] Got n_ctrl={n_ctrl}, n_adhd={n_adhd}. "
                               f"Increase --max_batches or check label mapping.")

        Ac_ctrl_mean = (sum_Ac_ctrl / n_ctrl).astype(np.float32)
        Ab_ctrl_mean = (sum_Ab_ctrl / n_ctrl).astype(np.float32)
        Ac_adhd_mean = (sum_Ac_adhd / n_adhd).astype(np.float32)
        Ab_adhd_mean = (sum_Ab_adhd / n_adhd).astype(np.float32)

        print(f"[AVG] n_ctrl={n_ctrl}, n_adhd={n_adhd}")

        if args.avg:
            out_avg = os.path.join(args.outdir, f"{train_args.dataset}_{args.split}_AVG_top{args.topk}_nC{n_ctrl}_nA{n_adhd}.png")
            plot_four_panels(
                Ac_ctrl_mean, Ab_ctrl_mean, Ac_adhd_mean, Ab_adhd_mean, coords, out_avg,
                titles=(
                    f"{train_args.dataset} {args.split} AVG CTRL CAUSAL top{args.topk} (n={n_ctrl})",
                    f"{train_args.dataset} {args.split} AVG CTRL BIAS   top{args.topk} (n={n_ctrl})",
                    f"{train_args.dataset} {args.split} AVG ADHD CAUSAL top{args.topk} (n={n_adhd})",
                    f"{train_args.dataset} {args.split} AVG ADHD BIAS   top{args.topk} (n={n_adhd})",
                )
            )

            if args.save_top_edges:
                edges_dir = os.path.join(args.outdir, "edge_lists_top50")
                save_top_edges_from_matrix(
                    Ac_ctrl_mean, aal_labels,
                    os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_AVG_CTRL_CAUSAL_top{args.edges_topk}.csv"),
                    topk=args.edges_topk, mode="largest"
                )
                save_top_edges_from_matrix(
                    Ab_ctrl_mean, aal_labels,
                    os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_AVG_CTRL_BIAS_top{args.edges_topk}.csv"),
                    topk=args.edges_topk, mode="largest"
                )
                save_top_edges_from_matrix(
                    Ac_adhd_mean, aal_labels,
                    os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_AVG_ADHD_CAUSAL_top{args.edges_topk}.csv"),
                    topk=args.edges_topk, mode="largest"
                )
                save_top_edges_from_matrix(
                    Ab_adhd_mean, aal_labels,
                    os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_AVG_ADHD_BIAS_top{args.edges_topk}.csv"),
                    topk=args.edges_topk, mode="largest"
                )

        if args.save_npz:
            out_npz = os.path.join(args.outdir, f"{train_args.dataset}_{args.split}_AVG_top{args.topk}_nC{n_ctrl}_nA{n_adhd}.npz")
            np.savez(
                out_npz,
                Ac_ctrl=Ac_ctrl_mean, Ab_ctrl=Ab_ctrl_mean,
                Ac_adhd=Ac_adhd_mean, Ab_adhd=Ab_adhd_mean,
                n_ctrl=n_ctrl, n_adhd=n_adhd
            )
            print(f"[OK] Saved AVG npz: {out_npz}")

    # --------------------------
    # (3) DIFF: (ADHD - CTRL) with POS/NEG split
    #     - CAUSAL_DIFF_POS: ADHD > CTRL
    #     - CAUSAL_DIFF_NEG: CTRL > ADHD  (stored as positive magnitude)
    #     similarly for BIAS
    # --------------------------
    if args.diff:
        if Ac_ctrl_mean is None:
            raise RuntimeError("DIFF requested but AVG means were not computed.")

        print("\n[DIFF] Computing group-difference maps: ADHD - CTRL (POS/NEG split)...")

        Dc = (Ac_adhd_mean - Ac_ctrl_mean).astype(np.float32)
        Db = (Ab_adhd_mean - Ab_ctrl_mean).astype(np.float32)

        Dc_pos = np.clip(Dc, 0, None)
        Dc_neg = np.clip(-Dc, 0, None)  # magnitude of CTRL>ADHD
        Db_pos = np.clip(Db, 0, None)
        Db_neg = np.clip(-Db, 0, None)

        # # Plot CAUSAL diff (pos/neg) and BIAS diff (pos/neg) as two separate 2x2 figures
        # out_diff_c = os.path.join(args.outdir, f"{train_args.dataset}_{args.split}_DIFF_CAUSAL_top{args.topk}_nC{n_ctrl}_nA{n_adhd}.png")
        # plot_four_panels(
        #     Dc_pos, Dc_neg, np.zeros_like(Dc_pos), np.zeros_like(Dc_pos), coords, out_diff_c,
        #     titles=(
        #         f"{train_args.dataset} {args.split} DIFF CAUSAL POS (ADHD>CTRL)",
        #         f"{train_args.dataset} {args.split} DIFF CAUSAL NEG (CTRL>ADHD)",
        #         " ",
        #         " ",
        #     )
        # )

        # out_diff_b = os.path.join(args.outdir, f"{train_args.dataset}_{args.split}_DIFF_BIAS_top{args.topk}_nC{n_ctrl}_nA{n_adhd}.png")
        # plot_four_panels(
        #     Db_pos, Db_neg, np.zeros_like(Db_pos), np.zeros_like(Db_pos), coords, out_diff_b,
        #     titles=(
        #         f"{train_args.dataset} {args.split} DIFF BIAS POS (ADHD>CTRL)",
        #         f"{train_args.dataset} {args.split} DIFF BIAS NEG (CTRL>ADHD)",
        #         " ",
        #         " ",
        #     )
        # )

        # ---- NEW: one combined DIFF figure (2x2, same layout style as AVG panel) ----
        out_diff_all = os.path.join(
            args.outdir,
            f"{train_args.dataset}_{args.split}_DIFF_top{args.topk}_nC{n_ctrl}_nA{n_adhd}.png"
        )

        plot_four_panels(
            Dc_pos, Dc_neg, Db_pos, Db_neg, coords, out_diff_all,
            titles=(
                f"{train_args.dataset} {args.split} DIFF CAUSAL POS (ADHD>CTRL)",
                f"{train_args.dataset} {args.split} DIFF CAUSAL NEG (CTRL>ADHD)",
                f"{train_args.dataset} {args.split} DIFF BIAS   POS (ADHD>CTRL)",
                f"{train_args.dataset} {args.split} DIFF BIAS   NEG (CTRL>ADHD)",
            )
        )

        print(f"[OK] Saved: {out_diff_all}")
        
        if args.save_top_edges:
            edges_dir = os.path.join(args.outdir, "edge_lists_top50")
            # For POS/NEG maps, "largest" is appropriate because values are already nonnegative magnitudes.
            save_top_edges_from_matrix(
                Dc_pos, aal_labels,
                os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_DIFF_CAUSAL_POS_top{args.edges_topk}.csv"),
                topk=args.edges_topk, mode="largest"
            )
            save_top_edges_from_matrix(
                Dc_neg, aal_labels,
                os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_DIFF_CAUSAL_NEG_top{args.edges_topk}.csv"),
                topk=args.edges_topk, mode="largest"
            )
            save_top_edges_from_matrix(
                Db_pos, aal_labels,
                os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_DIFF_BIAS_POS_top{args.edges_topk}.csv"),
                topk=args.edges_topk, mode="largest"
            )
            save_top_edges_from_matrix(
                Db_neg, aal_labels,
                os.path.join(edges_dir, f"{train_args.dataset}_{args.split}_DIFF_BIAS_NEG_top{args.edges_topk}.csv"),
                topk=args.edges_topk, mode="largest"
            )

        if args.save_npz:
            out_npz = os.path.join(args.outdir, f"{train_args.dataset}_{args.split}_DIFF_top{args.topk}_nC{n_ctrl}_nA{n_adhd}.npz")
            np.savez(
                out_npz,
                Dc=Dc, Db=Db,
                Dc_pos=Dc_pos, Dc_neg=Dc_neg,
                Db_pos=Db_pos, Db_neg=Db_neg,
                n_ctrl=n_ctrl, n_adhd=n_adhd
            )
            print(f"[OK] Saved DIFF npz: {out_npz}")

    print("\n[DONE] Outputs saved under:", os.path.abspath(args.outdir))
    if args.save_top_edges:
        print("[DONE] Edge lists (Top-K) saved under:", os.path.join(os.path.abspath(args.outdir), "edge_lists_top50"))


if __name__ == "__main__":
    main()
