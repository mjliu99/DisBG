import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

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
    # use encoder_c to infer (encoder_b shares the same dims)
    conv_keys = []
    for k in state_dict.keys():
        if k.startswith("encoder_c.convs.") and k.endswith(".weight"):
            parts = k.split(".")
            if len(parts) >= 4 and parts[2].isdigit():
                conv_keys.append((int(parts[2]), k))
    if not conv_keys:
        raise RuntimeError("Cannot infer encoder dims: no keys like encoder_c.convs.X.weight in checkpoint.")

    conv_keys.sort(key=lambda x: x[0])
    k0 = conv_keys[0][1]
    kL = conv_keys[-1][1]
    w0 = state_dict[k0]
    wL = state_dict[kL]

    # bias tells us out_dim
    b0_key = k0.replace(".weight", ".bias")
    b0 = state_dict.get(b0_key, None)

    if b0 is None:
        # fallback: assume [in, out] (matches your ckpt)
        num_feats  = int(w0.shape[0])
        hidden_dim = int(w0.shape[1])
    else:
        # if bias length matches w0 second dim => weight is [in, out]
        if int(b0.shape[0]) == int(w0.shape[1]):
            num_feats  = int(w0.shape[0])
            hidden_dim = int(w0.shape[1])
        else:
            # otherwise weight is [out, in]
            hidden_dim = int(w0.shape[0])
            num_feats  = int(w0.shape[1])

    # last layer out_dim similarly
    bL_key = kL.replace(".weight", ".bias")
    bL = state_dict.get(bL_key, None)
    if bL is not None:
        out_dim = int(bL.shape[0])
    else:
        # make best guess
        out_dim = int(wL.shape[1])  # for [in,out]
    num_layers = int(conv_keys[-1][0] + 1)

    return num_feats, hidden_dim, out_dim, num_layers


# -------------------------
# Edge weights + adjacency
# -------------------------
def get_base_edge_weight(data, edge_index, default_ones=True):
    """
    Try to fetch the "original adjacency weight" from data.
    Fallback: ones if not provided.

    Common candidates:
      - data.edge_weight: (E,)
      - data.edge_attr: (E,) or (E,1)
    """
    if hasattr(data, "edge_weight") and data.edge_weight is not None:
        w = data.edge_weight
        return w.view(-1)
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        ea = data.edge_attr
        if ea.ndim == 2 and ea.shape[1] == 1:
            return ea[:, 0].view(-1)
        return ea.view(-1)
    if default_ones:
        return torch.ones(edge_index.shape[1], device=edge_index.device)
    raise RuntimeError("No edge weights found in data.")


def slice_graph_edges(edge_index_cpu, batch_cpu, w_cpu, g: int):
    """
    Slice edges for graph g from a batched block-diagonal graph.
    Returns (ei_g, w_g) with node ids shifted to 0..N-1.
    """
    src = edge_index_cpu[0]
    edge_batch = batch_cpu[src]
    idx = (edge_batch == g)
    ei_g = edge_index_cpu[:, idx]
    w_g = w_cpu[idx]
    if ei_g.numel() == 0:
        return None, None
    offset = int(ei_g.min().item())
    ei_g = ei_g - offset
    return ei_g, w_g


def build_dense_A(ei_g, w_g, N=116, sym=True):
    """
    Build dense adjacency (N,N) from (edge_index, edge_weight).
    Uses max for duplicates; optionally symmetrize.
    """
    A = np.zeros((N, N), dtype=np.float32)
    ei = ei_g.numpy()
    ww = w_g.numpy()
    for (u, v), w in zip(ei.T, ww):
        u = int(u); v = int(v)
        if u == v:
            continue
        w = float(w)
        if w > A[u, v]:
            A[u, v] = w
        if sym and w > A[v, u]:
            A[v, u] = w
    return A


def topk_from_matrix(A, k=50):
    """
    Keep only top-k edges (by weight) in the upper triangle, then symmetrize.
    """
    N = A.shape[0]
    iu, iv = np.triu_indices(N, k=1)
    vals = A[iu, iv]
    order = np.argsort(-vals)
    order = order[:min(k, order.size)]
    B = np.zeros_like(A)
    for idx in order:
        u = int(iu[idx]); v = int(iv[idx])
        w = float(vals[idx])
        if w <= 0:
            continue
        B[u, v] = w
        B[v, u] = w
    return B


# -------------------------
# Plotting: matrices
# -------------------------
def save_matrix_png(A, out_png, title="", vmin=None, vmax=None):
    plt.figure(figsize=(7, 6))
    plt.imshow(A, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("ROI")
    plt.ylabel("ROI")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"[OK] Saved: {out_png}")


def save_triplet_png(A, Ac, Ab, out_png, titles):
    """
    1x3: A, A_causal, A_bias
    """
    fig = plt.figure(figsize=(18, 6))
    for i, (M, t) in enumerate(zip([A, Ac, Ab], titles), 1):
        ax = fig.add_subplot(1, 3, i)
        im = ax.imshow(M)
        ax.set_title(t)
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")


def save_diff_grid_png(D_A, D_Ac, D_Ab, out_png, prefix="DIFF (ADHD-CTRL)"):
    """
    3 rows × 2 cols: POS/NEG for each of A, A_c, A_b
      row1: A
      row2: A_c
      row3: A_b
    NEG is shown as magnitude (CTRL>ADHD).
    """
    Da_pos = np.clip(D_A, 0, None)
    Da_neg = np.clip(-D_A, 0, None)
    Dc_pos = np.clip(D_Ac, 0, None)
    Dc_neg = np.clip(-D_Ac, 0, None)
    Db_pos = np.clip(D_Ab, 0, None)
    Db_neg = np.clip(-D_Ab, 0, None)

    fig = plt.figure(figsize=(14, 16))
    mats = [
        (Da_pos, f"{prefix} | A POS (ADHD>CTRL)"),
        (Da_neg, f"{prefix} | A NEG (CTRL>ADHD)"),
        (Dc_pos, f"{prefix} | A_c POS (ADHD>CTRL)"),
        (Dc_neg, f"{prefix} | A_c NEG (CTRL>ADHD)"),
        (Db_pos, f"{prefix} | A_b POS (ADHD>CTRL)"),
        (Db_neg, f"{prefix} | A_b NEG (CTRL>ADHD)"),
    ]
    for i, (M, t) in enumerate(mats, 1):
        ax = fig.add_subplot(3, 2, i)
        im = ax.imshow(M)
        ax.set_title(t)
        ax.set_xlabel("ROI")
        ax.set_ylabel("ROI")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"[OK] Saved: {out_png}")


# -------------------------
# Edge export for medical analysis
# -------------------------
def save_top_edges_from_matrix(A, out_csv, topk=50, mode="largest"):
    """
    mode:
      - largest: pick top by A[u,v] (assumes nonnegative)
      - abs: pick top by abs(A[u,v]) but store signed weight
    """
    A = np.asarray(A)
    N = A.shape[0]
    iu, iv = np.triu_indices(N, k=1)
    vals = A[iu, iv]
    score = np.abs(vals) if mode == "abs" else vals
    order = np.argsort(-score)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("rank,u_roi,v_roi,weight\n")
        cnt = 0
        for idx in order:
            if cnt >= topk:
                break
            u = int(iu[idx]); v = int(iv[idx])
            w = float(vals[idx])
            if mode == "largest" and w <= 0:
                continue
            if mode == "abs" and abs(w) <= 0:
                continue
            cnt += 1
            f.write(f"{cnt},{u+1},{v+1},{w:.8f}\n")
    print(f"[OK] Saved Top-{topk} edges: {out_csv}")


# -------------------------
# Main
# -------------------------
@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ADHD", choices=["ADHD", "ABIDE"])
    p.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt", type=str, default="", help="checkpoint path (.pt). default: ./checkpoints/best.pt")
    p.add_argument("--outdir", type=str, default="vis_matrices")

    # labels
    p.add_argument("--label_ctrl", type=int, default=0)
    p.add_argument("--label_adhd", type=int, default=1)

    # views
    p.add_argument("--pair", action="store_true", help="Visualize one CTRL and one ADHD subject.")
    p.add_argument("--group", action="store_true", help="Visualize group mean and group difference.")
    p.add_argument("--max_batches", type=int, default=999999)

    # matrix build
    p.add_argument("--N", type=int, default=116)
    p.add_argument("--use_abs_mask", action="store_true")
    p.add_argument("--topk_matrix", type=int, default=0,
                   help="If >0, keep only top-k edges in each matrix before plotting/averaging.")
    p.add_argument("--export_topk_edges", type=int, default=50)

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    # ckpt
    ckpt_path = args.ckpt.strip()
    if not ckpt_path:
        candidate = os.path.join(os.getcwd(), "checkpoints", "best.pt")
        if os.path.exists(candidate):
            ckpt_path = candidate
    if not ckpt_path or (not os.path.exists(ckpt_path)):
        raise FileNotFoundError("Checkpoint not found. Provide --ckpt or put ./checkpoints/best.pt")

    state = load_state_dict_from_ckpt(ckpt_path)
    num_feats, hidden_dim, out_dim, num_layers = infer_encoder_dims_from_state(state)
    print(f"[CKPT] inferred num_feats={num_feats}, hidden={hidden_dim}, out={out_dim}, layers={num_layers}")

    # minimal args for model
    class A: pass
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
    train_args.mask_temperature = 1.0
    train_args.mask_hidden_dim = hidden_dim
    train_args.bias_scale = 0.5
    train_args.use_mask_node_proj = False
    train_args.mask_node_proj_dim = hidden_dim

    # data
    train_loader, val_loader, test_loader = create_dataloader(train_args.dataset, batch_size=train_args.batch_size)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]

    # model
    model = DisBGModel(train_args).to(device)
    model.eval()
    model.load_state_dict(state, strict=False)
    print(f"[OK] Loaded checkpoint: {ckpt_path}")

    def run_batch(data):
        data = data.to(device)
        _ = model(data.x, data.edge_index, data.batch)
        mc = model.mask_gen_c.last_mask.view(-1)
        mb = model.mask_gen_b.last_mask.view(-1)
        if args.use_abs_mask:
            mc = mc.abs(); mb = mb.abs()
        w0 = get_base_edge_weight(data, data.edge_index).view(-1)
        return (data, data.edge_index.detach().cpu(), data.batch.detach().cpu(),
                w0.detach().cpu(), mc.detach().cpu(), mb.detach().cpu())

    # --------------------------
    # (1) PAIR: one CTRL + one ADHD
    # --------------------------
    if args.pair:
        print("[PAIR] searching one CTRL and one ADHD...")
        found_ctrl = None
        found_adhd = None

        scanned = 0
        for data in loader:
            scanned += 1
            if scanned > args.max_batches:
                break
            data, ei_cpu, b_cpu, w0_cpu, mc_cpu, mb_cpu = run_batch(data)
            y = data.y.detach().cpu()
            B = int(b_cpu.max().item()) + 1
            if y.numel() != B:
                raise RuntimeError(f"Expect y shape (B,), got {tuple(y.shape)} with B={B}")

            for g in range(B):
                yg = int(y[g].item())
                ei_g, w_g = slice_graph_edges(ei_cpu, b_cpu, w0_cpu, g)
                if ei_g is None:
                    continue
                ei_g2, mc_g = slice_graph_edges(ei_cpu, b_cpu, mc_cpu, g)
                _, mb_g = slice_graph_edges(ei_cpu, b_cpu, mb_cpu, g)

                # matrices
                A  = build_dense_A(ei_g,  w_g,  N=args.N, sym=True)
                Ac = build_dense_A(ei_g2, mc_g, N=args.N, sym=True)
                Ab = build_dense_A(ei_g2, mb_g, N=args.N, sym=True)

                if args.topk_matrix > 0:
                    A  = topk_from_matrix(A,  args.topk_matrix)
                    Ac = topk_from_matrix(Ac, args.topk_matrix)
                    Ab = topk_from_matrix(Ab, args.topk_matrix)

                if (found_ctrl is None) and (yg == args.label_ctrl):
                    found_ctrl = (A, Ac, Ab, scanned, g)
                if (found_adhd is None) and (yg == args.label_adhd):
                    found_adhd = (A, Ac, Ab, scanned, g)

                if found_ctrl and found_adhd:
                    break
            if found_ctrl and found_adhd:
                break

        if not (found_ctrl and found_adhd):
            raise RuntimeError("PAIR not found both classes. Increase --max_batches or check label mapping.")

        # save figures
        edges_dir = os.path.join(args.outdir, "edge_lists_top50")
        A0, Ac0, Ab0, b0, g0 = found_ctrl
        A1, Ac1, Ab1, b1, g1 = found_adhd

        out0 = os.path.join(args.outdir, f"{args.dataset}_{args.split}_PAIR_CTRL_A_Ac_Ab.png")
        save_triplet_png(A0, Ac0, Ab0, out0,
                         [f"CTRL A (batch{b0},g{g0})", "CTRL A_causal (masked)", "CTRL A_bias (masked)"])

        out1 = os.path.join(args.outdir, f"{args.dataset}_{args.split}_PAIR_ADHD_A_Ac_Ab.png")
        save_triplet_png(A1, Ac1, Ab1, out1,
                         [f"ADHD A (batch{b1},g{g1})", "ADHD A_causal (masked)", "ADHD A_bias (masked)"])

        # export top-50 edges for each matrix
        k = args.export_topk_edges
        save_top_edges_from_matrix(A0,  os.path.join(edges_dir, f"{args.dataset}_{args.split}_PAIR_CTRL_A_top{k}.csv"),  topk=k, mode="largest")
        save_top_edges_from_matrix(Ac0, os.path.join(edges_dir, f"{args.dataset}_{args.split}_PAIR_CTRL_Ac_top{k}.csv"), topk=k, mode="largest")
        save_top_edges_from_matrix(Ab0, os.path.join(edges_dir, f"{args.dataset}_{args.split}_PAIR_CTRL_Ab_top{k}.csv"), topk=k, mode="largest")

        save_top_edges_from_matrix(A1,  os.path.join(edges_dir, f"{args.dataset}_{args.split}_PAIR_ADHD_A_top{k}.csv"),  topk=k, mode="largest")
        save_top_edges_from_matrix(Ac1, os.path.join(edges_dir, f"{args.dataset}_{args.split}_PAIR_ADHD_Ac_top{k}.csv"), topk=k, mode="largest")
        save_top_edges_from_matrix(Ab1, os.path.join(edges_dir, f"{args.dataset}_{args.split}_PAIR_ADHD_Ab_top{k}.csv"), topk=k, mode="largest")

    # --------------------------
    # (2) GROUP: mean CTRL/ADHD + DIFF for A, Ac, Ab
    # --------------------------
    if args.group:
        print("[GROUP] computing mean and diff...")
        sum_Ac = np.zeros((args.N, args.N), dtype=np.float64)
        sum_Ab = np.zeros((args.N, args.N), dtype=np.float64)
        sum_A  = np.zeros((args.N, args.N), dtype=np.float64)
        sum_Ac2 = np.zeros((args.N, args.N), dtype=np.float64)
        sum_Ab2 = np.zeros((args.N, args.N), dtype=np.float64)
        sum_A2  = np.zeros((args.N, args.N), dtype=np.float64)
        n_ctrl = 0
        n_adhd = 0

        scanned = 0
        for data in loader:
            scanned += 1
            if scanned > args.max_batches:
                break
            data, ei_cpu, b_cpu, w0_cpu, mc_cpu, mb_cpu = run_batch(data)
            y = data.y.detach().cpu()
            B = int(b_cpu.max().item()) + 1

            for g in range(B):
                yg = int(y[g].item())
                ei_g,  w_g  = slice_graph_edges(ei_cpu, b_cpu, w0_cpu, g)
                ei_g2, mc_g = slice_graph_edges(ei_cpu, b_cpu, mc_cpu, g)
                _,     mb_g = slice_graph_edges(ei_cpu, b_cpu, mb_cpu, g)
                if ei_g is None:
                    continue

                A  = build_dense_A(ei_g,  w_g,  N=args.N, sym=True)
                Ac = build_dense_A(ei_g2, mc_g, N=args.N, sym=True)
                Ab = build_dense_A(ei_g2, mb_g, N=args.N, sym=True)

                if args.topk_matrix > 0:
                    A  = topk_from_matrix(A,  args.topk_matrix)
                    Ac = topk_from_matrix(Ac, args.topk_matrix)
                    Ab = topk_from_matrix(Ab, args.topk_matrix)

                if yg == args.label_ctrl:
                    sum_A  += A;  sum_Ac += Ac; sum_Ab += Ab; n_ctrl += 1
                elif yg == args.label_adhd:
                    sum_A2 += A; sum_Ac2 += Ac; sum_Ab2 += Ab; n_adhd += 1

        if n_ctrl == 0 or n_adhd == 0:
            raise RuntimeError(f"n_ctrl={n_ctrl}, n_adhd={n_adhd}. Increase --max_batches or check labels.")

        A_ctrl  = (sum_A  / n_ctrl).astype(np.float32)
        Ac_ctrl = (sum_Ac / n_ctrl).astype(np.float32)
        Ab_ctrl = (sum_Ab / n_ctrl).astype(np.float32)

        A_adhd  = (sum_A2  / n_adhd).astype(np.float32)
        Ac_adhd = (sum_Ac2 / n_adhd).astype(np.float32)
        Ab_adhd = (sum_Ab2 / n_adhd).astype(np.float32)

        # mean triplets
        out_ctrl = os.path.join(args.outdir, f"{args.dataset}_{args.split}_MEAN_CTRL_n{n_ctrl}.png")
        save_triplet_png(A_ctrl, Ac_ctrl, Ab_ctrl, out_ctrl,
                         [f"CTRL mean A (n={n_ctrl})", "CTRL mean A_causal", "CTRL mean A_bias"])

        out_adhd = os.path.join(args.outdir, f"{args.dataset}_{args.split}_MEAN_ADHD_n{n_adhd}.png")
        save_triplet_png(A_adhd, Ac_adhd, Ab_adhd, out_adhd,
                         [f"ADHD mean A (n={n_adhd})", "ADHD mean A_causal", "ADHD mean A_bias"])

        # diffs
        D_A  = A_adhd  - A_ctrl
        D_Ac = Ac_adhd - Ac_ctrl
        D_Ab = Ab_adhd - Ab_ctrl
        out_diff = os.path.join(args.outdir, f"{args.dataset}_{args.split}_DIFF_A_Ac_Ab_nC{n_ctrl}_nA{n_adhd}.png")
        save_diff_grid_png(D_A, D_Ac, D_Ab, out_diff, prefix=f"{args.dataset} {args.split} DIFF (nC={n_ctrl}, nA={n_adhd})")

        # export top-50 edges (medical analysis)
        edges_dir = os.path.join(args.outdir, "edge_lists_top50")
        k = args.export_topk_edges

        # mean top edges
        save_top_edges_from_matrix(A_ctrl,  os.path.join(edges_dir, f"{args.dataset}_{args.split}_MEAN_CTRL_A_top{k}.csv"),  topk=k, mode="largest")
        save_top_edges_from_matrix(Ac_ctrl, os.path.join(edges_dir, f"{args.dataset}_{args.split}_MEAN_CTRL_Ac_top{k}.csv"), topk=k, mode="largest")
        save_top_edges_from_matrix(Ab_ctrl, os.path.join(edges_dir, f"{args.dataset}_{args.split}_MEAN_CTRL_Ab_top{k}.csv"), topk=k, mode="largest")

        save_top_edges_from_matrix(A_adhd,  os.path.join(edges_dir, f"{args.dataset}_{args.split}_MEAN_ADHD_A_top{k}.csv"),  topk=k, mode="largest")
        save_top_edges_from_matrix(Ac_adhd, os.path.join(edges_dir, f"{args.dataset}_{args.split}_MEAN_ADHD_Ac_top{k}.csv"), topk=k, mode="largest")
        save_top_edges_from_matrix(Ab_adhd, os.path.join(edges_dir, f"{args.dataset}_{args.split}_MEAN_ADHD_Ab_top{k}.csv"), topk=k, mode="largest")

        # diff: export by |diff| (signed weights)
        save_top_edges_from_matrix(D_A,  os.path.join(edges_dir, f"{args.dataset}_{args.split}_DIFF_A_top{k}.csv"),  topk=k, mode="abs")
        save_top_edges_from_matrix(D_Ac, os.path.join(edges_dir, f"{args.dataset}_{args.split}_DIFF_Ac_top{k}.csv"), topk=k, mode="abs")
        save_top_edges_from_matrix(D_Ab, os.path.join(edges_dir, f"{args.dataset}_{args.split}_DIFF_Ab_top{k}.csv"), topk=k, mode="abs")

        print(f"[GROUP] done. n_ctrl={n_ctrl}, n_adhd={n_adhd}")

    print("[DONE] saved to:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
