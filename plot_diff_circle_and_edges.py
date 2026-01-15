# plot_diff_circle_and_edges.py
# ------------------------------------------------------------
# Draw DIFF circle plots (paper-style: network-colored text + node dots)
# + export Top-K edge tables from DIFF npz
#
# Excel:
#   - Node display name: column D
#   - Network label:     column E
#
# Input npz:
#   - Dc_pos, Dc_neg, Db_pos, Db_neg (all 116x116)
#
# Output:
#   - 4 circle PNGs + 4 CSV edge tables
#
# Install:
#   pip install numpy pandas matplotlib openpyxl
#
# Example:
#   python plot_diff_circle_and_edges.py \
#     --diff_npz vis_subgraphs/ADHD_test_DIFF_top50_nC49_nA28.npz \
#     --excel AAL.xlsx \
#     --out_dir vis_subgraphs/diff_circle \
#     --top_percent 97 \
#     --topk_edges 50
# ------------------------------------------------------------

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


# ============================================================
# 1) Palette: match your screenshot text colors
# ============================================================
# Visual: black
# Somatomotor: orange-red
# Dorsal attention: pink/magenta
# Ventral attention: green
# Limbic: yellow
# Frontoparietal: blue
# Default mode: cyan
# Cerebellum: dark green
# Subcortical: purple
NETWORK_COLOR_MAP = {
    "VN": "#000000",   # black
    "SMN": "#E74C3C",   # red
    "DAN": "#FF4FB3",   # magenta
    "VAN": "#2ECC71",   # green
    "LIN": "#F1C40F",   # yellow
    "FPN": "#1F57FF",   # blue
    "DMN": "#00CFE3",   # cyan
    "CB":  "#2E8B57",   # dark green
    "SBN": "#8E44AD",   # purple
    "Unknown": "#BDBDBD"
}

def map_network_to_colors(net_labels, color_map):
    return [color_map.get(x, color_map["Unknown"]) for x in net_labels]


import numpy as np
import pandas as pd
from collections import Counter

def normalize_network_label(net) -> str:
    """
    Map Excel E-col labels to canonical keys:
      VIS, SMN, DAN, VAN, LIM, FPN, DMN, CB, SUB, Unknown
    Supports:
      - short codes: DMN/FPN/DAN/VAN/SMN/VIS/LIM/SUB/CB
      - full names: "Default mode network", etc.
      - single-letter codes you might be using: D/F/V/L/S/C/M/A/T
    """
    if net is None:
        return "Unknown"
    s = str(net).replace("\xa0", " ").strip()
    if s == "" or s.lower() == "nan":
        return "Unknown"

    # ---- single-letter codes (if your Excel uses them) ----
    if len(s) == 1:
        code = s.upper()
        letter_map = {
            "V": "VN",   # Visual
            "M": "SMN",   # Somatomotor
            "A": "DAN",   # Dorsal attention
            "T": "VAN",   # Ventral attention
            "L": "LIN",   # Limbic
            "F": "FPN",   # Frontoparietal
            "D": "DMN",   # Default mode
            "C": "CB",    # Cerebellum
            "S": "SBN",   # Subcortical
        }
        return letter_map.get(code, "Unknown")

    sl = s.lower()

    # ---- short codes / substrings ----
    if sl.startswith("vn") or "visual" in sl:
        return "VN"
    if sl.startswith("smn") or "somato" in sl or "sensorimotor" in sl:
        return "SMN"
    if sl.startswith("dan") or "dorsal attention" in sl:
        return "DAN"
    if sl.startswith("van") or "ventral attention" in sl:
        return "VAN"
    if sl.startswith("lin") or "limbic" in sl:
        return "LIN"
    if sl.startswith("fpn") or sl.startswith("cen") or "frontoparietal" in sl:
        return "FPN"
    if sl.startswith("dmn") or "default mode" in sl:
        return "DMN"
    if sl.startswith("cb") or "cerebell" in sl:
        return "CB"
    if sl.startswith("sbn") or "subcort" in sl or "thalam" in sl or "basal" in sl:
        return "SBN"

    return "Unknown"


def load_node_names_and_networks_from_excel(
    excel_path: str,
    col_name: str = "D",   # node display name
    col_net: str = "E",    # network label -> color
    start_row: int = 2,
    n: int = 116,
):
    df = pd.read_excel(excel_path, header=None)

    name_col = ord(col_name.upper()) - ord("A")
    net_col = ord(col_net.upper()) - ord("A")
    start = start_row - 1

    raw_name = df.iloc[start:, name_col].tolist()
    raw_net  = df.iloc[start:, net_col].tolist()

    node_names, net_labels = [], []
    for i in range(n):
        # ---- D col: display name (keep your original naming style) ----
        if i < len(raw_name):
            v = raw_name[i]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                node_names.append(f"ROI_{i+1:03d}")
            else:
                s = str(v).replace("\xa0", " ").strip()
                node_names.append(s if s != "" else f"ROI_{i+1:03d}")
        else:
            node_names.append(f"ROI_{i+1:03d}")

        # ---- E col: network label ----
        if i < len(raw_net):
            net_labels.append(normalize_network_label(raw_net[i]))
        else:
            net_labels.append("Unknown")

    print(f"[OK] Loaded {len(node_names)} node names (D) and {len(net_labels)} network labels (E).")
    print("[INFO] Network counts:", Counter(net_labels))
    return node_names, net_labels



def load_node_names_and_networks_from_excel(
    excel_path: str,
    col_name: str = "D",
    col_net: str = "E",
    start_row: int = 2,
    n: int = 116
):
    """Always returns length-n lists: node_names (D), net_labels (E normalized)."""
    df = pd.read_excel(excel_path, header=None)

    name_col = ord(col_name.upper()) - ord("A")
    net_col = ord(col_net.upper()) - ord("A")
    start = start_row - 1

    raw_name = df.iloc[start:, name_col].tolist()
    raw_net = df.iloc[start:, net_col].tolist()

    node_names, net_labels = [], []
    for i in range(n):
        # node display name
        if i < len(raw_name):
            v = raw_name[i]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                node_names.append(f"ROI_{i+1:03d}")
            else:
                s = str(v).replace("\xa0", " ").strip()
                node_names.append(s if s != "" else f"ROI_{i+1:03d}")
        else:
            node_names.append(f"ROI_{i+1:03d}")

        # network label
        if i < len(raw_net):
            v = raw_net[i]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                net_labels.append("Unknown")
            else:
                net_labels.append(normalize_network_label(v))
        else:
            net_labels.append("Unknown")

    # debug summary
    from collections import Counter
    print(f"[OK] Loaded node_names={len(node_names)}, net_labels={len(net_labels)} from Excel.")
    print("[INFO] Network counts:", Counter(net_labels))
    return node_names, net_labels


def map_network_to_colors(net_labels, color_map):
    return [color_map.get(x, color_map["Unknown"]) for x in net_labels]


# ============================================================
# 2) Circle / chord plot in matplotlib (fully controllable)
#    - node dots on ring (colored by network)
#    - label text colored by network
# ============================================================
def _bezier_arc(p0, p1, bend=0.35):
    """
    Quadratic Bezier from p0 to p1 with control point near center.
    bend controls how much it curves to center.
    """
    x0, y0 = p0
    x1, y1 = p1
    cx, cy = 0.0, 0.0
    # pull control point towards center but keep a bit of direction
    cx = (x0 + x1) * 0.5 * (1 - bend)
    cy = (y0 + y1) * 0.5 * (1 - bend)

    verts = [p0, (cx, cy), p1]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    return Path(verts, codes)


import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def plot_circle_matplotlib(
    A,
    node_names,     # D col names
    node_colors,    # colors from E col networks
    out_png,
    title,
    top_percent=97.0,
    dpi=300,
    fig_size=13.0,      # bigger canvas reduces overlap
    node_size=12,       # smaller dots like your example
    font_size=10.5,        # smaller text helps a lot
    ring_radius=1.0,
    label_radius=1.02,  # push labels outward to reduce collisions
    edge_alpha_min=0.15,
    edge_alpha_max=0.65,
    edge_lw_min=0.6,
    edge_lw_max=3.0
):
    import numpy as np

    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got {A.shape}")
    if len(node_names) != n or len(node_colors) != n:
        raise ValueError("node_names/node_colors length must match matrix size")

    # sym + zero diag
    A = (A + A.T) / 2.0
    np.fill_diagonal(A, 0.0)

    # threshold by abs percentile
    thr = np.percentile(np.abs(A), float(top_percent))
    W = np.where(np.abs(A) >= thr, A, 0.0)

    # edges list
    iu, iv = np.triu_indices(n, k=1)
    w = W[iu, iv]
    keep = w != 0
    iu, iv, w = iu[keep], iv[keep], np.abs(w[keep])

    w_max = float(w.max()) if w.size else 1.0
    w_min = float(w.min()) if w.size else 0.0
    denom = (w_max - w_min + 1e-12)

    # positions: start at top, clockwise
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2*np.pi, n, endpoint=False)
    xs = ring_radius * np.cos(angles)
    ys = ring_radius * np.sin(angles)

    def bezier_path(p0, p1, bend=0.48):
        x0, y0 = p0
        x1, y1 = p1
        cx = (x0 + x1) * 0.5 * (1 - bend)
        cy = (y0 + y1) * 0.5 * (1 - bend)
        verts = [p0, (cx, cy), p1]
        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        return Path(verts, codes)

    fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    # edges (behind)
    for a, b, ww in zip(iu, iv, w):
        t = (float(ww) - w_min) / denom
        alpha = edge_alpha_min + t * (edge_alpha_max - edge_alpha_min)
        lw = edge_lw_min + t * (edge_lw_max - edge_lw_min)

        p0 = (xs[a], ys[a])
        p1 = (xs[b], ys[b])

        ec = node_colors[a]  # multi-color chords like your example

        path = bezier_path(p0, p1, bend=0.52)
        ax.add_patch(PathPatch(path, facecolor="none", edgecolor=ec, lw=lw, alpha=alpha))

    # node dots
    ax.scatter(xs, ys, s=node_size, c=node_colors, edgecolors="none", zorder=10)

    # labels: strictly tangential (perpendicular to radius)
    # ---------------------------
# labels: RADIAL (from center outward)
# ---------------------------
    for i in range(n):
        ang = angles[i]
        lx = label_radius * np.cos(ang)
        ly = label_radius * np.sin(ang)

        # radial rotation: along the radius direction
        rot = np.degrees(ang)

        # keep text readable:
        # if point is on left half, flip 180 so text is not upside-down
        if np.cos(ang) < 0:
            rot += 180
            ha = "right"
        else:
            ha = "left"

        ax.text(
            lx, ly, node_names[i],
            color=node_colors[i],
            fontsize=font_size,
            rotation=rot,
            rotation_mode="anchor",
            ha=ha, va="center"
        )

    ax.set_title(title, fontsize=14, pad=18)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved circle: {out_png}")


# ============================================================
# 3) Top-K edge table export
# ============================================================
def save_topk_edges_from_matrix(A, node_names, out_csv, topk=50, mode="largest"):
    """
    A: (116,116) symmetric.
    mode:
      - "largest": top by A[u,v] (assumes nonnegative for POS/NEG maps)
      - "abs":     top by abs(A[u,v]) but stores signed weight
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError(f"A must be square, got {A.shape}")

    iu, iv = np.triu_indices(n, k=1)
    vals = A[iu, iv]
    score = np.abs(vals) if mode == "abs" else vals

    keep = score > 0
    iu, iv, vals, score = iu[keep], iv[keep], vals[keep], score[keep]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("rank,u_roi,u_label,v_roi,v_label,weight\n")
        if vals.size == 0:
            print(f"[WARN] No nonzero edges. Saved empty csv: {out_csv}")
            return

        k = min(int(topk), vals.size)
        order = np.argsort(-score)[:k]

        for r, idx in enumerate(order, 1):
            u = int(iu[idx])
            v = int(iv[idx])
            w = float(vals[idx])
            f.write(f"{r},{u+1},{node_names[u]},{v+1},{node_names[v]},{w:.8f}\n")

    print(f"[OK] Saved Top-{min(int(topk), vals.size)} edges: {out_csv}")


# ============================================================
# 4) Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diff_npz", type=str, required=True)
    p.add_argument("--excel", type=str, default="AAL.xlsx",
                   help="Excel containing node name (D) & network label (E)")
    p.add_argument("--out_dir", type=str, default="diff_circle_out")
    p.add_argument("--top_percent", type=float, default=97.0,
                   help="edge threshold by abs percentile (95-99 typical)")
    p.add_argument("--topk_edges", type=int, default=50)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load node names (D) + network labels (E)
    node_names, net_labels = load_node_names_and_networks_from_excel(
    excel_path=args.excel,
    col_name="D",
    col_net="E",
    start_row=2,
    n=116
)
    node_colors = map_network_to_colors(net_labels, NETWORK_COLOR_MAP)

    # Load diff npz
    z = np.load(args.diff_npz)
    required = ["Dc_pos", "Dc_neg", "Db_pos", "Db_neg"]
    for k in required:
        if k not in z:
            raise KeyError(f"Missing '{k}' in npz. Found keys: {list(z.keys())}")

    Dc_pos = z["Dc_pos"]
    Dc_neg = z["Dc_neg"]
    Db_pos = z["Db_pos"]
    Db_neg = z["Db_neg"]

    # ---- Circle plots (paper-style) ----
    plot_circle_matplotlib(
        Dc_pos, node_names, node_colors,
        os.path.join(args.out_dir, "DIFF_CAUSAL_POS_circle.png"),
        title="DIFF CAUSAL POS (ABIDE > CTRL)",
        top_percent=args.top_percent, dpi=args.dpi
    )

    plot_circle_matplotlib(
        Dc_neg, node_names, node_colors,
        os.path.join(args.out_dir, "DIFF_CAUSAL_NEG_circle.png"),
        title="DIFF CAUSAL NEG (CTRL > ABIDE)",
        top_percent=args.top_percent, dpi=args.dpi
    )

    plot_circle_matplotlib(
        Db_pos, node_names, node_colors,
        os.path.join(args.out_dir, "DIFF_BIAS_POS_circle.png"),
        title="DIFF BIAS POS (ABIDE > CTRL)",
        top_percent=args.top_percent, dpi=args.dpi
    )

    plot_circle_matplotlib(
        Db_neg, node_names, node_colors,
        os.path.join(args.out_dir, "DIFF_BIAS_NEG_circle.png"),
        title="DIFF BIAS NEG (CTRL > ABIDE)",
        top_percent=args.top_percent, dpi=args.dpi
    )

    # ---- Top-K edge tables ----
    edges_dir = os.path.join(args.out_dir, "edge_tables_topk")
    save_topk_edges_from_matrix(Dc_pos, node_names, os.path.join(edges_dir, "DIFF_CAUSAL_POS_topK.csv"),
                                topk=args.topk_edges, mode="largest")
    save_topk_edges_from_matrix(Dc_neg, node_names, os.path.join(edges_dir, "DIFF_CAUSAL_NEG_topK.csv"),
                                topk=args.topk_edges, mode="largest")
    save_topk_edges_from_matrix(Db_pos, node_names, os.path.join(edges_dir, "DIFF_BIAS_POS_topK.csv"),
                                topk=args.topk_edges, mode="largest")
    save_topk_edges_from_matrix(Db_neg, node_names, os.path.join(edges_dir, "DIFF_BIAS_NEG_topK.csv"),
                                topk=args.topk_edges, mode="largest")

    print("\n[DONE] Outputs saved under:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
