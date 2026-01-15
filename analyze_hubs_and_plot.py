import os
import numpy as np
import torch
from collections import Counter

from nilearn import datasets, plotting

# ====== 项目内 import（按你的真实路径） ======
from src.utils import parse_arguments
from src.dataset_loader import create_dataloader
from nets.disbg import DisBGModel   # ⚠️ 如果文件名不同，这一行改一下


# ---------- AAL116 ----------
def load_aal():
    aal = datasets.fetch_atlas_aal(version="SPM12")
    coords = plotting.find_parcellation_cut_coords(labels_img=aal.maps)
    labels = aal.labels  # labels[0] = Background
    return coords, labels


def weighted_degree(edge_index, edge_weight, num_nodes=116):
    deg = np.zeros(num_nodes, dtype=np.float32)
    ei = edge_index.cpu().numpy()
    ew = edge_weight.cpu().numpy()
    for (u, v), w in zip(ei.T, ew):
        if u == v:
            continue
        deg[int(u)] += float(w)
        deg[int(v)] += float(w)
    return deg


def topk_adj(edge_index, edge_weight, num_nodes=116, topk=50):
    k = min(topk, edge_weight.numel())
    idx = torch.topk(edge_weight, k=k, largest=True).indices
    ei = edge_index[:, idx].cpu().numpy()
    ew = edge_weight[idx].cpu().numpy()

    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for (u, v), w in zip(ei.T, ew):
        A[int(u), int(v)] = max(A[int(u), int(v)], float(w))
        A[int(v), int(u)] = max(A[int(v), int(u)], float(w))
    return A


# ---------- 全局扫描 label（非常重要） ----------
def scan_sens(loader, name):
    s_min, s_max = 1e9, -1e9
    a_min, a_max = 1e9, -1e9
    for data in loader:
        s = data.y_s.view(-1)
        a = data.y_a.view(-1)
        s_min = min(s_min, int(s.min()))
        s_max = max(s_max, int(s.max()))
        a_min = min(a_min, int(a.min()))
        a_max = max(a_max, int(a.max()))
    print(f"[{name}] y_s min/max: {s_min} {s_max} | y_a min/max: {a_min} {a_max}")


@torch.no_grad()
def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = "vis_hubs"
    os.makedirs(outdir, exist_ok=True)

    # dataloader
    train_loader, val_loader, test_loader = create_dataloader(
        args.dataset, batch_size=args.batch_size
    )

    scan_sens(train_loader, "train")
    scan_sens(val_loader, "val")
    scan_sens(test_loader, "test")

    # model
    model = DisBGModel(args).to(device)
    model.eval()

    # 如果你有 checkpoint，在这里加载
    # ckpt = torch.load("best.pt", map_location="cpu")
    # model.load_state_dict(ckpt["state_dict"], strict=False)

    coords, aal_labels = load_aal()

    hubs_c, hubs_b = [], []
    saved = 0
    MAX_SUBJECTS = 5
    TOPK = 50

    for data in test_loader:
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = model(x, edge_index, batch, return_masks=True)

        # ====== 关键：兼容 9 输出 ======
        if len(out) < 9:
            raise RuntimeError(
                "model.forward() 没有返回 m_c, m_b，请确认 forward return ..."
            )

        logits_d, logits_s, logits_a, z_c, z_b, u_c, u_b, m_c, m_b = out[:9]

        edge_batch = batch[edge_index[0]].cpu()
        B = int(batch.max()) + 1

        for g in range(B):
            idx = (edge_batch == g)

            ei_g = edge_index[:, idx]
            mc_g = m_c[idx]
            mb_g = m_b[idx]
            print("mc_g stats:", float(mc_g.min()), float(mc_g.max()), float(mc_g.mean()))
            print("mb_g stats:", float(mb_g.min()), float(mb_g.max()), float(mb_g.mean()))

            deg_c = weighted_degree(ei_g, mc_g, num_nodes=116)
            deg_b = weighted_degree(ei_g, mb_g, num_nodes=116)
            print("deg_c stats:", deg_c.min(), deg_c.max(), deg_c.mean())
            print("deg_b stats:", deg_b.min(), deg_b.max(), deg_b.mean())
            print("top5 deg_c idx:", np.argsort(-deg_c)[:5])
            print("top5 deg_b idx:", np.argsort(-deg_b)[:5])


            deg_c = weighted_degree(ei_g, mc_g)
            deg_b = weighted_degree(ei_g, mb_g)

            hub_c = int(np.argmax(deg_c))
            hub_b = int(np.argmax(deg_b))

            hubs_c.append(hub_c)
            hubs_b.append(hub_b)

            print(
                f"[Subj {saved}] "
                f"Causal hub: ROI {hub_c+1:03d} ({aal_labels[hub_c+1]}) | "
                f"Bias hub: ROI {hub_b+1:03d} ({aal_labels[hub_b+1]})"
            )

            A_c = topk_adj(ei_g, mc_g, topk=TOPK)
            A_b = topk_adj(ei_g, mb_g, topk=TOPK)

            disp = plotting.plot_connectome(
                A_c, coords, node_size=10,
                title=f"{args.dataset} subj{saved} causal"
            )
            disp.savefig(f"{outdir}/subj{saved}_causal.png", dpi=200)
            disp.close()

            disp = plotting.plot_connectome(
                A_b, coords, node_size=10,
                title=f"{args.dataset} subj{saved} bias"
            )
            disp.savefig(f"{outdir}/subj{saved}_bias.png", dpi=200)
            disp.close()

            saved += 1
            if saved >= MAX_SUBJECTS:
                break
        if saved >= MAX_SUBJECTS:
            break

    print("\n=== Hub consistency ===")
    print("Causal:", Counter(hubs_c))
    print("Bias  :", Counter(hubs_b))
    print(f"[OK] Figures saved to {outdir}")


if __name__ == "__main__":
    main()
