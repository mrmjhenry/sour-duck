"""
Visualization helpers for moduli_pruning_demo.ipynb.

Three figures, all DERIVED from live objects (not hardcoded):

    plot_network_layers(net)            -- the architecture, traced from a forward pass
    plot_quiver(quiver)                 -- the quiver built by QuiverRepresentation(net)
    plot_data_to_representation(net, x) -- one input traced through the layers, then
                                           into its Eq. 5 representation W_x^f for fc1

Palette matches education/project_visualizations/05_quiver_construction.py.
"""

from math import prod

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Rectangle

# palette
INPUT_C = "#94a3b8"
CONV_C  = "#f59e0b"
RELU_C  = "#cbd5e1"
FLAT_C  = "#94a3b8"
FC_C    = "#8b5cf6"
FC_C2   = "#ec4899"
VERT_C  = "#0ea5e9"
INK     = "#1e293b"
MUTED   = "#64748b"

_LINEAR_COLORS = [FC_C, FC_C2, "#10b981", "#f97316"]   # cycle if many Linear layers


def _shape_str(shape):
    return "x".join(str(s) for s in shape)


# ---------------------------------------------------------------------------
# 1. Network layers
# ---------------------------------------------------------------------------
def plot_network_layers(net, input_shape=(1, 4, 4), ax=None, save_path=None):
    """Draw the layer stack of `net`, tracing tensor shapes via a forward pass."""
    leaves = [(n, m) for n, m in net.named_modules()
              if len(list(m.children())) == 0 and n != ""]

    # Trace per-layer output shapes (drop batch dim).
    shapes, hooks = {}, []
    for n, m in leaves:
        hooks.append(m.register_forward_hook(
            lambda mod, inp, out, n=n: shapes.__setitem__(n, tuple(out.shape[1:]))))
    was_training = net.training
    net.eval()
    with torch.no_grad():
        net(torch.zeros(1, *input_shape))
    for h in hooks:
        h.remove()
    if was_training:
        net.train()

    # Build the ordered card list, inserting a Flatten where a Linear follows a
    # multi-dim activation.
    cards = [("Input", _shape_str(input_shape), None, INPUT_C)]
    prev_shape = input_shape
    for n, m in leaves:
        out = shapes[n]
        if isinstance(m, nn.Linear):
            if len(prev_shape) > 1:
                cards.append(("Flatten", _shape_str((m.in_features,)), None, FLAT_C))
            cards.append(("Linear", _shape_str(out),
                          f"{m.in_features}->{m.out_features}\n{m.weight.numel()} w", FC_C))
        elif isinstance(m, nn.Conv2d):
            cards.append(("Conv2d", _shape_str(out),
                          f"{m.in_channels}->{m.out_channels}, {m.kernel_size[0]}x{m.kernel_size[1]}"
                          f"\n{m.weight.numel()} w", CONV_C))
        elif isinstance(m, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
            cards.append((type(m).__name__, _shape_str(out), "no params", RELU_C))
        else:
            cards.append((type(m).__name__, _shape_str(out), None, RELU_C))
        prev_shape = out

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(min(2.0 * len(cards), 16), 3.2))
    ax.set_xlim(0, len(cards)); ax.set_ylim(0, 1); ax.axis("off")

    w, h, y = 0.82, 0.5, 0.28
    for i, (title, shape, sub, color) in enumerate(cards):
        x = i + (1 - w) / 2
        solid = color != RELU_C and color != FLAT_C
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc=color, ec=color, alpha=0.18 if solid else 0.55, lw=0))
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.06",
                     fc="none", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h - 0.1, title, ha="center", va="center",
                fontsize=10, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 0.27, shape, ha="center", va="center",
                fontsize=8.5, color=INK)
        if sub:
            ax.text(x + w / 2, y + 0.1, sub, ha="center", va="center",
                    fontsize=7.2, color=MUTED)
        if i < len(cards) - 1:
            ax.add_patch(FancyArrowPatch((x + w, y + h / 2), (i + 1 + (1 - w) / 2, y + h / 2),
                         arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.3))

    ax.text(len(cards) / 2, 0.92,
            "Network layers  (only Conv2d / Linear carry weights -> quiver edges)",
            ha="center", fontsize=11, fontweight="bold", color=INK)
    if own_fig:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        return fig
    return ax


# ---------------------------------------------------------------------------
# 2. Derived quiver
# ---------------------------------------------------------------------------
def plot_quiver(quiver, ax=None, save_path=None):
    """Draw the quiver in `quiver` (a QuiverRepresentation) as a layered graph.

    Conv2d layers use the "filter = edge" convention (out_channels edges, fan-in
    collapsed); Linear layers draw one edge per scalar weight.
    """
    net = quiver.network
    players = [(n, m) for n, m in net.named_modules()
               if isinstance(m, (nn.Linear, nn.Conv2d))]

    edges_by_layer = {}
    for e_idx, e_info, lname in quiver.edges:
        edges_by_layer.setdefault(lname, []).append(e_info)

    def dims(m):
        if isinstance(m, nn.Linear):
            return m.in_features, m.out_features
        return m.in_channels, m.out_channels

    # Column plan: input column, then a column per layer output, inserting a
    # reshape column when a layer's in-width != the previous column width.
    in0, _ = dims(players[0][0] if False else players[0][1])
    columns = [{"size": in0, "label": "input", "kind": "input"}]
    steps = []   # (layer_name, color, kind, src_col_index, dst_col_index, reshape_col_index|None)
    li = 0
    for name, m in players:
        cin, cout = dims(m)
        prev = columns[-1]
        reshape_idx = None
        if prev["size"] != cin:
            columns.append({"size": cin, "label": "flatten", "kind": "reshape"})
            reshape_idx = len(columns) - 1
        src_idx = len(columns) - 1
        color = (_LINEAR_COLORS[li % len(_LINEAR_COLORS)]
                 if isinstance(m, nn.Linear) else CONV_C)
        if isinstance(m, nn.Linear):
            li += 1
        columns.append({"size": cout, "label": name, "kind": "layer", "color": color})
        steps.append((name, color, "linear" if isinstance(m, nn.Linear) else "conv",
                      src_idx, len(columns) - 1, reshape_idx))

    n_cols = len(columns)
    xs = np.linspace(0.6, 9.4, n_cols)
    lo, hi = 1.0, 8.6

    def col_ys(n):
        if n == 1:
            return np.array([(lo + hi) / 2])
        return np.linspace(hi, lo, n)

    ys = [col_ys(c["size"]) for c in columns]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # --- edges first (under nodes) ---
    for name, color, kind, src_idx, dst_idx, reshape_idx in steps:
        infos = edges_by_layer.get(name, [])
        if reshape_idx is not None:
            # dashed reshape connectors: previous (conv-out) column -> flatten column
            prev_idx = reshape_idx - 1
            pn, fn = columns[prev_idx]["size"], columns[reshape_idx]["size"]
            grp = max(1, fn // max(1, pn))
            for pi, py in enumerate(ys[prev_idx]):
                for fy in ys[reshape_idx][pi * grp:(pi + 1) * grp]:
                    ax.plot([xs[prev_idx], xs[reshape_idx]], [py, fy],
                            color=MUTED, alpha=0.35, lw=0.7, ls=(0, (2, 2)), zorder=1)
        if kind == "linear":
            alpha = max(0.18, min(0.55, 12.0 / max(1, len(infos))))
            lw = 0.6 if len(infos) > 16 else 0.9
            for info in infos:
                i_in, j_out = info
                ax.plot([xs[src_idx], xs[dst_idx]],
                        [ys[src_idx][i_in], ys[dst_idx][j_out]],
                        color=color, alpha=alpha, lw=lw, zorder=1)
        else:  # conv: out_channels edges from the input-column centroid (fan-in collapsed)
            cx = xs[src_idx]; cy = ys[src_idx].mean()
            if len(ys[src_idx]) > 1:
                for sy in ys[src_idx]:
                    ax.plot([cx, cx + 0.18], [sy, cy], color=MUTED, alpha=0.4, lw=0.7, zorder=1)
                cx = cx + 0.18
            for fi in infos:
                ax.plot([cx, xs[dst_idx]], [cy, ys[dst_idx][fi]],
                        color=color, alpha=0.9, lw=2.0, zorder=1)

    # --- nodes ---
    for ci, col in enumerate(columns):
        r = 0.18 if col["size"] <= 5 else 0.12
        if col["kind"] == "input":
            nc = INPUT_C
        elif col["kind"] == "reshape":
            nc = FLAT_C
        else:
            nc = col["color"]
        for yy in ys[ci]:
            ax.add_patch(Circle((xs[ci], yy), r, fc="white", ec=nc, lw=1.5, zorder=3))
            ax.add_patch(Circle((xs[ci], yy), r * 0.55, fc=nc, ec="none", zorder=4))
        label = col["label"]
        ax.text(xs[ci], 9.15, label, ha="center", fontsize=8.6, color=INK, fontweight="bold")
        ax.text(xs[ci], 0.45, f"{col['size']}", ha="center", fontsize=8.4, color=MUTED)

    # --- title / stats / legend ---
    n_e = len(quiver.edges); n_v = len(quiver.vertices)
    ax.text(5, 9.75, "Derived quiver representation", ha="center",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(5, 9.5,
            f"#E (edges/coordinates) = {n_e}    #V (vertices) = {n_v}    "
            f"moduli dim = #E - #V = {n_e - n_v}",
            ha="center", fontsize=9.5, color=MUTED)

    handles = []
    for name, color, kind, *_ in steps:
        n_layer = len(edges_by_layer.get(name, []))
        tag = f"{name}: {n_layer} edge" + ("" if n_layer == 1 else "s")
        if kind == "conv":
            tag += " (filter=edge)"
        handles.append(plt.Line2D([0], [0], color=color, lw=2.4, label=tag))
    handles.append(plt.Line2D([0], [0], color=MUTED, lw=0.9, ls=(0, (2, 2)),
                              label="reshape (not weighted)"))
    ax.legend(handles=handles, loc="lower center", ncol=len(handles),
              frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, -0.02))

    if own_fig:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        return fig
    return ax


# ---------------------------------------------------------------------------
# 3. One input -> layers -> representation W_x^f
# ---------------------------------------------------------------------------
def _imshow(ax, data, cmap, signed=False, annot=False, fs=6.5, vmax=None):
    data = np.asarray(data, dtype=float)
    if signed:
        vm = vmax if vmax is not None else (np.abs(data).max() or 1.0)
        im = ax.imshow(data, cmap=cmap, vmin=-vm, vmax=vm, aspect="auto")
        hot = 0.6 * vm
    else:
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        hot = 0.6 * (data.max() if data.size else 1.0)
    ax.set_xticks([]); ax.set_yticks([])
    if annot:
        for (r, c), v in np.ndenumerate(data):
            ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=fs,
                    color="white" if abs(v) > hot else "black")
    return im


def plot_data_to_representation(net, x, true_label=None, save_path=None):
    """Trace ONE input `x` (shape (1,1,4,4)) through the network and show how the
    resulting activations build the Eq. 5 representation W_x^f for the fc1 layer.
    """
    # Capture activations with forward hooks.
    acts, hooks = {}, []
    for name, mod in [("conv1", net.conv1), ("relu1", net.relu1),
                      ("fc1", net.fc1), ("relu2", net.relu2), ("fc2", net.fc2)]:
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, n=name: acts.__setitem__(n, o.detach())))
    was_training = net.training
    net.eval()
    with torch.no_grad():
        _ = net(x)
    for h in hooks:
        h.remove()
    if was_training:
        net.train()

    img    = x[0, 0].cpu().numpy()                       # (4,4)
    conv   = acts["relu1"][0].cpu().numpy()              # (2,2,2) post-ReLU
    a      = acts["relu1"][0].reshape(-1).cpu().numpy()  # (8,) fc1 input
    preact = acts["fc1"][0].cpu().numpy()                # (4,) fc1 pre-activation
    hidden = acts["relu2"][0].cpu().numpy()              # (4,)
    logit  = acts["fc2"][0].cpu().numpy()                # (3,)
    W      = net.fc1.weight.detach().cpu().numpy()       # (4,8)
    eps    = 1e-8
    denom  = np.where(np.abs(preact) < eps, np.where(preact >= 0, eps, -eps), preact)
    M      = W * a[None, :] / denom[:, None]             # (4,8) = W_x^f
    pred   = int(logit.argmax())

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.965, "One input, traced through the network and into its representation $W_x^f$",
             ha="center", fontsize=15, fontweight="bold", color=INK)

    # ---------------- Section 1: forward pass ----------------
    fig.text(0.5, 0.915, "1 · Data flowing through the layers",
             ha="center", fontsize=12, fontweight="bold", color=INK)

    ax_in   = fig.add_axes([0.045, 0.66, 0.10, 0.18])
    ax_conv = fig.add_axes([0.225, 0.66, 0.13, 0.18])
    ax_flat = fig.add_axes([0.455, 0.60, 0.032, 0.30])
    ax_hid  = fig.add_axes([0.600, 0.66, 0.030, 0.18])
    ax_log  = fig.add_axes([0.740, 0.685, 0.030, 0.13])

    _imshow(ax_in, img, "viridis", annot=True, fs=7)
    ax_in.set_title("input  1×4×4", fontsize=9, color=INK)

    montage = np.concatenate([conv[0], conv[1]], axis=1)   # (2,4)
    _imshow(ax_conv, montage, "viridis", annot=True, fs=7)
    ax_conv.axvline(1.5, color="white", lw=2)
    ax_conv.set_title("conv+ReLU  2×(2×2)", fontsize=9, color=INK)

    _imshow(ax_flat, a[:, None], "viridis", annot=True, fs=7)
    ax_flat.set_title("$a_i$ (8)", fontsize=9, color=INK)

    _imshow(ax_hid, hidden[:, None], "viridis", annot=True, fs=7)
    ax_hid.set_title("fc1+ReLU (4)", fontsize=9, color=INK)

    im_log = _imshow(ax_log, logit[:, None], "coolwarm", signed=True, annot=True, fs=7)
    ax_log.set_title("logits (3)", fontsize=9, color=INK)
    ax_log.add_patch(Rectangle((-0.5, pred - 0.5), 1, 1, fill=False, ec="#16a34a", lw=2.5))

    def farrow(x0, x1, y, label):
        fig.add_artist(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure,
                       arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.6))
        fig.text((x0 + x1) / 2, y + 0.02, label, ha="center", fontsize=8, color=MUTED)

    farrow(0.150, 0.222, 0.75, "conv")
    farrow(0.358, 0.452, 0.75, "ReLU\n→flatten")
    farrow(0.490, 0.597, 0.75, "fc1\n→ReLU")
    farrow(0.632, 0.737, 0.75, "fc2")
    tl = "" if true_label is None else f"  (true {true_label})"
    fig.text(0.80, 0.75, f"→ predicted\nclass {pred}{tl}", ha="left", va="center",
             fontsize=10, fontweight="bold", color="#16a34a")

    # ---------------- Section 2: representation construction ----------------
    fig.text(0.5, 0.49,
             r"2 · How those activations build the fc1 representation   "
             r"$W_x^f[j,i] = W[j,i]\,a_i \,/\, \mathrm{preact}_j$",
             ha="center", fontsize=12, fontweight="bold", color=INK)

    ax_a  = fig.add_axes([0.10, 0.385, 0.34, 0.045])
    ax_W  = fig.add_axes([0.10, 0.10, 0.34, 0.265])
    ax_p  = fig.add_axes([0.455, 0.10, 0.025, 0.265])
    ax_M  = fig.add_axes([0.62, 0.10, 0.34, 0.265])

    # Robust color scale for M: a near-dead output neuron (preact ~ 0) can make a
    # single W_x^f entry huge; clip the scale to the 95th pct so the bulk is visible.
    vmM = float(np.percentile(np.abs(M), 95)) or (np.abs(M).max() or 1.0)
    _imshow(ax_a, a[None, :], "viridis", annot=True, fs=6.5)
    _imshow(ax_W, W, "coolwarm", signed=True, annot=True, fs=6)
    _imshow(ax_p, preact[:, None], "coolwarm", signed=True, annot=True, fs=6.5)
    im_M = _imshow(ax_M, M, "coolwarm", signed=True, annot=True, fs=6, vmax=vmM)

    fig.text(0.27, 0.435, r"$a_i$  (input activations, 8 columns)",
             ha="center", fontsize=9, color=INK)
    fig.text(0.27, 0.085, r"$W[j,i]$  (fc1 weights, 4×8)", ha="center", fontsize=9, color=INK)
    fig.text(0.4675, 0.375, r"$\mathrm{preact}_j$", ha="center", fontsize=9, color=INK)
    fig.text(0.79, 0.085, r"$W_x^f[j,i]$  (contribution fractions)",
             ha="center", fontsize=9, color=INK)
    fig.text(0.55, 0.23, "=", ha="center", va="center", fontsize=26, color=INK)
    fig.text(0.27, 0.405, r"$\times\, a_i$ per column      $\div\, \mathrm{preact}_j$ per row",
             ha="center", fontsize=8, color=MUTED)

    # Highlight a coordinate feeding a HEALTHY output neuron (largest |preact|),
    # not the global argmax|M| (which is usually a near-dead neuron's blow-up).
    j = int(np.argmax(np.abs(preact)))
    i = int(np.argmax(np.abs(W[j] * a)))
    for ax, (cc, rr) in [(ax_W, (i, j)), (ax_M, (i, j)), (ax_a, (i, 0)), (ax_p, (0, j))]:
        ax.add_patch(Rectangle((cc - 0.5, rr - 0.5), 1, 1, fill=False, ec="#16a34a", lw=2.2))
    fig.text(0.5, 0.045,
             f"highlighted coordinate  (i={i}, j={j}):   "
             f"$W_x^f$ = {W[j,i]:.2f} × {a[i]:.2f} ÷ {preact[j]:.2f} = {M[j,i]:.2f}",
             ha="center", fontsize=9.5, color="#16a34a")

    # small shared colorbar for the signed panels
    cax = fig.add_axes([0.965, 0.10, 0.008, 0.265])
    fig.colorbar(im_M, cax=cax)
    cax.tick_params(labelsize=6)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    from torch.utils.data import TensorDataset, DataLoader
    from moduli_space_pruning import QuiverRepresentation
    from experiments import ExperimentRunner

    class TinyCNN(nn.Module):
        def __init__(self, num_classes=3):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 2, 3); self.relu1 = nn.ReLU()
            self.fc1 = nn.Linear(8, 4); self.relu2 = nn.ReLU()
            self.fc2 = nn.Linear(4, num_classes)

        def forward(self, x):
            x = self.relu1(self.conv1(x)); x = x.view(x.size(0), -1)
            x = self.relu2(self.fc1(x)); return self.fc2(x)

    def make_image_dataset(n_per_class, noise=0.25, seed=0):
        rng = np.random.default_rng(seed)
        t0 = np.zeros((4, 4), np.float32); t0[:, 1:3] = 1.0
        t1 = np.zeros((4, 4), np.float32); t1[1:3, :] = 1.0
        t2 = np.zeros((4, 4), np.float32); np.fill_diagonal(t2, 1.0)
        X, y = [], []
        for c, t in enumerate([t0, t1, t2]):
            for _ in range(n_per_class):
                X.append(t + rng.normal(0, noise, (4, 4)).astype(np.float32)); y.append(c)
        return np.stack(X)[:, None].astype(np.float32), np.array(y)

    torch.manual_seed(0); np.random.seed(0)
    Xtr, ytr = make_image_dataset(200, seed=1)
    Xte, yte = make_image_dataset(50, seed=2)
    mu, sd = Xtr.mean(), Xtr.std()
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    net = TinyCNN()
    ckpt = "tinycnn_trained.pt"
    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt)); print("loaded", ckpt)
    else:
        tl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=32, shuffle=True)
        ExperimentRunner(device="cpu").train_network(net, tl, epochs=40, lr=1e-2)
        torch.save(net.state_dict(), ckpt); print("trained + saved", ckpt)

    plot_network_layers(net, save_path="tinycnn_layers.png")
    plot_quiver(QuiverRepresentation(net), save_path="tinycnn_quiver.png")
    x0 = torch.from_numpy(Xte[0:1])
    plot_data_to_representation(net, x0, true_label=int(yte[0]),
                               save_path="tinycnn_dataflow.png")
    print("wrote tinycnn_layers.png, tinycnn_quiver.png, tinycnn_dataflow.png")
