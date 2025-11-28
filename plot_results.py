# plot_results.py
import math
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

DTYPE  = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pytorch_skeleton import ( # reuse physics & complex helpers
    LSForward, YukawaCutoff,
    creal, cimag, cplx
)  # :contentReference[oaicite:12]{index=12}

def to_np(t):
    """Detach if needed, move to CPU, and convert to numpy."""
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)

def mag(z): return torch.sqrt(creal(z)**2 + cimag(z)**2)

# ---- load data and checkpoint ----
data_path = Path("data/synth_case.pt")
ckpt_path = Path("ckpt/best.pt")
ds   = torch.load(data_path, map_location=device)
ckpt = torch.load(ckpt_path, map_location=device)

pts      = ds["pts"].to(device)
W        = ds["W"].to(device)
k_values = [float(k) for k in ds["k_values"]]
theta    = ds["theta"].to(device)
obs_dirs = ds["obs_dirs"].to(device)
V_true   = ds["V_true"].to(device)

# rebuild LS layer for evaluation
N_eval = int(ckpt.get("config", {}).get("N_eval", 3))
ls = LSForward(pts, W, k_values, N_iter=N_eval, chunk_size=512).to(device)  # :contentReference[oaicite:13]{index=13}

# learned potential module (load state)
pot_module = YukawaCutoff(learn_w=True, init=(2.0,0.8,2.0,0.35)).to(device)
pot_module.load_state_dict(ckpt["pot_state"])
with torch.no_grad():
    r_nodes = torch.linalg.norm(pts, dim=-1)
    V_learn, params = pot_module(r_nodes)
    f_learn = ls(V_learn, obs_dirs)
    f_true_eval = ls(V_true,  obs_dirs)  # evaluate truth at same N_eval for fairness

# ---- plots ----
# ---------------- plotting ----------------
# sort nodes by radius for a clean 1D potential profile
r = torch.linalg.norm(pts, dim=-1)
idx = torch.argsort(r)
r_sorted       = to_np(r[idx])
V_true_sorted  = to_np(V_true[idx])
V_learn_sorted = to_np(V_learn[idx])

K        = len(k_values)
theta_np = to_np(theta)

# -------- Figure 1: Potential true vs learned --------
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(r_sorted, V_true_sorted, label="Initial $V_{\\mathrm{true}}(r)$", linewidth=1.5)
ax1.plot(r_sorted, V_learn_sorted, "--", label="Learned $V_{\\phi}(r)$", linewidth=1.5)

A, mu, Rc, w = params  # from YukawaCutoff forward
ax1.set_title(
    f"Potential (N_eval={N_eval})\n"
    f"Recovered A={A.item():.2f}, μ={mu.item():.2f}, Rc={Rc.item():.2f}, w={w.item():.2f}"
)
ax1.set_xlabel("r")
ax1.set_ylabel("V(r)")
ax1.grid(True, alpha=0.3)
ax1.legend()
fig1.tight_layout()
fig1.savefig("potential_comparison.png", dpi=150)

# -------- Figure 2: |f(θ,k)| for ALL k-values --------
rows = int(math.ceil(K / 2))   # 2 columns, enough rows to cover all K
cols = 2
fig2, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
axes = axes.reshape(-1)

for ki in range(K):
    ax = axes[ki]
    ax.plot(theta_np, to_np(mag(f_true_eval[ki])),
            label=f"True | k={k_values[ki]:.2f}", linewidth=1.5)
    ax.plot(theta_np, to_np(mag(f_learn[ki])),
            "--", label=f"Learned | k={k_values[ki]:.2f}", linewidth=1.5)
    ax.set_xlabel("θ")
    ax.set_ylabel("|f(θ,k)|")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

# hide any unused subplots if K is odd
for j in range(K, len(axes)):
    axes[j].axis("off")

fig2.suptitle("Far-field magnitude: true vs learned for all k-values", y=0.99, fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig("farfield_comparison_all_k.png", dpi=150)

plt.show()
print("[plot_results] wrote potential_comparison.png and farfield_comparison_all_k.png")