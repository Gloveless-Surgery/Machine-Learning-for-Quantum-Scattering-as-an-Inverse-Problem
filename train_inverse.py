# train_inverse.py
import os, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# laptop-friendly threading
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
torch.set_num_threads(min(4, max(1, (os.cpu_count() or 4)//2)))

DTYPE  = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pytorch_skeleton import (  # reuse your physics & complex helpers
    build_ball_quadrature, LSForward, YukawaCutoff,
    cplx, cadd, creal, cimag
) 

# ---------- PINN wrapper ----------
class ParametricPINN(nn.Module):
    def __init__(self, potential_module, ls_layer):
        super().__init__()
        self.pot = potential_module
        self.ls  = ls_layer
    def forward(self, r_nodes, obs_dirs):
        r = torch.linalg.norm(r_nodes, dim=-1)
        V, params = self.pot(r)
        f_pred = self.ls(V, obs_dirs)  # (K,T,2)
        if not torch.isfinite(f_pred).all():
            raise FloatingPointError("Non-finite f_pred detected.")
        return f_pred, V, params
# (Original forward and loss design come from here.)

# ---------- load synthetic data (created by make_synth_data.py) ----------
data_path = Path("data/synth_case.pt")
ds = torch.load(data_path, map_location=device)
pts      = ds["pts"].to(device)
W        = ds["W"].to(device)
k_values_full = [float(k) for k in ds["k_values"]]
theta    = ds["theta"].to(device)
obs_dirs = ds["obs_dirs"].to(device)
V_true   = ds["V_true"].to(device)          # for evaluation/plots
f_meas_full = ds["f_meas"].to(device)       # all k's from the dataset
true_params = ds["true_params"]

# -Selecting only large k's (k > 2.0) for training
k_tensor     = torch.tensor(k_values_full, dtype=DTYPE, device=device)  # (K_full,)
high_k_mask  = k_tensor > 2.0                                          # boolean mask
high_k_idx   = torch.nonzero(high_k_mask, as_tuple=False).squeeze(-1)  # indices of k>2

# these are the k-values and measurements actually used in training
k_values = [k_values_full[i] for i in high_k_idx.cpu().tolist()]       # list of k>2
f_meas   = f_meas_full[high_k_idx]                                     # (K_train, T, 2)

print("Training on", len(k_values), "high-k values:",
      ", ".join(f"{k:.2f}" for k in k_values))

# Precomputes per-k weights so each wavenumber contributes roughly equally on the high k subset
with torch.no_grad():
    mag_meas   = torch.sqrt(creal(f_meas)**2 + cimag(f_meas)**2)       # |f_meas|(K_train,T)
    mean_mag_k = mag_meas.mean(dim=1, keepdim=True)                    # (K_train,1)
    weight_k   = 1.0 / (mean_mag_k + 1e-6)                             # (K_train,1)


# ---------- instantiate model (small N, small chunk) ----------
# starting with a cheap LS depth (N=3), will be increased during training
ls_layer = LSForward(pts, W, k_values, N_iter=3, chunk_size=2048).to(device)
pot_module = YukawaCutoff(learn_w=True, init=(2.0,0.8,2.0,0.35)).to(device)
model = ParametricPINN(pot_module, ls_layer).to(device)

#opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9,0.99), weight_decay=1e-6)
#num_epochs = 120

# slightly bolder step size; still safe with 4 params + grad clipping
opt = torch.optim.Adam(model.parameters(), lr=3e-2, betas=(0.9,0.99), weight_decay=1e-6)

# running a bit longer, decay LR partway through
num_epochs = 200

# step scheduler: after 160 epochs, drop lr by a factor 0.3
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=160, gamma=0.3)

# ---------- training ----------
best_loss = float("inf")
Path("ckpt").mkdir(exist_ok=True, parents=True)
for epoch in range(num_epochs):
    if epoch == 40:                                         # tiny curriculum on LS iterations after 80 epochs
        model.ls.N = 5                                      # increase LS iterations to 5, higher fidelity
    if epoch == 80:
        model.ls.N = 8                                      # increase LS iterations to 8, higher fidelity

    opt.zero_grad()
    f_pred, V_pred, params = model(pts, obs_dirs)

    # ----- weighted complex L2 misfit over k,θ -----
    # apply per-k weights so low-k (large |f|) doesn't dominate, this results in a weighted misfit
    # Explanation: Each k-value contributes equally to the loss, preventing dominance by low-k values with larger amplitudes. 
    # The Born approximation tends to be less accurate at lower k (respectively more accurate at higher k), 
    # so this weighting helps the model learn better across the spectrum by discounting the low-k bias.
    # residuals
    res_re = creal(f_pred) - creal(f_meas)   # (K,T)                For each k and theta, compute real part difference
    res_im = cimag(f_pred) - cimag(f_meas)   # (K,T)                For each k and theta, compute imaginary part difference
    res2   = res_re**2 + res_im**2           # (K,T)                Squared magnitude of difference  
    # weighted mean over k,θ
    data_misfit = (res2 * weight_k).mean() 


    # light smoothness on V(r) along sorted radii
    r = torch.linalg.norm(pts, dim=-1)                      # compute radii of points
    idx = torch.argsort(r)                                  # indices to sort radii
    dV = V_pred[idx][1:] - V_pred[idx][:-1]                 # finite diff along radius, measures smoothness
    smooth = 1e-4 * torch.mean(dV**2)                       # smoothness regularizer, weight 1e-4

    loss = data_misfit + smooth
    loss.backward() # backpropagate loss
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
    opt.step() # gradient step

    if (epoch+1) % 20 == 0:                                 # print every 20 epochs
        A, mu, Rc, w = params
        print(f"Epoch {epoch+1:03d}  loss={loss.item():.3e}  "
              f"A={A.item():.3f}  mu={mu.item():.3f}  Rc={Rc.item():.3f}")

    # save best checkpoint
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save({
            "pot_state": pot_module.state_dict(),
            "learned_params": tuple(float(x.item()) for x in params),
            "loss": best_loss,
            "k_values": k_values,
            "config": {"N_eval": max(3, model.ls.N)}
        }, Path("ckpt/best.pt"))

print(f"[train_inverse] best loss = {best_loss:.3e}  → ckpt/best.pt saved")
