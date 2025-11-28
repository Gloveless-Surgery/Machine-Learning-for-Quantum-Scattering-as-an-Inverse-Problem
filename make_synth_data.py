# make_synth_data.py
import os, math, argparse
from pathlib import Path
import torch
import matplotlib.pyplot as plt

# Keep CPU runs snappy on laptop
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
torch.set_num_threads(min(4, max(1, (os.cpu_count() or 4)//2)))

# --- config: dtype & device
DTYPE  = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import of physics building blocks
from pytorch_skeleton import (
    build_ball_quadrature, LSForward, YukawaCutoff,
    cplx, cadd, creal, cimag, cexp_i
)  # :contentReference[oaicite:2]{index=2}

def parse_args():
    p = argparse.ArgumentParser("Generate synthetic scattering data and save to file.")
        # grid / geometry  (denser quadrature)
    p.add_argument("--Rmax", type=float, default=7.0)
    p.add_argument("--Nr",   type=int,   default=24)
    p.add_argument("--Nct",  type=int,   default=12)
    p.add_argument("--Nphi", type=int,   default=12)
    p.add_argument("--Ntheta", type=int, default=97)
    # k sampling  (use 8 k-values, from 0.8 up to 6.0)
    p.add_argument("--kmin", type=float, default=0.8)
    p.add_argument("--kmax", type=float, default=6.0)
    p.add_argument("--K",    type=int,   default=8)
    # ground-truth potential parameters
    p.add_argument("--A",  type=float, default=4.0)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--Rc", type=float, default=3.0)
    p.add_argument("--w",  type=float, default=0.3)
    # LS iterations & chunks
    p.add_argument("--Ngen",  type=int, default=8, help="LS iterations for synthetic generation")
    p.add_argument("--chunk", type=int, default=512, help="Row-block size for kernel mat-vec")
    # noise
    p.add_argument("--snr_db", type=float, default=20.0, help="Target complex SNR (dB)")
    # output / plotting
    p.add_argument("--out",  type=str, default="data/synth_case.pt")
    p.add_argument("--quickplot", action="store_true", help="Make a quick-look plot of V(r) and |f|")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build quadrature points inside ball and observation directions
    pts, W = build_ball_quadrature(args.Rmax, args.Nr, args.Nct, args.Nphi, device=device)  # (M,3), (M,)
    theta = torch.linspace(0, math.pi, args.Ntheta, dtype=DTYPE, device=device)
    obs_dirs = torch.stack([torch.sin(theta), torch.zeros_like(theta), torch.cos(theta)], dim=-1)  # (T,3)
    k_values = torch.linspace(args.kmin, args.kmax, args.K, dtype=DTYPE, device=device).tolist() # list of K floats

    # Ground-truth potential on the nodes
    gt_pot = YukawaCutoff(learn_w=False, init=(args.A, args.mu, args.Rc, args.w)).to(device)  # fixed GT potential
    r_nodes = torch.linalg.norm(pts, dim=-1)
    V_true, true_params = gt_pot(r_nodes)

    # Differentiable LS forward (generation fidelity)
    ls = LSForward(pts, W, k_values, N_iter=args.Ngen, chunk_size=args.chunk).to(device) 

    # Generates far-field amplitudes
    with torch.no_grad():
        f_true = ls(V_true, obs_dirs)  # (K,T,2)

    # Adds complex Gaussian noise to hit requested SNR
    K, T, _ = f_true.shape
    N = K*T
    norm_f = torch.sqrt(creal(f_true).pow(2).sum() + cimag(f_true).pow(2).sum())
    sigma = (norm_f / math.sqrt(2*N)) * (10.0 ** (-args.snr_db/20))
    noise = cplx(torch.randn_like(creal(f_true))*sigma, torch.randn_like(creal(f_true))*sigma)
    f_meas = cadd(f_true, noise)

    # Saves everything needed to reproduce training & plots
    payload = {
        "Rmax": args.Rmax, "Nr": args.Nr, "Nct": args.Nct, "Nphi": args.Nphi,
        "Ntheta": args.Ntheta, "k_values": k_values, "Ngen": args.Ngen,
        "pts": pts.cpu(), "W": W.cpu(),
        "theta": theta.cpu(), "obs_dirs": obs_dirs.cpu(),
        "V_true": V_true.cpu(), "true_params": tuple(tp.item() for tp in true_params),
        "f_true": f_true.cpu(), "f_meas": f_meas.cpu(),
        "A_mu_Rc_w": (args.A, args.mu, args.Rc, args.w),
        "snr_db": args.snr_db, "chunk": args.chunk, "seed": args.seed
    }
    torch.save(payload, out_path)
    print(f"[make_synth_data] wrote {out_path.resolve()}")

    # Optional quick-look plot
    if args.quickplot:
        import numpy as np
        def mag(z): return torch.sqrt(creal(z)**2 + cimag(z)**2)
        r_sorted_idx = torch.argsort(r_nodes)
        r_sorted = r_nodes[r_sorted_idx].cpu().numpy()
        V_sorted = V_true[r_sorted_idx].cpu().numpy()
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        ax[0].plot(r_sorted, V_sorted)
        ax[0].set_title("Synthetic ground-truth potential")
        ax[0].set_xlabel("r"); ax[0].set_ylabel("V(r)")
        ki = 0 if args.K==1 else args.K-1
        ax[1].plot(theta.cpu().numpy(), mag(f_true[ki]).cpu().numpy(), label="|f_true|")
        ax[1].plot(theta.cpu().numpy(), mag(f_meas[ki]).cpu().numpy(), "--", label="|f_noisy|")
        ax[1].set_title(f"Far field at k={k_values[ki]:.2f}")
        ax[1].set_xlabel("θ"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(out_path.with_suffix(".quicklook.png"), dpi=150)
        print(f"[make_synth_data] quick plot -> {out_path.with_suffix('.quicklook.png').resolve()}")

if __name__ == "__main__":
    main()
