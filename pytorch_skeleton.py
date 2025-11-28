# pip install torch numpy matplotlib
import math, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32  # to keep real/imag pairs in float32

# ---------- Complex helpers (real/imag as last dim of size 2) ----------
def cplx(real, imag): return torch.stack([real, imag], dim=-1) # complex number
def creal(z): return z[..., 0] # real part
def cimag(z): return z[..., 1] # imag part
def cconj(z): return torch.stack([creal(z), -cimag(z)], dim=-1) # complex conjugate
def cadd(z,w): return z+w
def csub(z,w): return z-w
def cmul(z,w):
    ar, ai = creal(z), cimag(z) # a,b real/imag
    br, bi = creal(w), cimag(w) # b,d real/imag
    return cplx(ar*br - ai*bi, ar*bi + ai*br) 
def cexp_i(phi):  # e^{i phi}
    return cplx(torch.cos(phi), torch.sin(phi)) 
def csum(z, dim):  return torch.stack([creal(z).sum(dim), cimag(z).sum(dim)], dim=-1)

# ---------- Quadrature grid inside ball ----------
def gauss_legendre(n, a, b): # n pts on [a,b], return_x_and_weights
    x, w = np.polynomial.legendre.leggauss(n) # on [-1,1]
    x = 0.5*(b-a)*x + 0.5*(b+a) 
    w = 0.5*(b-a)*w 
    return torch.tensor(x, dtype=DTYPE), torch.tensor(w, dtype=DTYPE) # nodes and weights to tensor 

def build_ball_quadrature(Rmax=7.0, Nr=24, Nct=12, Nphi=12, device=device):   # build pts, weights, Nct = polar nodes, Nphi = azimuthal nodes, Rmax = ball radius
    # Radial, polar, azimuthal nodes and weights
    r, wr = gauss_legendre(Nr, 0.0, Rmax)                                   # r and weights in [0,Rmax]
    ct, wt = gauss_legendre(Nct, -1.0, 1.0)                                 # ct = cos(theta'), weights in [-1,1]
    phi = torch.linspace(0, 2*math.pi, Nphi+1, dtype=DTYPE)[:-1]            # exclude endpoint, 0 to 2pi, size Nphi
    wphi = (2*math.pi)/Nphi * torch.ones_like(phi)                          # uniform weights in phi

    # product grid in spherical coords -> Cartesian 
    r3  = r[:, None, None]                                                  # r, (Nr,1,1)
    ct3 = ct[None, :, None]                                                 # cos(theta'), (1,Nct,1)
    st3 = torch.sqrt(torch.clamp(1 - ct3**2, min=0.0))                      # sin(theta'), (1,Nct,1)
    phi3 = phi[None, None, :]                                               # phi, (1,1,Nphi)

    x = (r3 * st3 * torch.cos(phi3)).reshape(-1)                            # x-coord, y-coord, z-coord
    y = (r3 * st3 * torch.sin(phi3)).reshape(-1)                            # y-coord
    # replicate z across the φ dimension so shapes match
    z = (r3 * ct3).expand(-1, -1, phi.numel()).reshape(-1)
    # alternatively: z = (r3*ct3*torch.ones_like(phi3)).reshape(-1)

    pts = torch.stack([x, y, z], dim=-1).to(device)                         # (M,3) points in ball, M=Nr*Nct*Nphi

    W = (wr[:, None, None] * (r3**2) * wt[None, :, None] * wphi[None, None, :]).reshape(-1).to(device) # (M,) weights with Jacobian r^2 sin(theta') included
    return pts, W

# ---------- Potential module ----------
class YukawaCutoff(nn.Module):
    def __init__(self, learn_w, init=(4.0,1.0,3.0,0.35)):
        super().__init__()
        A0, mu0, Rc0, w0 = init
        # Unconstrained parameters -> positive via softplus
        self.alpha = nn.Parameter(torch.tensor(math.log(math.exp(A0)-1), dtype=DTYPE))  # amplitude
        self.beta  = nn.Parameter(torch.tensor(math.log(math.exp(mu0)-1), dtype=DTYPE)) # decay rate
        self.gamma = nn.Parameter(torch.tensor(math.log(math.exp(Rc0)-1), dtype=DTYPE)) # cutoff radius
        self.learn_w = learn_w
        if learn_w:
            self.delta = nn.Parameter(torch.tensor(math.log(math.exp(w0)-1), dtype=DTYPE)) # cutoff smoothness
        else:
            self.register_buffer('w_fixed', torch.tensor(w0, dtype=DTYPE)) 

    def forward(self, r):
        A = F.softplus(self.alpha)
        mu = F.softplus(self.beta)
        Rc = F.softplus(self.gamma)
        w  = (F.softplus(self.delta) if self.learn_w else self.w_fixed).clamp_min(1e-3)
        # smooth cutoff
        chi = 0.5*(1.0 - torch.tanh((r - Rc)/w))
        # avoid division by zero at r=0 by safe r
        rs = torch.clamp(r, min=1e-6)
        V  = A*torch.exp(-mu*rs)/rs * chi
        return V, (A,mu,Rc,w)

# ---------- Green's matrix per k ---------- (This is is not used in the training loop, but kept for reference)
def helmholtz_green_matrix(pts, k): 
    # reference implementation of G_k(r_i,r_j) = exp(i k |r_i - r_j|)/(4π |r_i - r_j|)
    r2 = torch.sum(pts**2, dim=1, keepdim=True)
    D2 = r2 + r2.T - 2.0*(pts @ pts.T)
    D = torch.sqrt(torch.clamp(D2, min=0.0))
    phase = k*D
    G = cexp_i(phase)  # (M,M,2)
    G = torch.stack([creal(G)/(4*math.pi), cimag(G)/(4*math.pi)], dim=-1)
    invD = torch.zeros_like(D)
    mask = D > 0
    invD[mask] = 1.0 / torch.clamp(D[mask], min=1e-3)   # slightly larger floor
    # kill self-interaction
    invD.fill_diagonal_(0.0)
    G = cmul(G, cplx(invD, torch.zeros_like(invD))) # (M,M,2)
    return G

# ---------- LS forward (N iterations), memory efficient ----------
class LSForward(nn.Module):
    def __init__(self, pts, weights, k_values, N_iter=5, incident_dir=None, chunk_size=2048):
        # pts: (M,3), weights: (M,), k_values: list of K floats
        super().__init__()
        # geometry
        self.register_buffer('pts', pts)                                    # (M,3)
        self.register_buffer('w', weights)                                  # (M,)
        kvals = torch.tensor(k_values, dtype=pts.dtype, device=pts.device)  # list of wavenumbers (K,)
        self.register_buffer('kvals', kvals)                                # list of wavenumbers (K,)
        self.N = N_iter                                                     # number of LS iterations

        if incident_dir is None: # default
            incident_dir = torch.tensor([0., 0., 1.], dtype=pts.dtype, device=pts.device) # z-direction
        dhat = incident_dir / incident_dir.norm() # unit vector
        self.register_buffer('dhat', dhat) 

        # precompute for distances and plane waves
        r2 = torch.sum(pts**2, dim=1)            # |r|^2 for all points
        self.register_buffer('r2', r2)           # (M,)
        dot_d = pts @ dhat                       # r·dhat (plane-wave phase anchor)
        self.register_buffer('dot_d', dot_d)     # (M,)

        # constants
        self.c_const   = 1.0 / (4.0 * math.pi) # 1/(4π) for Green's function, later referred to as the constant c
        self.chunk_size = int(chunk_size)

    # e^{i k d·r} at all nodes
    def plane_wave(self, k):
        return cexp_i(k * self.dot_d)            # (M,2)

    # V(r) * weights
    def V_times_weights(self, V):
        return V * self.w                         # (M,)

    # y = G_k @ vec, with G_k(r_i,r_j) = exp(i k |r_i - r_j|)/(4π |r_i - r_j|)
    def _apply_kernel(self, k, vec):
        M = self.pts.shape[0]
        u = creal(vec)                            # (M,)
        v = cimag(vec)                            # (M,)
        y_re = torch.zeros(M, dtype=u.dtype, device=u.device)
        y_im = torch.zeros(M, dtype=u.dtype, device=u.device)

        for s in range(0, M, self.chunk_size):
            e = min(M, s + self.chunk_size)
            pts_blk = self.pts[s:e]               # (B,3)

            # D^2 = |ri|^2 + |rj|^2 - 2 ri·rj, for this row-block
            D2_blk = self.r2[s:e].unsqueeze(1) + self.r2.unsqueeze(0) - 2.0 * (pts_blk @ self.pts.T) # computes D^2
            D_blk  = torch.sqrt(torch.clamp(D2_blk, min=1e-12))
            invD   = 1.0 / torch.clamp(D_blk, min=1e-6)  # computes 1/|ri - rj|, safer floor than 1e-3

            # zero self-interaction on the diagonal entries that fall in this block
            row_idx = torch.arange(s, e, device=u.device).unsqueeze(1)
            col_idx = torch.arange(0, M, device=u.device).unsqueeze(0)
            invD = invD.masked_fill(row_idx == col_idx, 0.0)

            # phase = k * |ri - rj|
            phase  = k * D_blk
            cos_p  = torch.cos(phase)
            sin_p  = torch.sin(phase)

            # Gr = c * invD * cos_p ; Gi = c * invD * sin_p, real and imag parts of G_k
            scale = self.c_const * invD # scaling factor
            Gr = scale * cos_p # real part of G
            Gi = scale * sin_p # imag part of G

            # y_re = Gr@u - Gi@v ; y_im = Gr@v + Gi@u
            y_re[s:e] = Gr @ u - Gi @ v
            y_im[s:e] = Gr @ v + Gi @ u

            # free block tensors to save memory 
            del pts_blk, D2_blk, D_blk, invD, phase, cos_p, sin_p, scale, Gr, Gi

        return cplx(y_re, y_im)                   # (M,2)

    # N iterations of LS: psi_N = psi_0 + G_k V psi_{N-1} (ψ=ψ_0​+Gk​(Vψ))
    def ls_iterate(self, V, k):
        psi0 = self.plane_wave(k)                 # (M,2)
        psi  = psi0
        VW   = self.V_times_weights(V)            # computes quadrature-weighted potential (M,)
        for _ in range(self.N):
            Kpsi_arg = psi * VW[:, None]          # computes V*psi with broadcasting (M,2) = diag(VW) @ psi
            Kpsi = self._apply_kernel(k, Kpsi_arg)# applies kernel operation, returns G_k V psi_{N-1} (M,2)
            psi  = cadd(psi0, Kpsi)
        return psi                                # (M,2)

    def forward(self, V, obs_dirs):               # obs_dirs: (T,3)
        # geometry for far-field phases (independent of k once per call)
        Q = self.pts @ obs_dirs.T                 # (M,T) , Q_ij = r_i · dhat_j

        f_out = []
        for k in self.kvals:                      # loop over wavenumbers
            psiN   = self.ls_iterate(V, k)        # solves the LS equation for this k in N interations, (M,2)
            # far-field integral per observation direction
            phases = cexp_i(-k * Q)               # builds the phase factors e^{-i k r_i · dhat_j}, (M,T,2) (real/imag channels)
            # integrand: (V(r_i) * psiN(r_i)) * e^{-i k r_i · dhat_j}
            integrand = cmul(cmul(cplx(V, torch.zeros_like(V)).unsqueeze(1), psiN.unsqueeze(1)), phases)  # computes (V(r_i​)ψ_N​(r_i​))e^{-i k r_i · dhat_j}
            fw = csum(integrand * self.w[:, None, None], dim=0)  # numerical approximation of far-field integral, (T,2)
            const = cplx(torch.tensor(-self.c_const, dtype=V.dtype, device=V.device), # -1/(4π)) + 0i as (2,)
                         torch.tensor(0.0,          dtype=V.dtype, device=V.device))
            f_out.append(cmul(const, fw))         # scattered far-field amplitude for this k, (T,2)

        return torch.stack(f_out, dim=0)          # (K,T,2) 