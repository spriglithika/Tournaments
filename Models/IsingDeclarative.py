from preamble import *

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _sym(X: torch.Tensor) -> torch.Tensor:
    """Symmetrize a square matrix (or batch thereof)."""
    return 0.5 * (X + X.transpose(-1, -2))

def _tanh_sym(X_raw: torch.Tensor, s_max: float = 1.0) -> torch.Tensor:
    """Apply symmetrize -> tanh (elementwise) -> (optional) scale."""
    Xs = _sym(X_raw)
    return s_max * torch.tanh(Xs)

def _stiefel_project_polar(V: torch.Tensor) -> torch.Tensor:
    r"""
    Nearest Stiefel projection (orthonormal columns) via SVD/polar factor.

    Given V ∈ R^{E×d}, returns Q = argmin_{Q^T Q=I} ||Q - V||_F.
    Implementation: V = U Σ W^T (thin SVD), Q := U W^T.
    (Orthogonal Procrustes / polar factor)  [Schönemann 1966].  [5](https://cran.r-project.org/web//packages//BradleyTerry2/vignettes/BradleyTerry.html)[6](https://web.stanford.edu/class/archive/stats/stats200/stats200.1172/Lecture24.pdf)
    """
    # Use thin SVD; handles rank-deficiency gracefully.
    U, _, Vh = torch.linalg.svd(V.to(linalg_device), full_matrices=False)
    Q = U.to(device) @ Vh.to(device)
    return Q

def _stiefel_tangent_pullback(GQ: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    r"""
    Adjoint of the Stiefel projection at Q (Riemannian gradient on Stiefel):

        grad_euclid -> grad_riem = GQ - Q sym(Q^T GQ).

    This is the orthogonal projection to the tangent space T_Q St(E,d).  [3](https://en.wikipedia.org/wiki/Ising_model)[4](https://lampz.tugraz.at/~hadley/ss2/magnetism/mft_2.php)
    """
    return GQ - Q @ _sym(Q.transpose(-1, -2) @ GQ)

def _apply_J(A: torch.Tensor, M: torch.Tensor,
             Q: torch.Tensor, S: torch.Tensor,
             x: torch.Tensor) -> torch.Tensor:
    """
    J·x = A M A^T x + Q S Q^T x ; avoids forming J explicitly.
    """
    return A @ (M @ (A.transpose(-1, -2) @ x)) + Q @ (S @ (Q.transpose(-1, -2) @ x))

def _mf_fixed_point(h: torch.Tensor, Jmv, T: float,
                    max_iter: int = 50, tol: float = 1e-6, alpha: float = 0.25) -> torch.Tensor:
    """
    Mean-field fixed-point iteration:
        m_{t+1} = (1-α) m_t + α tanh((h + J m_t)/T)
    Stops when ||Δm||_∞ < tol.
    """
    m = torch.tanh(h / T)
    for _ in range(max_iter):
        m_prev = m
        m = (1 - alpha) * m_prev + alpha * torch.tanh((h + Jmv(m_prev)) / T)
        if (m - m_prev).abs().max().item() < tol:
            break
    return m

def _cg_solve(Aop, b: torch.Tensor, iters: int = 100, tol: float = 1e-6) -> torch.Tensor:
    """
    Conjugate Gradient for SPD system A x = b, where Aop(v) = A·v.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs_old = (r * r).sum()
    for _ in range(iters):
        Ap = Aop(p)
        denom = (p * Ap).sum() + 1e-12
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if torch.sqrt(rs_new) < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

# -----------------------------------------------------------------------------
# Declarative MF function with structured J and Stiefel Q
# -----------------------------------------------------------------------------

class IsingMFStructuredWithQ(torch.autograd.Function):
    """
    Batched declarative MF with structured J using precomputed Q.

    Inputs (positional):
        h        : (E,) or (B,E)
        A        : (E,K)
        AtA_inv  : (K,K)
        M_raw    : (K,K)   (param)
        S_raw    : (d,d)   (param)
        V_raw    : (E,d)   (param)
        Q        : (E,d)   (precomputed)
        T, s_max, max_iter, alpha, tol, cg_max, cg_tol, damp : scalars
    """

    @staticmethod
    def forward(ctx,
                h, A, AtA_inv, M_raw, S_raw, V_raw, Q,
                T, s_max, max_iter, alpha, tol, cg_max, cg_tol, damp):

        # Shapes and broadcast
        batched = (h.ndim == 2)
        if not batched:
            h = h.unsqueeze(0)  # (1,E)
        B, E = h.shape

        # Blocks
        M = _tanh_sym(M_raw, s_max=1.0)           # (K,K)
        S = _tanh_sym(S_raw, s_max=s_max)         # (d,d)

        # Batched Jmv: X -> X*A @ M @ A^T + X*Q @ S @ Q^T
        def Jmv_batch(X):                         # X: (B,E)
            return ( (X @ A) @ M ) @ A.T + ( (X @ Q) @ S ) @ Q.T

        # MF fixed-point (batched)
        with torch.no_grad():
            m = torch.tanh(h / T)
            for _ in range(max_iter):
                m_prev = m
                m = (1 - alpha) * m_prev + alpha * torch.tanh((h + Jmv_batch(m_prev)) / T)
                if (m - m_prev).abs().max().item() < tol:
                    break
        m_star = m  # (B,E)

        # Save for backward
        ctx.save_for_backward(m_star, A, AtA_inv, M_raw, S_raw, V_raw, Q)
        ctx.T = float(T); ctx.s_max = float(s_max)
        ctx.cg_max = int(cg_max); ctx.cg_tol = float(cg_tol); ctx.damp = float(damp)

        return m_star.squeeze(0) if not batched else m_star

    @staticmethod
    def backward(ctx, grad_out):
        m_star, A, AtA_inv, M_raw, S_raw, V_raw, Q = ctx.saved_tensors
        T, cg_max, cg_tol, damp = ctx.T, ctx.cg_max, ctx.cg_tol, ctx.damp

        # Ensure batched shapes
        batched = (grad_out.ndim == 2)
        if not batched:
            grad_out = grad_out.unsqueeze(0)       # (1,E)
            m_star   = m_star.unsqueeze(0)         # (1,E)
        B, E = grad_out.shape
        K = A.shape[1]; d = V_raw.shape[1]

        # Rebuild blocks
        with torch.no_grad():
            M = _tanh_sym(M_raw, s_max=1.0)
            S = _tanh_sym(S_raw, s_max=ctx.s_max)

        # Batched Jmv
        def Jmv_batch(X):
            return ( (X @ A) @ M ) @ A.T + ( (X @ Q) @ S ) @ Q.T

        # Batched Hmv
        inv_var = 1.0 / (1.0 - m_star.clamp(-0.999999, 0.999999).pow(2))   # (B,E)
        def Hmv_batch(U):   # (B,E)
            return -Jmv_batch(U) + T * inv_var * U

        # Batched (H + damp I) w = v via vectorized CG
        v = grad_out.clone()                               # (B,E)
        x = torch.zeros_like(v)                            # (B,E)
        r = v.clone()                                      # (B,E)
        p = r.clone()
        rs_old = (r * r).sum(dim=1)                        # (B,)

        for _ in range(cg_max):
            Ap = Hmv_batch(p) + damp * p                   # (B,E)
            denom = (p * Ap).sum(dim=1) + 1e-12            # (B,)
            alpha = rs_old / denom                          # (B,)
            x = x + alpha.unsqueeze(1) * p
            r = r - alpha.unsqueeze(1) * Ap
            rs_new = (r * r).sum(dim=1)                    # (B,)
            # stop if ALL samples converged
            if torch.sqrt(rs_new.max()) < cg_tol:
                break
            beta = rs_new / (rs_old + 1e-12)
            p = r + beta.unsqueeze(1) * p
            rs_old = rs_new

        W = x                                             # (B,E), i.e., w_b stacked
        dL_dh = W.squeeze(0) if not batched else W

        # ---------- Batch-accumulated parameter grads ----------
        # dM = 0.5 * A^T ( W^T M* + M*^T W ) A
        WTMs = W.transpose(0,1) @ m_star                  # (E,E)
        MsTW  = m_star.transpose(0,1) @ W                 # (E,E)
        dM = 0.5 * (A.T @ (WTMs + MsTW) @ A)              # (K,K)

        # dS = 0.5 * [ (WQ)^T (M*Q) + (M*Q)^T (WQ) ]
        WQ  = W @ Q                                       # (B,d)
        MQ  = m_star @ Q                                  # (B,d)
        dS = 0.5 * (WQ.transpose(0,1) @ MQ + MQ.transpose(0,1) @ WQ)  # (d,d)

        # G_Q = W^T ( (M*Q) S ) + M*^T ( (WQ) S )   -> (E,d)
        MQS = MQ @ S                                      # (B,d)
        WQS = WQ @ S                                      # (B,d)
        GQ  = W.transpose(0,1) @ MQS + m_star.transpose(0,1) @ WQS     # (E,d)

        # Pullback through Stiefel projection to V_proj
        G_Vproj = _stiefel_tangent_pullback(GQ, Q)        # (E,d)

        # Low-rank adjoint to V_raw: dV = G_Vproj - A (AtA_inv (A^T G_Vproj))
        ATG = A.T @ G_Vproj                               # (K,d)
        dV  = G_Vproj - A @ (AtA_inv @ ATG)               # (E,d)

        # Chain rules for raw params
        sech2_M = 1.0 - torch.tanh(_sym(M_raw))**2
        dM_raw  = _sym(sech2_M * dM)

        sech2_S = 1.0 - torch.tanh(_sym(S_raw))**2
        dS_raw  = _sym(sech2_S * dS) * ctx.s_max

        # No grads for A, AtA_inv, Q
        dA = None; dAtAinv = None; dQ = None

        # Return one slot per forward input (15)
        return (dL_dh,
                dA, dAtAinv,
                dM_raw, dS_raw, dV, dQ,
                None, None, None, None, None, None, None, None)
# -----------------------------------------------------------------------------
# Convenience wrapper nn.Module
# -----------------------------------------------------------------------------

def _sym(X): return 0.5 * (X + X.T)

@torch.jit.script
def declarative_forward(h: torch.Tensor, A: torch.Tensor, AtA_inv: torch.Tensor, M_raw: torch.Tensor, S_raw: torch.Tensor, V_raw: torch.Tensor, Q: torch.Tensor, T: float = 1.0, s_max: float = 0.85, max_iter: int = 50, alpha: float = 0.25, tol: float = 1e-6, cg_max: int = 100, cg_tol: float = 1e-6, damp: float = 1e-3):
    return IsingMFStructuredWithQ.apply(h, A, AtA_inv, M_raw, S_raw, V_raw, Q, T, s_max, max_iter, alpha, tol, cg_max, cg_tol, damp)

class IsingJBlock(nn.Module):
    def __init__(self, A: torch.Tensor, d: int, s_max: float = 0.85):
        super().__init__()
        E, K = A.shape
        assert d <= E
        self.E, self.K, self.d = E, K, d
        self.s_max = float(s_max)
        self.device = device
        self.register_buffer("A", A.clone())
        self.register_buffer("AtA_inv", torch.linalg.inv(self.A.T @ self.A))
        self.M_raw = nn.Parameter(torch.zeros(K, K))
        self.S_raw = nn.Parameter(torch.zeros(d, d))
        self.V_raw = nn.Parameter(0.05 * torch.randn(E, d))

    def _lowrank_project(self, V: torch.Tensor) -> torch.Tensor:
        AV = self.A.T @ V               # (K,d)
        return V - self.A @ (self.AtA_inv @ AV)

    def _compute_Q(self) -> torch.Tensor:
        V_proj = self._lowrank_project(self.V_raw).to(linalg_device)
        Q, _ = torch.linalg.qr(V_proj, mode='reduced')
        return Q.to(self.device)

    def forward(self, h, *,
                T: float = 1.0,
                max_iter: int = 50,
                alpha: float = 0.25,
                tol: float = 1e-6,
                cg_max: int = 100,
                cg_tol: float = 1e-6,
                damp: float = 1e-3):
        Q = self._compute_Q()
        return declarative_forward(h, self.A, self.AtA_inv, self.M_raw, self.S_raw, self.V_raw, Q,
                                  T=T, s_max=self.s_max,
                                  max_iter=max_iter, alpha=alpha, tol=tol,
                                  cg_max=cg_max, cg_tol=cg_tol,
                                  damp=damp)
        # return IsingMFStructuredWithQ.apply(h, self.A, self.AtA_inv, self.M_raw, self.S_raw, self.V_raw, self._compute_Q(), T, self.s_max, max_iter, alpha, tol, cg_max, cg_tol, damp)
    @torch.no_grad()
    def current_blocks(self):
        M = _tanh_sym(self.M_raw, s_max=1.0)
        S = _tanh_sym(self.S_raw, s_max=self.s_max)
        Q = self._compute_Q()
        return M, S, Q

    @torch.no_grad()
    def J_mv(self, x: torch.Tensor):
        M, S, Q = self.current_blocks()
        return _apply_J(self.A, M, Q, S, x)