from preamble import *
import numpy as np
from itertools import combinations
class Tournament(torch.nn.Module):
    def __init__(self, num_classes, mode='mean'):
        super(Tournament, self).__init__()
        self.num_classes = num_classes
        self.euc_dim = num_classes - 1
        self.num_edges = self.nSimplex(num_classes)
        self.cond_threshold = 1e4
        self.eps = 1e-5
        self.n, self.d, self.e = self.num_classes, self.euc_dim, self.num_edges
        # register tensors that are not learnable so they move with .to(device)
        cevians, idx = self.selectionIndicies()
        self.register_buffer('cevians', cevians)
        self.register_buffer('idx', idx)
        crd, starts, vecs = self.coordinates()
        self.register_buffer('crd', crd)
        self.register_buffer('starts', starts)
        self.register_buffer('vecs', vecs)
        gt, perms = self.get_gt()
        self.register_buffer('gt', gt)
        self.register_buffer('perms', perms)
        self.forward_ = self.forward_mean if mode == 'mean' else self.forward_solver
        # min_logit = self.get_min_logit()
        # self.register_buffer('min_logit', min_logit)

    # def get_min_logit(self):
    #     x = torch.ones((1,self.num_edges)) *.5
    #     x[:self.num_classes - 1] = 0
    #     return self(x).min()

    def get_gt(self):
        # first we need all permutations of two class labels
        perms = torch.tensor(list(combinations(range(self.num_classes), 2)), dtype=torch.float32)
        # now we create a tensor to hold the ground truth values
        gt = torch.zeros((self.num_classes, self.num_edges), dtype=torch.float32)
        for j in range(self.num_classes):
            for i, (a, b) in enumerate(perms):
                if a == j:
                    gt[j, i] = 1.0
                elif b == j:
                    gt[j, i] = -1.0
                else:
                    gt[j, i] = 0.0
        return gt, perms

    def nSimplex(self,n):
        num_edges = torch.arange(n).sum().item()
        return num_edges

    def coordinates(self):
        n =self.euc_dim
        r2 = 2**.5
        es = torch.eye(n)/(r2)
        base = (torch.ones((n,n)) + 1/((n+1)**.5))/(n*r2)
        extra = torch.unsqueeze((torch.ones(n)/((2*(n+1))**.5)),0)
        crd = torch.cat((es-base,extra),0)
        thing = torch.triu(torch.ones((self.num_classes,self.num_classes)),1)
        idx = thing.nonzero()
        starts = crd[idx[:, 0]]
        ends = crd[idx[:, 1]]
        vecs = ends - starts
        return crd, starts, vecs

    def selectionIndicies(self):
        corners = torch.ones((self.num_classes,self.num_classes,self.num_classes))
        edges = torch.triu(torch.ones((self.num_classes,self.num_classes)),1).flatten()
        l = edges.flatten().nonzero()
        corners = corners - torch.unsqueeze(torch.eye(self.num_classes),0).repeat(self.num_classes, 1, 1)
        for i in range(self.num_classes):
            corners[i,i,:] = 0
            corners[i,:,i] = 0
        mask = corners.sum(-1) != 0
        corners = corners[mask,:].reshape(self.num_classes,-1,self.num_classes)
        edge_index = torch.nonzero(torch.abs((1-corners))) [:, -1].reshape(-1,2)
        cevians = torch.eye(self.num_classes**2)[edge_index[:,0]*self.num_classes+edge_index[:,1]].reshape((self.num_classes,self.euc_dim,self.num_classes**2))[:,:,l].reshape(self.num_classes, self.euc_dim, self.num_edges)
        combined = np.concatenate((corners, cevians),-1)
        idx = torch.arange(1,(1+combined.shape[-1]))
        out = (idx*combined).flatten()
        idx = (out[out.nonzero()].reshape(self.n,self.d,self.d) -1).to(torch.int64)
        return cevians.sum(1), idx.flatten()

    def forward_mean(self, x):
        # print("Mean is being used!")
        
        assert x.shape[-1] == self.num_edges, f"Input last dimension must be {self.num_edges}, got {x.shape[-1]}"
        chis = self.starts[None, :, :] + x[..., None] * self.vecs[None, :, :]
        means = self.cevians @ chis / (self.euc_dim)
        cevians = self.crd - means
        out = 1 - torch.linalg.norm(cevians, axis=-1)
        # ensure stable dtype for loss/backward when using AMP (return float32)
        return out.to(torch.float32)

    def _safe_solve_batch(self, mats, ones_vec=None):
        """
        Robust solve for A x = 1 (or A x = ones_vec). Works for inputs in any dtype
        by upcasting to float32 for the linear algebra, then casting the solutions
        back to the original dtype.

        Strategy:
         - compute SVD-based condition estimates in float32
         - for well-conditioned matrices use torch.linalg.solve (float32)
         - for ill-conditioned matrices use torch.linalg.pinv with rcond=1/cond_threshold
         - return solutions cast back to the input dtype
        """
        was_batched = (mats.ndim == 4)
        if was_batched:
            B, M, d, _ = mats.shape
            flat = mats.reshape(-1, d, d)
        else:
            flat = mats
            B = None
            M = flat.shape[0] if flat.ndim == 3 else 0
            d = flat.shape[1]

        device = flat.device
        orig_dtype = flat.dtype

        # prepare right-hand side (ones)
        if ones_vec is None:
            ones_vec = torch.ones((d,), device=device, dtype=orig_dtype)
        ones32 = ones_vec.to(device=device, dtype=torch.float32)

        # Use your amp_ctx (preserves your autocast behavior in the rest of training)
        with amp_ctx:
            A32 = flat.to(device=device, dtype=torch.float32)  # explicit upcast for safe linalg
            N = A32.shape[0]
            sols32 = torch.empty((N, d), device=device, dtype=torch.float32)

            # try to get condition estimates
            try:
                svals = torch.linalg.svdvals(A32)  # (N, d), float32
                s_max = svals[:, 0]
                s_min = svals[:, -1].clamp(min=float(self.eps))
                cond = s_max / s_min
            except Exception:
                # if svd fails, mark all as "bad"
                cond = torch.full((N,), float('inf'), device=device, dtype=torch.float32)

            cond_threshold = float(self.cond_threshold)

            # indices for good/bad matrices
            good_mask = cond <= cond_threshold
            bad_mask = ~good_mask

            # Solve well-conditioned matrices in a batched manner when possible
            if good_mask.any():
                good_idx = torch.nonzero(good_mask, as_tuple=False).view(-1)
                try:
                    # batched solve for the good subset
                    Agood = A32[good_idx]
                    rhs = ones32.unsqueeze(-1).expand(Agood.shape[0], d, 1)
                    sols_tmp = torch.linalg.solve(Agood, rhs).squeeze(-1)
                    sols32[good_idx] = sols_tmp
                except Exception:
                    # fallback to per-matrix solve/pinv for these
                    for ii in good_idx.tolist():
                        try:
                            sols32[ii] = torch.linalg.solve(A32[ii], ones32)
                        except Exception:
                            sols32[ii] = torch.matmul(torch.linalg.pinv(A32[ii]), ones32)

            # For ill-conditioned matrices, use pinv with rcond = 1/cond_threshold (float32)
            if bad_mask.any():
                bad_idx = torch.nonzero(bad_mask, as_tuple=False).view(-1)
                rcond = float(1.0 / max(cond_threshold, 1.0))
                # try batched pinv for the bad set (if supported)
                try:
                    Abad = A32[bad_idx]  # (K,d,d)
                    Pinv = torch.linalg.pinv(Abad, rcond=rcond)  # (K,d,d)
                    rhs = ones32.unsqueeze(-1).expand(Pinv.shape[0], d, 1)
                    sols32[bad_idx] = (Pinv @ rhs).squeeze(-1)
                except Exception:
                    # fallback per-matrix
                    for ii in bad_idx.tolist():
                        Ai = A32[ii]
                        try:
                            Pi = torch.linalg.pinv(Ai, rcond=rcond)
                            sols32[ii] = torch.matmul(Pi, ones32)
                        except Exception:
                            # last resort: tiny Tikhonov ridge on normal equations
                            lam = 1e-6
                            AtA = Ai.transpose(-2, -1) @ Ai
                            Atb = Ai.transpose(-2, -1) @ ones32
                            try:
                                sols32[ii] = torch.linalg.solve(AtA + lam * torch.eye(d, device=device, dtype=torch.float32), Atb)
                            except Exception:
                                sols32[ii] = torch.matmul(torch.linalg.pinv(Ai), ones32)

        # cast back to original dtype (so downstream sees the same dtype as mats)
        sols_out = sols32.to(device=device, dtype=orig_dtype)

        if was_batched:
            return sols_out.view(B, M, d)
        else:
            return sols_out.view(-1, d)

    def forward_solver(self, x):
        """
        x: (B, e)
        """
        # Move and cast buffers to the input dtype on the target device
        

        # Make sure idx is on the device and is long (int64)
        # print("Solver is being used!")
        assert x.ndim == 2 and x.shape[1] == self.e, f"expected (B,{self.e})"
        B = x.shape[0]
        device = x.device
        target_dtype = x.dtype  # keep ops consistent with the input/activations (handles AMP)

        crd = self.crd.to(device=device, dtype=target_dtype)
        starts = self.starts.to(device=device, dtype=target_dtype)
        vecs = self.vecs.to(device=device, dtype=target_dtype)
        # cevians used in forward_mean:
        cevians_buf = self.cevians.to(device=device, dtype=target_dtype)
        idx = self.idx.to(device=device, dtype=torch.long)

        # build chis and verts (straightforward)
        chis = starts.unsqueeze(0) + x.unsqueeze(-1) * vecs.unsqueeze(0)   # (B,e,d)
        verts = torch.cat([crd.unsqueeze(0).expand(B, -1, -1), chis], dim=1)  # (B, n+e, d)

        # gather rows by idx -> (B, n*d*d, d)
        idx_expand = idx.view(1, -1, 1).expand(B, -1, self.d)
        gathered = verts.gather(1, idx_expand)   # (B, n*d*d, d)

        # reshape into (B, n*d, d, d)
        vert_spread = gathered.view(B, self.n * self.d, self.d, self.d)

        # Solve A x = 1 for each (B, n*d) matrix -> planes (B, n*d, d)
        ones = torch.ones((self.d,), device=device)
        planes = self._safe_solve_batch(vert_spread, ones)

        # reshape to (B, n, d, d) by grouping every d rows
        planes_reshaped = planes.view(B, self.n, self.d, self.d)

        # Solve again to get intersections (B, n, d)
        intersections = self._safe_solve_batch(planes_reshaped, ones)

        cevians = crd.unsqueeze(0) - intersections    # (B, n, d)
        out = 1.0 - torch.linalg.norm(cevians, dim=-1)  # (B, n)

        return out# , intersections
    def forward(self,x):
        return self.forward_(x)

def symmetric_cross_entropy(preds, targets, reduction='mean'):
    safe_preds, safe_targets = preds.clamp(1e-7, 1 - 1e-7), targets.clamp(1e-7, 1 - 1e-7)
    # loss = -(safe_targets * safe_preds.log() + (1 - safe_targets) * (1 - safe_preds).log())
    loss = -(safe_targets * safe_preds + (1 - safe_targets) * (1 - safe_preds))
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def log_symmetric_cross_entropy(preds, targets, reduction='mean'):
    # safe_preds, safe_targets = preds.clamp(1e-7, 1 - 1e-7), targets.clamp(1e-7, 1 - 1e-7)
    safe_preds, safe_targets = preds.clamp(1e-7, 1 - 1e-7).to(torch.float32), targets.float()
    # loss = -(safe_targets * safe_preds.log() + (1 - safe_targets) * (1 - safe_preds).log())
    loss = -(safe_targets * safe_preds.log() )
    # loss = -(safe_targets * safe_preds + (1 - safe_targets) * (1 - safe_preds))
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def ioannis_symmetric_cross_entropy(preds, targets, reduction='mean'):
    safe_preds, safe_targets = preds.clamp(1e-7, 1 - 1e-7), targets.clamp(1e-7, 1 - 1e-7)
    loss = -(safe_targets * safe_preds.log() + safe_preds * safe_targets.log())
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss



class TournamentGaussianModel(nn.Module):
    def __init__(self, input_dim, num_classes, num_components=3):
        super().__init__()
        self.num_classes = num_classes
        num_games = num_classes * (num_classes -1) //2
        self.num_components = num_components
        self.num_games = num_games
        # Output mean vector
        self.mean_layer = nn.Linear(input_dim, num_games)

        # Output lower-triangular Cholesky factor
        self.cholesky_layer = nn.Linear(input_dim, num_games * num_games)
        self.cholesky_layer.bias.data = torch.eye(num_games).flatten() * 0.5
        # Fixed prior components (VaDE-style)
        self.register_buffer("prior_means", torch.randn(num_components, num_games))
        self.register_buffer("prior_covs", torch.stack([self.build_signed_prior() for _ in range(num_components)]))

    def build_signed_prior(self, gamma=0.1):
        game_pairs = torch.tensor(list(combinations(range(self.num_classes), 2)), dtype=torch.float32)
        prior_cov = torch.eye(self.num_games) * .1
        for idx1, (i1, j1) in enumerate(game_pairs):
            for idx2, (i2, j2) in enumerate(game_pairs):
                if idx1 == idx2: continue
                shared = {i1, j1}.intersection({i2, j2})
                if shared:
                    shared_player = list(shared)[0]
                    # Check roles
                    same_role = ((shared_player == i1 and shared_player == i2) or
                                (shared_player == j1 and shared_player == j2))
                    prior_cov[idx1, idx2] += gamma if same_role else -gamma
        return prior_cov

    def inference(self, x, game_pairs):
        mu = self.mean_layer(x)
        raw_cholesky = self.cholesky_layer(x)
        L = raw_cholesky.view(x.size(0), self.num_games, self.num_games)
        L = torch.tril(L).to(x.dtype)
        diag_idx = torch.arange(self.num_games)
        L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-3

        cov = torch.bmm(L, L.transpose(1, 2))  # [batch, num_games, num_games]

        probs = []
        for b in range(x.size(0)):
            p_vec = []
            for (i, j) in game_pairs:
                delta = mu[b, j] - mu[b, i]
                var_delta = cov[b, j, j] + cov[b, i, i] - 2 * cov[b, i, j]
                denom = torch.sqrt(1 + (torch.pi**2 * var_delta / 8))
                p_vec.append(torch.sigmoid(delta / denom))
            probs.append(torch.stack(p_vec))
        return torch.stack(probs)  # [batch, num_games]
    
    def forward(self, x):
        batch_size = x.size(0)

        # Mean vector
        mu = self.mean_layer(x)

        # Cholesky factor
        raw_cholesky = self.cholesky_layer(x)
        # raw_cholesky = F.tanh(raw_cholesky)
        L = raw_cholesky.view(batch_size, self.num_games, self.num_games)
        L = torch.tril(L).to(x.dtype)
        diag_idx = torch.arange(self.num_games)
        L[:, diag_idx, diag_idx] = F.softplus(L[:, diag_idx, diag_idx]) + 1e-3


        # Sample from multivariate Gaussian
        eps = torch.randn(batch_size, self.num_games, 1, device=x.device)
        sample = mu.unsqueeze(-1) + torch.bmm(L, eps)
        sample = sample.squeeze(-1)

        # Mahalanobis distance
        diff = sample - mu  # shape: [batch_size, num_games]
        L_ = L.float()
        diff_ = diff.float()
        whitened = torch.linalg.solve_triangular(L_, diff_.unsqueeze(-1), upper=False).squeeze(-1)  # shape: [batch_size, num_games]
        mahalanobis = torch.sum(whitened ** 2, dim=-1)  # shape: [batch_size]
        mahalanobis = torch.log1p(mahalanobis)

        # Regularization loss

        reg_loss = 0.0
        for i in range(batch_size):
            cov = L[i] @ L[i].T + 1e-5 * torch.eye(self.num_games, device=x.device)
            sign, logdet = torch.slogdet(cov)
            trace_val = torch.trace(cov)
            trace_term = trace_val / self.num_games
            # print(f"Batch {i}: trace={trace_val:.4f}, logdet={logdet:.4f}, sign={sign}")
            if sign <= 0:
                print(f"Warning: Non-PD matrix at batch {i}")
                logdet_term = -100.0  # or some penalty / self.num_games
            else:
                logdet_term = 1e-3 * logdet / self.num_games
            # reg_loss += 0.5 * (trace_term - logdet )#- self.num_games)
            # reg_loss += trace_term + logdet_term
            reg_loss += 0.5 * ((trace_val / self.num_games) - (logdet) - 1)
            penalty = torch.sum(torch.relu(0.1 - torch.diagonal(L[i])))
            reg_loss += penalty


        # VaDE-style KL divergence
        kl_div = 0.0
        for i in range(batch_size):
            cov_post = L[i] @ L[i].T + 1e-3 * torch.eye(self.num_games, device=x.device)
            mu_post = mu[i]

            kl_vals = []
            for k in range(self.num_components):
                mu_prior = self.prior_means[k]
                cov_prior = self.prior_covs[k] + 1e-3 * torch.eye(self.num_games, device=x.device)
                inv_cov_prior = torch.inverse(cov_prior)

                trace_term = torch.trace(inv_cov_prior @ cov_post)
                diff = mu_prior - mu_post
                quad_term = diff @ inv_cov_prior @ diff
                log_det_ratio = torch.logdet(cov_prior) - torch.logdet(cov_post)
                kl = 0.5 * (trace_term + quad_term - self.num_games + log_det_ratio)
                kl_vals.append(kl)

            kl_div += torch.min(torch.stack(kl_vals))

        return sample, mahalanobis, reg_loss, kl_div

def main():
    t = Tournament(10)
    # x = torch.rand((10,t.num_edges))
    # y = t(x)
    # print(y)
    # print(t.gt)
    # print(t.gt.shape)
    # print(t.min_logit)
    # Number of classes and tournament games
    K = 10
    num_games = K * (K - 1) // 2
    batch_size = 4
    feature_dim = 1000

    # Synthetic input features
    x = torch.randn(batch_size, feature_dim)
    x = F.sigmoid(x)
    # Instantiate and run model
    model = TournamentGaussianModel(feature_dim, num_games)
    sample, mahalanobis, reg_loss, kl_div = model(x)

    # Dummy tournament loss
    prediction_vector = t(sample)
    # target = F.one_hot(torch.randn_like(prediction_vector).argmax(-1), K)
    target = torch.randn_like(prediction_vector).argmax(-1)
    # print(target.shape, prediction_vector.shape)
    tournament_loss = F.cross_entropy(prediction_vector, target)

    alpha = 0.01
    beta = 0.001
    gamma = 0.005

    total_loss = tournament_loss + alpha * mahalanobis.mean() + beta * reg_loss + gamma * kl_div

    print("Total loss:", total_loss.item())
    print("Tournament loss:", tournament_loss.item())
    print("Mahalanobis mean:", mahalanobis.mean().item())
    print("Regularization loss:", reg_loss.item())
    print("KL divergence:", kl_div.item())
    print("Sample mean:", sample.mean().item())
    print("Sample std:", sample.std().item())


if __name__ == "__main__":
    main()



