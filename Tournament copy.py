import torch
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
        self.n, self.d, self.e = self.num_classes.item(), self.euc_dim.item(), self.num_edges
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
        assert x.shape[-1] == self.num_edges, f"Input last dimension must be {self.num_edges}, got {x.shape[-1]}"
        chis = self.starts[None, :, :] + x[..., None] * self.vecs[None, :, :]
        means = self.cevians @ chis / (self.euc_dim)
        cevians = self.crd - means
        out = 1 - torch.linalg.norm(cevians,axis=-1)
        return out #, means

    def _batch_solve(self, mats):
        """
        mats: (B, K, d, d) -> solve for x in A x = 1 (ones vector)
        returns: (B, K, d)
        """
        B, K, d, _ = mats.shape
        flat = mats.reshape(-1, d, d)               # (B*K, d, d)
        device = flat.device
        dtype = flat.dtype
        homog = torch.ones((d,), device=device, dtype=dtype)

        # Try batched solve where possible; compute condition numbers to decide fallback
        # Compute singular values and cond numbers (batched)
        sv = torch.linalg.svdvals(flat)              # (B*K, d)
        cond = (sv[:, 0] / sv[:, -1].clamp(min=1e-12))
        use_solve = cond < self.cond_threshold

        sols = torch.empty((flat.shape[0], d), device=device, dtype=dtype)

        # Attempt to solve for the subset that looks well-conditioned
        if use_solve.any():
            A_good = flat[use_solve]
            b_good = homog.unsqueeze(0).expand(A_good.shape[0], -1)
            try:
                sols[use_solve] = torch.linalg.solve(A_good, b_good)
            except RuntimeError:
                # if solve fails for some reason, fall back to pinv for those
                pinv_good = torch.linalg.pinv(A_good)
                sols[use_solve] = torch.matmul(pinv_good.to(dtype), b_good.unsqueeze(-1)).squeeze(-1).to(dtype)
        # For the rest (ill-conditioned), use pinv with jitter
        if (~use_solve).any():
            A_bad = flat[~use_solve]
            jitter = self.eps * torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
            A_bad_j = A_bad + jitter
            pinv_bad = torch.linalg.pinv(A_bad_j)
            sols[~use_solve] = torch.matmul(pinv_bad.to(dtype), homog.unsqueeze(0).unsqueeze(-1)).squeeze(-1).to(dtype)

        return sols.view(B, K, d)

    def forward_solver(self, x):
        """
        x: (B, e)
        returns: out (B, n), intersections (B, n, d)
        """
        # print(x.shape)
        assert x.ndim == 2 and x.shape[1] == self.num_edges, f"expected (B, {self.num_edges})"
        B = x.shape[0]
        device = x.device

        crd = self.crd.to(device)
        starts = self.starts.to(device)
        vecs = self.vecs.to(device)
        idx = self.idx.to(device)

        # chis (B, e, d)
        chis = starts.unsqueeze(0) + x.unsqueeze(-1) * vecs.unsqueeze(0)

        # verts (B, n+e, d)
        verts = torch.cat([crd.unsqueeze(0).expand(B, -1, -1), chis], dim=1)

        # gather rows by idx -> (B, n*d*d, d)
        idx_expand = idx.view(1, -1, 1).expand(B, -1, self.euc_dim)
        gathered = verts.gather(1, idx_expand)

        # reshape into (B, n*d, d, d)
        vert_spread = gathered.view(B, self.num_classes * self.euc_dim, self.euc_dim, self.euc_dim)

        # first solve -> planes: (B, n*d, d)
        planes = self._batch_solve(vert_spread)

        # reshape to (B, n, d, d)
        planes_reshaped = planes.view(B, self.num_classes, self.euc_dim, self.euc_dim)

        # second solve -> intersections: (B, n, d)
        intersections = self._batch_solve(planes_reshaped)

        cevians = crd.unsqueeze(0) - intersections    # (B, n, d)
        out = 1.0 - torch.norm(cevians, dim=-1)       # (B, n)

        return out.to(torch.float32)
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
    safe_preds, safe_targets = preds.clamp(1e-7, 1 - 1e-7), targets.clamp(1e-7, 1 - 1e-7)
    loss = -(safe_targets * safe_preds.log() + (1 - safe_targets) * (1 - safe_preds).log())
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

def main():
    t = Tournament(100)
    # x = torch.rand((10,t.num_edges))
    # y = t(x)
    # print(y)
    print(t.gt)
    # print(t.gt.shape)
    print(t.min_logit)

if __name__ == "__main__":
    main()
