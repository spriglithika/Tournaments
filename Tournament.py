import torch
from itertools import combinations
class Tournament(torch.nn.Module):
    def __init__(self, num_classes):
        super(Tournament, self).__init__()
        self.num_classes = num_classes
        self.euc_dim = num_classes - 1
        self.num_edges = self.nSimplex(num_classes)
        # register tensors that are not learnable so they move with .to(device)
        self.register_buffer('cevians', self.selectionIndicies())
        crd, starts, vecs = self.coordinates()
        self.register_buffer('crd', crd)
        self.register_buffer('starts', starts)
        self.register_buffer('vecs', vecs)
        gt, perms = self.get_gt()
        self.register_buffer('gt', gt)
        self.register_buffer('perms', perms)
        min_logit = self.get_min_logit()
        self.register_buffer('min_logit', min_logit)

    def get_min_logit(self):
        x = torch.ones(self.num_edges) *.5
        x[:self.num_classes - 1] = 0
        return self(x).min()

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
        return cevians.sum(1)

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

    def forward(self, x):
        assert x.shape[-1] == self.num_edges, f"Input last dimension must be {self.num_edges}, got {x.shape[-1]}"
        chis = self.starts[None, :, :] + x[..., None] * self.vecs[None, :, :]
        means = self.cevians @ chis / (self.euc_dim)
        cevians = self.crd - means
        out = 1 - torch.linalg.norm(cevians,axis=-1)
        return out #, means

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
