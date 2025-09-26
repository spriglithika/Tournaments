import torch
class Tournament(torch.nn.Module):
    def __init__(self, num_classes):
        super(Tournament, self).__init__()
        self.num_classes = num_classes
        self.euc_dim = num_classes - 1
        self.num_edges = self.nSimplex(num_classes)
        self.cevians = self.selectionIndicies()
        self.crd, self.starts, self.vecs = self.coordinates()

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
        return torch.tensor(cevians.sum(1))

    def coordinates(self):
        n =self.euc_dim
        r2 = 2**.5
        es = torch.eye(n)/(r2)
        base = (torch.ones((n,n)) + 1/((n+1)**.5))/(n*r2)
        extra = torch.unsqueeze((torch.ones(n)/((2*(n+1))**.5)),0)
        crd = torch.cat((es-base,extra),0)
        thing = torch.triu(torch.ones((self.num_classes,self.num_classes)),1)
        idx = torch.tensor(thing.nonzero())
        starts = crd[idx[:, 0]]
        ends = crd[idx[:, 1]]
        vecs = ends - starts
        return crd, starts, vecs

    def forward(self, x):
        assert x.shape[-1] == self.num_edges, f"Input last dimension must be {self.num_edges}, got {x.shape[-1]}"
        chis = self.starts[None, :, :] + x[..., None] * self.vecs[None, :, :]
        means = self.cevians @ chis / (self.num_classes - 1)
        cevians = self.crd - means
        out = 1 - torch.linalg.norm(cevians,axis=-1)
        return out #, means

def symmetric_cross_entropy(preds, targets):
    return -(targets * preds.log() + (1 - targets) * (1 - preds).log()).mean()

def main():
    t = Tournament(4)
    x = torch.rand((10,t.num_edges))
    y = t(x)
    print(y)

if __name__ == "__main__":
    main()
