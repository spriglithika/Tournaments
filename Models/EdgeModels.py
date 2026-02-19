from preamble import *
from Models.BackBoneModels import ResNet18Backbone, MobileNetBackbone
from Models.IsingDeclarative import IsingJBlock

class NeuralIsingEnergyDeclarative(nn.Module):
    def __init__(self, num_classes, backbone = 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=2, softening = False):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_edges = num_classes * (num_classes - 1)// 2
        self.backbone = backbone
        backbone = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = backbone(device=device, output_dim=self.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        self.batchnorm = nn.LayerNorm(self.num_edges)
        self.layers = [self.model, self.batchnorm]
        self.middle = nn.Sequential(*self.layers)
        self.edge_list = list(combinations(range(num_classes), 2))
        d = min(20, self.num_edges)
        A = self.get_decoder()
        self.ising_j_block = IsingJBlock(A, d)
        self.register_buffer("A_pinv", torch.linalg.pinv(A.to(linalg_device)).to(device))
        if softening:
            self.log_Tsoft = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('log_Tsoft', torch.tensor(0.0))

    def J(self):
        M, S, Q = self.ising_j_block.current_blocks()
        return self.ising_j_block.A @ M @ self.ising_j_block.A.T + Q @ S @ Q.T

    def get_decoder(self):
        A = torch.zeros((self.num_edges, self.num_classes), device=self.device)
        for e, (i, j) in enumerate(self.edge_list):
            A[e, i] = 1; A[e, j] = -1
        return A

    def edge_probs_from_scores(self, z):
        probs = []
        for (i, j) in self.edge_list:
            probs.append(torch.sigmoid(z[:, i] - z[:, j]))
        return torch.stack(probs, dim=1)

    def decode_z_from_edges(self, probs, eps=1e-6, rho=1e-3):
        z = (self.A_pinv @ torch.logit(probs.clamp(eps, 1 - eps)).T).T
        return z - z.mean(dim=1, keepdim=True)

    def forward(self, x, labels = None, train=True, max_iter=5, alpha = 0.1, tol=1e-4,):
        h = self.middle(x)
        m = self.ising_j_block(h, T=1.0, max_iter=max_iter, alpha=alpha, tol=tol)
        probs = ((m + 1) / 2)
        z = self.decode_z_from_edges(probs) / self.log_Tsoft.exp()  # closed-form projection
        ce = F.cross_entropy(z, labels) if labels is not None else torch.tensor(0.0, device=x.device, requires_grad=True)
        return ce, z

class BradleyTerryEdgeModel(nn.Module):
    def __init__(self, num_classes, backbone = 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=2, softening=False):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_edges = num_classes * (num_classes - 1)// 2
        self.backbone = backbone
        backbone = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = backbone(device=device, output_dim=self.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        self.batchnorm = nn.LayerNorm(self.num_edges)
        self.layers = [self.model, self.batchnorm]
        self.middle = nn.Sequential(*self.layers)
        self.edge_list = list(combinations(range(num_classes), 2))
        d = min(20, self.num_edges)
        A = self.get_decoder()
        # self.ising_j_block = IsingJBlock(A, d)
        self.register_buffer("A_pinv", torch.linalg.pinv(A.to(linalg_device)).to(device))
        if softening:
            self.log_Tsoft = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('log_Tsoft', torch.tensor(0.0))

    # def J(self):
    #     M, S, Q = self.ising_j_block.current_blocks()
    #     return self.ising_j_block.A @ M @ self.ising_j_block.A.T + Q @ S @ Q.T

    def get_decoder(self):
        A = torch.zeros((self.num_edges, self.num_classes), device=self.device)
        for e, (i, j) in enumerate(self.edge_list):
            A[e, i] = 1; A[e, j] = -1
        return A

    def edge_probs_from_scores(self, z):
        probs = []
        for (i, j) in self.edge_list:
            probs.append(torch.sigmoid(z[:, i] - z[:, j]))
        return torch.stack(probs, dim=1)

    def decode_z_from_edges(self, probs, eps=1e-6, rho=1e-3):
        z = (self.A_pinv @ torch.logit(probs.clamp(eps, 1 - eps)).T).T
        return z - z.mean(dim=1, keepdim=True)

    def forward(self, x, labels = None, train=True):
        h = self.middle(x)
        # m = self.ising_j_block(h, T=T_mf, max_iter=max_iter, alpha=alpha, tol=tol)
        probs = torch.sigmoid(h)
        z = self.decode_z_from_edges(probs) / self.log_Tsoft.exp()  # closed-form projection
        ce = F.cross_entropy(z, labels) if labels is not None else torch.tensor(0.0, device=x.device, requires_grad=True)
        return ce, z

    """Vectorized regularizer loss.

    Computes per-sample BCE only on edges involving the sample's true class,
    then averages over the batch. This avoids Python loops and uses batched
    tensor operations for speed.
    """
    def __init__(self, num_classes: int, device: str = 'cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        edge_list = list(combinations(range(num_classes), 2))

        masks = []
        targets = []
        for c in range(num_classes):
            mask = torch.tensor([1 if (i == c or j == c) else 0 for (i, j) in edge_list], dtype=torch.bool)
            t = torch.tensor([1.0 if i == c else 0.0 if j == c else 0.5 for (i, j) in edge_list], dtype=torch.float32)
            masks.append(mask)
            targets.append(t)

        # Stack into tensors of shape (K, E) and register as buffers so they
        # move with the module to the right device automatically.
        edge_masks = torch.stack(masks, dim=0).to(device)      # (K, E)
        edge_targets = torch.stack(targets, dim=0).to(device)  # (K, E)
        self.register_buffer('edge_masks', edge_masks)
        self.register_buffer('edge_targets', edge_targets)

    def forward(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """probs: (B, E), labels: (B,) or (B,1)

        Returns scalar loss: mean over valid per-sample masked BCE means.
        """
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        labels = labels.long().to(self.edge_masks.device)

        # index masks/targets per-sample: (B, E)
        mask_batch = self.edge_masks[labels]
        target_batch = self.edge_targets[labels]

        # element-wise BCE without reduction
        bce_elem = F.binary_cross_entropy(probs, target_batch, reduction='none')

        # zero out non-masked positions and compute per-sample means
        masked_bce = bce_elem * mask_batch.to(bce_elem.dtype)
        counts = mask_batch.sum(dim=1).to(bce_elem.dtype)

        valid = counts > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=probs.device, requires_grad=True)

        per_sample_mean = torch.zeros(probs.shape[0], device=probs.device, dtype=bce_elem.dtype)
        per_sample_mean[valid] = (masked_bce.sum(dim=1)[valid] / counts[valid].to(bce_elem.dtype))

        return per_sample_mean[valid].mean()