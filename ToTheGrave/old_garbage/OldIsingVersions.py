from preamble import *
from AdjacencyInit import J_normed_diag_gamma, J_random_normed, factorize_J
from BackBoneModels import MFArgmin, ResNet18Backbone, MobileNetBackbone
from old_garbage.TournamentThresholds import SingleConfidence, CenterThresholding
from IsingDeclarative import IsingJBlock

@torch.jit.script
def mf_argmin(h:torch.Tensor, J:torch.Tensor, T:float, max_iter:int=50, alpha:float = 0.25, tol:float=1e-6, damp:float=1e-4):
    return MFArgmin.apply(h, J, T, max_iter, alpha, tol, damp)
# forward(self, h, A, M_raw, S_raw, V_raw, P_perp, **kwargs)

class NeuralIsingEnergyDeclarative_J(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone = 'resnet18',
                 device = 'cpu',
                 freeze_backbone=False,
                 unfreeze_last_n=2):
        super().__init__()
        self.device = device
        self.linalg_device = 'cpu' if device.type == 'mps' else device
        self.num_classes = num_classes
        self.num_edges = num_classes * (num_classes - 1)// 2
        self.backbone = backbone
        backbone = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = backbone(device=device, output_dim=self.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        self.batchnorm = nn.LayerNorm(self.num_edges)
        self.layers = [self.model, self.batchnorm]
        self.middle = nn.Sequential(*self.layers)
        self.edge_list = list(combinations(range(num_classes), 2))
        self.A = A = self.get_decoder()
        AtA = (A.T @ A)            # (K,K)
        AtA_inv = torch.linalg.pinv(AtA.cpu()).to(self.device)     # stable inverse
        self.decode_inv = None
        self.register_buffer("P_A",  A @ AtA_inv @ A.T)        # (E,E)
        self.register_buffer("P_perp", torch.eye(self.num_edges, device=self.device) - self.P_A)

        # class-induced piece
        self.M_raw = nn.Parameter(torch.zeros(self.num_classes, self.num_classes))  # symmetric via tanh later (K,K)

        # residual in the orthogonal complement
        d = min(20, self.num_edges)  # latent rank
        self.V_raw = nn.Parameter(torch.randn(self.num_edges, d) * 0.05) # (E, d) random init
        self.S_raw = nn.Parameter(torch.zeros(d, d))  # symmetric via tanh later (d,d)

    def J(self):
        # class term
        M_sym = 0.5*(self.M_raw + self.M_raw.T)
        M = torch.tanh(M_sym)
        J_class = self.A @ M @ self.A.T

        # residual term
        V_proj = self.P_perp @ self.V_raw
        Q = torch.linalg.qr(V_proj.to(self.linalg_device))[0].to(self.device)  # (E,d) with Q^T Q = I
        S_sym = 0.5*(self.S_raw + self.S_raw.T)
        s_max = 0.85
        S = s_max * torch.tanh(S_sym)
        J_res = Q @ S @ Q.T

        return J_class + J_res

    def get_decoder(self):
        """
        returns: (E, K) adjacency matrix mapping class scores to edge logits
        """
        A = torch.zeros((self.num_edges, self.num_classes), device=self.device)
        for e, (i, j) in enumerate(self.edge_list):
            A[e, i] = 1
            A[e, j] = -1
        # ATA_inv_T = torch.linalg.pinv(A.T @ A).T            # (K, K) #TODO Right slash operator
        return A # @ ATA_inv_T                                # (E, K) #TODO

    def edge_probs_from_scores(self, z):
        """
        z: (B, K) class scores
        returns: (B, E) predicted edge probabilities
        """
        probs = []
        for (i, j) in self.edge_list:
            probs.append(torch.sigmoid(z[:, i] - z[:, j]))
        return torch.stack(probs, dim=1)

    def decode_z_from_edges(self, probs, eps=1e-6, rho=1e-3):
        logits = torch.logit(probs.clamp(eps, 1 - eps))  # (B,E)

        A_pinv = torch.linalg.pinv(self.A.to(self.linalg_device)).to(self.device)         # (K,E) once at init
        z = (A_pinv @ logits.T).T

        # if probs.device.type == 'mps':
        #     z = torch.linalg.lstsq(self.A.cpu(), logits.T.cpu()).solution.T.to(probs.device)
        # else:
        #     z = torch.linalg.lstsq(self.A, logits.T).solution.T
        z = z - z.mean(dim=1, keepdim=True)

        return z

    def forward(self,
                x,
                labels = None,
                train=True,
                max_iter=5,
                alpha = 0.1,
                tol=1e-4,
                T_mf = 1.0,
                T_sof = 1.0,
                ):
        h = self.middle(x)
        J_mat = self.J()
        m = mf_argmin(h, J_mat, T_mf, max_iter=max_iter, alpha=alpha, tol=tol)
        probs = ((m + 1) / 2)
        z = self.decode_z_from_edges(probs) / T_sof  # closed-form projection
        ce = F.cross_entropy(z, labels) if labels is not None else torch.tensor(0.0, device=x.device, requires_grad=True)
        return ce, z

class NeuralIsingEnergy(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone = 'resnet18',
                 device = 'cpu',
                 learn_J=True,
                 freeze_backbone=False,
                 unfreeze_last_n=2,
                 gamma=0.5,
                 normalize_energy=True,
                 no_diag =False):
        super().__init__()
        self.device = device
        self.linalg_device = 'cpu' if device.type == 'mps' else device
        self.num_classes = num_classes
        self.num_edges = num_classes * (num_classes - 1)// 2
        self.backbone = backbone
        backbone = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = backbone(device=device, output_dim=self.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        # self.batchnorm = nn.BatchNorm1d(self.num_edges)
        self.batchnorm = nn.LayerNorm(self.num_edges)
        self.layers = [self.model, self.batchnorm]
        self.middle = nn.Sequential(*self.layers)
        self.edge_list = list(combinations(range(num_classes), 2))
        # A = self.get_decoder()  # (E, K) @ (K, K) -> (E, K)
        # self.register_buffer("A", A)
        self.A = A = self.get_decoder()
        AtA = (A.T @ A)            # (K,K)
        AtA_inv = torch.linalg.pinv(AtA.cpu()).to(self.device)     # stable inverse
        self.decode_inv = None
        self.register_buffer("P_A",  A @ AtA_inv @ A.T)        # (E,E)
        self.register_buffer("P_perp", torch.eye(self.num_edges, device=self.device) - self.P_A)

        # class-induced piece
        self.M_raw = nn.Parameter(torch.zeros(self.num_classes, self.num_classes))  # symmetric via tanh later (K,K)

        # residual in the orthogonal complement
        d = min(20, self.num_edges)  # latent rank
        self.V_raw = nn.Parameter(torch.randn(self.num_edges, d) * 0.05) # (E, d) random init
        self.S_raw = nn.Parameter(torch.zeros(d, d))  # symmetric via tanh later (d,d)

        self.normalize_energy = normalize_energy

    def mf_entropy(self, m, eps=1e-6):
        p = (m.clamp(-1+eps, 1-eps) + 1) * 0.5
        return -(p*torch.log(p) + (1-p)*torch.log(1-p)).sum(dim=1)

    def J(self):
        # class term
        M_sym = 0.5*(self.M_raw + self.M_raw.T)
        M = torch.tanh(M_sym)
        J_class = self.A @ M @ self.A.T

        # residual term
        V_proj = self.P_perp @ self.V_raw
        Q = torch.linalg.qr(V_proj.to(self.linalg_device))[0].to(self.device)  # (E,d) with Q^T Q = I
        S_sym = 0.5*(self.S_raw + self.S_raw.T)
        s_max = 0.85
        S = s_max * torch.tanh(S_sym)
        J_res = Q @ S @ Q.T

        return J_class + J_res

    def get_decoder(self):
        """
        returns: (E, K) adjacency matrix mapping class scores to edge logits
        """
        A = torch.zeros((self.num_edges, self.num_classes), device=self.device)
        for e, (i, j) in enumerate(self.edge_list):
            A[e, i] = 1
            A[e, j] = -1
        # ATA_inv_T = torch.linalg.pinv(A.T @ A).T            # (K, K) #TODO Right slash operator
        return A # @ ATA_inv_T                                # (E, K) #TODO

    def edge_probs_from_scores(self, z):
        """
        z: (B, K) class scores
        returns: (B, E) predicted edge probabilities
        """
        probs = []
        for (i, j) in self.edge_list:
            probs.append(torch.sigmoid(z[:, i] - z[:, j]))
        return torch.stack(probs, dim=1)

    def decoding_energy_loss(self, z, edge_probs):
        """
        z: (B, K)
        edge_probs: (B, E)  # from Ising
        """
        pred_edge_probs = self.edge_probs_from_scores(z)
        return ((pred_edge_probs - edge_probs) ** 2).mean()

    def ising_energy(self, h, m, J):
        """
        h: (B, E)   local fields
        m: (B, E)   mean-field spins (in [-1, 1])
        J: (E, E)   symmetric coupling matrix
        returns: (B,) energy per sample
        """
        unary = -(h * m).sum(dim=1)                     # (B,)
        # pairwise = -0.5 * (m @ J @ m.T).sum(dim=1)        # (B,)
        # pairwise = -0.5 * (m @ J * m).sum(dim=1)        # (B,)
        pairwise = -0.5 * (m @ J @ m.T).diag()        # (B,)
        return (unary + pairwise) / (self.num_edges if self.normalize_energy else 1)

    def decode_z_from_edges(self, probs, eps=1e-6, rho=1e-3):
        logits = torch.logit(probs.clamp(eps, 1 - eps))  # (B,E)
        K = self.num_classes
        At = self.A.T
        # (A^T A + rho I)^{-1} A^T logits  〈— ridge-lstsq
        if self.decode_inv is None:
            AtA = (At @ self.A).to(self.linalg_device)
            self.decode_inv = torch.linalg.pinv((AtA + rho * torch.eye(K, device=self.linalg_device))).to(self.device) @ At  # (K,E)
        # inv = torch.linalg.pinv(At @ self.A + rho * torch.eye(K, device=probs.device))
        z = (self.decode_inv @ logits.T).T
        z = z - z.mean(dim=1, keepdim=True)
        return z

    def consistency_loss(self, z, m):
        """
        z: (B, K)
        m: (B, E)
        """
        pred_m = torch.tanh(self.A @ z.T).T
        return ((pred_m - m)**2).mean()

    def forward(self,
                x,
                labels = None,
                max_iter=5,
                alpha = 0.1,
                beta=0.1,
                lambda_ = 0.001,
                train=False,
                tol=1e-4,
                softplus_before_energy=True,
                mean_field=True,
                consistency_loss_weight=0.000001,
                mu = 0.00001,
                eta = 0.0001,
                T = 4):
        # Extract features
        h = self.middle(x)
        J_mat = self.J()

        m = torch.tanh(h / T)  # T=4
        for _ in range(max_iter):
            m_prev = m
            # m = (1 - alpha) * m_prev + alpha * torch.tanh((h + m_prev @ J_mat.detach()) / T)
            m = (1 - alpha) * m_prev + alpha * torch.tanh((h + m_prev @ ((1-beta) * J_mat.detach() + beta * J_mat)) / T)
            if (m - m_prev).abs().max() < tol:
                break

        # TODO: Return back to normal eventually, this is just for testing
        probs = (m + 1) / 2
        # probs = (mf + 1) / 2
        z = self.decode_z_from_edges(probs)   # closed-form projection
        ce = F.cross_entropy(z, labels) if labels is not None else torch.tensor(0.0, device=x.device, requires_grad=True)
        E = self.ising_energy(h, m, J_mat) # .abs().mean()
        EF = (E - T * self.mf_entropy(m)).mean()

        loss = ce
        # loss += lambda_ * E
        loss += lambda_ * EF
        # loss += consistency_loss_weight * self.consistency_loss(z, m)
        # loss += eta * (h**2).mean()
        # loss += 0.000 * ((mf - m)**2).mean()
        # loss += mu * torch.norm(self.V_raw, p='fro')

        return loss, z

class NeuralIsingEnergySpectral_failed(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone = 'resnet18',
                 device = 'cpu',
                 learn_J=True,
                 freeze_backbone=False,
                 unfreeze_last_n=2,
                 gamma=0.5,
                 normalize_energy=True,
                 no_diag =False):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_edges = num_classes * (num_classes - 1)// 2
        self.backbone = backbone
        backbone = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = backbone(device=device, output_dim=self.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        # self.batchnorm = nn.BatchNorm1d(self.num_edges)
        self.batchnorm = nn.LayerNorm(self.num_edges)
        self.layers = [self.model, self.batchnorm]
        self.middle = nn.Sequential(*self.layers)
        self.edge_list = list(combinations(range(num_classes), 2))
        self.A = self.get_decoder()  # (E, K) @ (K, K) -> (E, K)
        self.normalize_energy = normalize_energy
        init_J = J_normed_diag_gamma(num_classes, self.num_edges, gamma=gamma, no_diag=no_diag)
        U, W_raw = factorize_J(init_J, d=min(20, self.num_edges))
        if learn_J:
            self.U_raw = nn.Parameter(U.clone())
            self.W_raw = nn.Parameter(W_raw.clone())
            # self.register_buffer("W_raw", W_raw)
        else:
            self.register_buffer("J", init_J)

    # def W(self):
    #     # W_ = self.W_raw
    #     W_ = self.normalize_W(self.W_raw)
    #     W = 1/2 * (W_ + W_.T)
    #     return W
    #     # return W

    def W(self):
        W = 0.5 * (self.W_raw + self.W_raw.T)
        return W

    def U(self):
        return torch.tanh(self.U_raw)

    # def normalize_W(self, W_raw):
    #     u, s, v = torch.linalg.svd(W_raw.cpu())
    #     s = torch.tanh(s)       # smooth clipping, not hard
    #     return (u @ torch.diag(s) @ v.T).to(W_raw.device)

    def J(self):
        U_ = self.U()
        return U_ @ torch.tanh(self.W()) @ U_.T

    def get_decoder(self):
        """
        returns: (E, K) adjacency matrix mapping class scores to edge logits
        """
        A = torch.zeros((self.num_edges, self.num_classes), device=self.device)
        for e, (i, j) in enumerate(self.edge_list):
            A[e, i] = 1
            A[e, j] = -1
        # ATA_inv_T = torch.linalg.pinv(A.T @ A).T            # (K, K) #TODO Right slash operator
        return A # @ ATA_inv_T                                # (E, K) #TODO

    def edge_probs_from_scores(self, z):
        """
        z: (B, K) class scores
        returns: (B, E) predicted edge probabilities
        """
        probs = []
        for (i, j) in self.edge_list:
            probs.append(torch.sigmoid(z[:, i] - z[:, j]))
        return torch.stack(probs, dim=1)

    def decoding_energy_loss(self, z, edge_probs):
        """
        z: (B, K)
        edge_probs: (B, E)  # from Ising
        """
        pred_edge_probs = self.edge_probs_from_scores(z)
        return ((pred_edge_probs - edge_probs) ** 2).mean()

    def ising_energy(self, h, m, J):
        """
        h: (B, E)   local fields
        m: (B, E)   mean-field spins (in [-1, 1])
        J: (E, E)   symmetric coupling matrix
        returns: (B,) energy per sample
        """
        unary = -(h * m).sum(dim=1)                     # (B,)
        # pairwise = -0.5 * (m @ J @ m.T).sum(dim=1)        # (B,)
        # pairwise = -0.5 * (m @ J * m).sum(dim=1)        # (B,)
        pairwise = -0.5 * (m @ J @ m.T).diag()        # (B,)
        return (unary + pairwise) / (self.num_edges if self.normalize_energy else 1)

    def decode_z_from_edges(self, probs, eps=1e-6):
        """
        probs: (B, E)
        A:     (E, K)
        returns z: (B, K)
        """
        logits = torch.logit(probs.clamp(eps, 1 - eps))  # (B, E)
        # Least squares: z = argmin ||A z − logits||
        # Closed form: z = (AᵀA)^⁻¹ Aᵀ logits
        # z = logits @ self.decoder                   # (B, K)
        # print(self.decoder.shape, logits.shape)

        if probs.device.type == 'mps':
            z = torch.linalg.lstsq(self.A.cpu(), logits.T.cpu()).solution.T.to(probs.device)
        else:
            z = torch.linalg.lstsq(self.A, logits.T).solution.T

        # print(z.shape)

        # Optional gauge fixing
        z = z - z.mean(dim=1, keepdim=True) # unneeded because of cross entropy softmax

        return z
    def consistency_loss(self, z, m):
        """
        z: (B, K)
        m: (B, E)
        """
        pred_m = torch.tanh(self.A @ z.T).T
        return ((pred_m - m)**2).mean()

    def forward(self,
                x,
                labels = None,
                max_iter=5,
                alpha = 0.1,
                beta=0.1,
                lambda_ = 0.001,
                train=False,
                tol=1e-4,
                softplus_before_energy=True,
                mean_field=True,
                consistency_loss_weight=0.000001,
                mu = 0.00001,
                eta = 0.0001,
                T = 4):
        # Extract features
        h = self.middle(x)
        J_mat = self.J()
        m = mf = F.tanh(h/T)
        # if mean_field or not train:
        #     for _ in range(max_iter):
        #         mf_prev = mf
        #         mf = ( 1- alpha) * mf + alpha * F.tanh((h + torch.matmul(mf, self.J.detach()))/2)  # update rule, inplace
        #         if torch.max(torch.abs(mf - mf_prev)) < tol:
        #             break
        if mean_field or not train:
            for _ in range(max_iter):
                m_prev = m
                # m = ( 1- alpha) * m + alpha * F.tanh((h + torch.matmul(mf, self.J.detach()))/2)  # update rule
                # m = ( 1- alpha) * m + alpha * F.tanh((h + torch.matmul(m, J_mat.detach()))/T)  # update rule
                m = ( 1- alpha) * m + alpha * F.tanh((h + torch.matmul(m, (beta) * J_mat + (1-beta) * J_mat.detach()))/T)  # update rule
                # mf = ( 1- alpha) * mf + alpha * F.tanh((h + torch.matmul(mf, self.J.detach()))/2)  # update rule
                if torch.max(torch.abs(m - m_prev)) < tol:
                # if torch.max(torch.abs(m - m_prev)) < tol:
                    break
        # TODO: Return back to normal eventually, this is just for testing
        probs = (m + 1) / 2
        # probs = (mf + 1) / 2
        z = self.decode_z_from_edges(probs)   # closed-form projection
        ce = F.cross_entropy(z, labels) if labels is not None else torch.tensor(0.0, device=x.device, requires_grad=True)
        E = self.ising_energy(h, m, J_mat).mean()

        loss = ce
        loss += lambda_ * E
        # loss += consistency_loss_weight * self.consistency_loss(z, m)
        loss += mu * torch.norm(self.U(), p='fro')
        loss += eta * (h**2).mean()
        # loss += mu * (J_mat**2).mean()
        # loss += 0.000 * ((mf - m)**2).mean()
        # loss += mu * torch.norm(self.J(), p=2)

        return loss, z

class NeuralIsingEnergyOld(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone = 'resnet18',
                 device = 'cpu',
                 learn_J=True,
                 freeze_backbone=False,
                 unfreeze_last_n=2,
                 gamma=0.5,
                 normalize_energy=True,
                 no_diag =False):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.num_edges = num_classes * (num_classes - 1)// 2
        self.backbone = backbone
        backbone = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = backbone(device=device, output_dim=self.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        self.batchnorm = nn.BatchNorm1d(self.num_edges)
        self.layers = [self.model, self.batchnorm]
        self.middle = nn.Sequential(*self.layers)
        self.edge_list = list(combinations(range(num_classes), 2))
        self.A = self.get_decoder()  # (E, K) @ (K, K) -> (E, K)
        self.normalize_energy = normalize_energy
        adjacency_matrix = J_normed_diag_gamma(num_classes, self.num_edges, gamma=gamma, no_diag=no_diag)
        if learn_J:
            self.J = nn.Parameter(adjacency_matrix.clone())
        else:
            self.register_buffer("J", adjacency_matrix)

    def get_decoder(self):
        """
        returns: (E, K) adjacency matrix mapping class scores to edge logits
        """
        A = torch.zeros((self.num_edges, self.num_classes), device=self.device)
        for e, (i, j) in enumerate(self.edge_list):
            A[e, i] = 1
            A[e, j] = -1
        # ATA_inv_T = torch.linalg.pinv(A.T @ A).T            # (K, K) #TODO Right slash operator
        return A # @ ATA_inv_T                                # (E, K) #TODO

    def edge_probs_from_scores(self, z):
        """
        z: (B, K) class scores
        returns: (B, E) predicted edge probabilities
        """
        probs = []
        for (i, j) in self.edge_list:
            probs.append(torch.sigmoid(z[:, i] - z[:, j]))
        return torch.stack(probs, dim=1)

    def decoding_energy_loss(self, z, edge_probs):
        """
        z: (B, K)
        edge_probs: (B, E)  # from Ising
        """
        pred_edge_probs = self.edge_probs_from_scores(z)
        return ((pred_edge_probs - edge_probs) ** 2).mean()

    def ising_energy(self, h, m, J):
        """
        h: (B, E)   local fields
        m: (B, E)   mean-field spins (in [-1, 1])
        J: (E, E)   symmetric coupling matrix
        returns: (B,) energy per sample
        """
        unary = -(h * m).sum(dim=1)                     # (B,)
        # pairwise = -0.5 * (m @ J @ m.T).sum(dim=1)        # (B,)
        # pairwise = -0.5 * (m @ J * m).sum(dim=1)        # (B,)
        pairwise = -0.5 * (m @ J @ m.T).diag()        # (B,)
        return (unary + pairwise) / (self.num_edges if self.normalize_energy else 1)

    def decode_z_from_edges(self, probs, eps=1e-6):
        """
        probs: (B, E)
        A:     (E, K)
        returns z: (B, K)
        """
        logits = torch.logit(probs.clamp(eps, 1 - eps))  # (B, E)
        # Least squares: z = argmin ||A z − logits||
        # Closed form: z = (AᵀA)^⁻¹ Aᵀ logits
        # z = logits @ self.decoder                   # (B, K)
        # print(self.decoder.shape, logits.shape)
        if device.type == 'mps':
            z = torch.linalg.lstsq(self.A.cpu(), logits.T.cpu()).solution.T.to(device=probs.device)               # (B, K)
        else:
            z = torch.linalg.lstsq(self.A, logits.T).solution.T
        # print(z.shape)

        # Optional gauge fixing
        z = z - z.mean(dim=1, keepdim=True) # unneeded because of cross entropy softmax

        return z
    def consistency_loss(self, z, m):
        """
        z: (B, K)
        m: (B, E)
        """
        pred_m = torch.tanh(self.A @ z.T).T
        return ((pred_m - m)**2).mean()

    def forward(self,
                x,
                labels = None,
                max_iter=5,
                alpha = 0.1,
                lambda_ = 0.1,
                train=False,
                tol=1e-4,
                softplus_before_energy=True,
                mean_field=True,
                consistency_loss_weight=0.001,
                mu = 0.01):
        # Extract features
        h = self.middle(x)

        m = mf = F.tanh(h)
        # if mean_field or not train:
        #     for _ in range(max_iter):
        #         mf_prev = mf
        #         mf = ( 1- alpha) * mf + alpha * F.tanh((h + torch.matmul(mf, self.J.detach()))/2)  # update rule, inplace
        #         if torch.max(torch.abs(mf - mf_prev)) < tol:
        #             break
        if mean_field or not train:
            for _ in range(max_iter):
                m_prev = mf
                # m = ( 1- alpha) * m + alpha * F.tanh((h + torch.matmul(mf, self.J.detach()))/2)  # update rule
                mf = ( 1- alpha) * mf + alpha * F.tanh((h + torch.matmul(mf, self.J.detach()))/2)  # update rule
                if torch.max(torch.abs(mf - m_prev)) < tol:
                # if torch.max(torch.abs(m - m_prev)) < tol:
                    break
        # TODO: Return back to normal eventually, this is just for testing
        # probs = (m + 1) / 2
        probs = (mf + 1) / 2
        z = self.decode_z_from_edges(probs)   # closed-form projection
        ce = F.cross_entropy(z, labels) if labels is not None else torch.tensor(0.0, device=x.device, requires_grad=True)
        E = self.ising_energy(h, m, self.J).mean()
        if softplus_before_energy > 0:
            loss = ce + lambda_ * F.softplus(E)
        elif softplus_before_energy < 0:
            loss = ce * 1  \
                + lambda_ * E \
                + self.consistency_loss(z, m) * consistency_loss_weight \
                + 0.000 * ((mf - m)**2).mean() \
                + mu * torch.norm(self.J, p=2)
                # + mu * (self.J**2).mean()
        else:
            loss = ce + F.softplus(lambda_ * E)
        return loss, z

class NeuralIsingRegularizer(nn.Module):
    def __init__(self, num_classes, backbone = 'resnet18', device = 'cpu', learn_J=True, freeze_backbone=False, unfreeze_last_n=2, trinary=False, gamma=0.5, no_diag = False):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        num_edges = num_classes * (num_classes - 1)// 2
        self.num_edges = num_edges
        self.backbone = backbone
        backbone = ResNet18Backbone if backbone == 'resnet18' else MobileNetBackbone
        self.model = backbone(device=device, output_dim=self.num_edges, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        self.batchnorm = nn.BatchNorm1d(self.num_edges)
        self.layers = [self.model, self.batchnorm]
        self.middle = nn.Sequential(*self.layers)
        adjacency_matrix = J_normed_diag_gamma(num_classes, self.num_edges, gamma=gamma, no_diag=no_diag)
        self.lf = RegularizerLoss(num_classes)
        self.decoder = SingleConfidence(num_classes) if not trinary else CenterThresholding(num_classes, alpha=0.16666667) #TODO: if using this, change in TournamentThresholds.py
                                                                                                                        # to match abstenetion thresholding, training too
        if learn_J:
            self.J = nn.Parameter(adjacency_matrix.clone())
        else:
            self.register_buffer("J", adjacency_matrix)

    def normalize_J(self, p: int = 2, eps: float = 1e-12):
        if not hasattr(self, 'J'):
            return
        with torch.no_grad():
            n = float(self.J.data.norm(p=p))
            if n == 0.0:
                return
            self.J.data.div_(n + eps)

    def attach_post_step_hook(self, optimizer: torch.optim.Optimizer):
        if not hasattr(optimizer, '_post_step_hooks'):
            optimizer._post_step_hooks = []
            optimizer._orig_step = optimizer.step

            def _step_and_call_hooks(*args, **kwargs):
                res = optimizer._orig_step(*args, **kwargs)
                for cb in list(getattr(optimizer, '_post_step_hooks', [])):
                    try:
                        cb()
                    except Exception as e:
                        print(f"Warning: post-step hook failed: {e}")
                return res

            optimizer.step = _step_and_call_hooks

        for cb in optimizer._post_step_hooks:
            if getattr(cb, '__model_hook__', None) is self:
                return

        def _hook():
            self.normalize_J()

        _hook.__model_hook__ = self
        optimizer._post_step_hooks.append(_hook)

    def forward(self, x, y=None, max_iter=5, alpha = 0.2, train=False, tol=1e-4):
        h = self.middle(x)
        # Mean-field inference for marginals
        # Initialize m_i = tanh(h_i)
        m = F.tanh(h)
        for _ in range(max_iter):
            m_prev = m
            # m = torch.tanh(h + alpha * torch.matmul(m, self.J))  # update rule
            m = ( 1- alpha) * m_prev + alpha * F.tanh(h + torch.matmul(m_prev, self.J))  # update rule
            # m = ( 1- alpha) * m + alpha * torch.tanh(torch.matmul(m, self.J))  # update rule
            # # Optional: check for convergence
            # if torch.max(torch.abs(m - m_prev)) < tol:
            #     break
        # Convert to probabilities in [0,1]
        probs = (m + 1) / 2
        preds = self.decoder(probs)
        loss = self.lf(probs, y) if y is not None else torch.tensor(0.0, device=x.device, requires_grad=True)
        return loss, preds

class RegularizerLoss(nn.Module):
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