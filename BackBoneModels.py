
import math
from preamble import *
from torch import autograd, nn
class MobileNetBackbone(nn.Module):
    def __init__(self, device='cpu', output_dim=128, pretrained = False, freeze=False, unfreeze_last_n=1):
        super(MobileNetBackbone, self).__init__()
        self.device = device
        output_dim = int(output_dim)
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)
        self.features = mobilenet.features
        self.avgpool = mobilenet.avgpool
        self.flatten = nn.Flatten()

        # Freeze all layers first
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            for param in self.avgpool.parameters():
                param.requires_grad = False

            # Unfreeze last `n` blocks
            if unfreeze_last_n > 0:
                blocks = list(self.features.children())
                for block in blocks[-unfreeze_last_n:]:
                    for param in block.parameters():
                        param.requires_grad = True

        # Final projection to output_dim
        # Final projection to output_dim (no activation here -- produce raw features/logits)
        self.class_head = nn.Sequential(mobilenet.classifier[0], mobilenet.classifier[1], mobilenet.classifier[2])
        self.dropout = nn.Dropout(.2)
        self.fc = nn.Linear(mobilenet.classifier[3].in_features, output_dim)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.class_head(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class ResNet18Backbone(nn.Module):
    def __init__(self, device='cpu', output_dim=128, freeze=False, unfreeze_last_n=1):
        super(ResNet18Backbone, self).__init__()
        self.device = device

        # resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet18 = models.resnet18()
        resnet18.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(resnet18.children())[:-1])  # Exclude the final FC layer
        self.flatten = nn.Flatten()

        # Freeze all layers first
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

            # Unfreeze last `n` blocks
            if unfreeze_last_n > 0:
                blocks = list(self.features.children())
                for block in blocks[-unfreeze_last_n:]:
                    for param in block.parameters():
                        param.requires_grad = True

        # Final projection to output_dim
        # Final projection to output_dim (no activation here -- produce raw features/logits)
        self.fc = nn.Linear(resnet18.fc.in_features, output_dim)

    def forward(self, x):
        # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

class ResNet18BackboneExtended(nn.Module):
    def __init__(self, device='cpu', output_dim=128, freeze=False, unfreeze_last_n=1):
        super(ResNet18BackboneExtended, self).__init__()
        self.device = device

        # resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet18 = models.resnet18()
        resnet18.maxpool = nn.Identity()
        self.features = nn.Sequential(*list(resnet18.children())[:-1])  # Exclude the final FC layer
        self.flatten = nn.Flatten()

        # Freeze all layers first
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False

            # Unfreeze last `n` blocks
            if unfreeze_last_n > 0:
                blocks = list(self.features.children())
                for block in blocks[-unfreeze_last_n:]:
                    for param in block.parameters():
                        param.requires_grad = True

        # Final projection to output_dim
        # Final projection to output_dim (no activation here -- produce raw features/logits)
        self.fc1 = nn.Linear(resnet18.fc.in_features, 2000)
        self.hswish = nn.Hardswish()
        self.dropout = nn.Dropout(.2, inplace=True)
        self.fc2 = nn.Linear(2000, output_dim)

    def forward(self, x):
        # x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.hswish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class BaseModel(torch.nn.Module):
    def __init__(self, num_classes, backbone= 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=1):
        super(BaseModel, self).__init__()
        model = ResNet18Backbone if backbone == 'resnet18' else ResNet18BackboneExtended if backbone == 'resnet18ext' else MobileNetBackbone
        self.device = device
        self.num_classes = num_classes
        self.model = model(device=device, output_dim=num_classes, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        # Do not apply BatchNorm/ReLU to final logits; keep logits raw.

    def forward(self, x, y = None, train = False):
        x = self.model(x)
        loss = F.cross_entropy(x, y) if y is not None else torch.tensor(0.0, device=x.device, requires_grad=True)
        return loss, x

class MidModel(torch.nn.Module):
    def __init__(self, num_classes, backbone= 'resnet18', device = 'cpu', freeze_backbone=False, unfreeze_last_n=1):
        super(MidModel, self).__init__()
        self.num_classes = num_classes
        edge_count = int(num_classes * (num_classes - 1) * 0.5)
        model = MobileNetBackbone
        mid_features = math.ceil((1280 * edge_count)/(1280 + num_classes))
        if backbone == 'resnet18':
            model = ResNet18Backbone
            mid_features = math.ceil((512 * edge_count)/(512 + num_classes))
        if backbone == 'resnet18ext':
            model = ResNet18BackboneExtended
            mid_features = math.ceil((512 * edge_count)/(512 + num_classes))
        self.device = device
        self.model = model(device=device, output_dim=mid_features, freeze=freeze_backbone, unfreeze_last_n=unfreeze_last_n)
        # self.fc2 = nn.Linear(128, num_classes * (num_classes - 1) * 0.5)
        self.fc3 = nn.Linear(mid_features, num_classes)
        self.batchnorm2 = nn.BatchNorm1d(mid_features)
        # self.batch_norm3 = nn.BatchNorm1d(num_classes)
        # keep logits raw at the end

    def forward(self, x, y = None, train = False):
        x = self.model(x)
        x = self.batchnorm2(x)
        x = F.mish(x)
        x = self.fc3(x)
        loss = F.cross_entropy(x, y) if y is not None else torch.tensor(0.0, device=x.device, requires_grad=True)
        return loss, x

    import torch

class MFArgmin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h:torch.Tensor, J:torch.Tensor, T:float, max_iter:int=50, alpha:float = 0.25, tol:float=1e-6, damp:float=1e-4):
        # Graph-free forward solver (any method is fine)
        with torch.no_grad():
            m = torch.tanh(h / T)
            for _ in range(max_iter):
                m_prev = m
                m = (1 - alpha) * m_prev + alpha * torch.tanh((h + m_prev @ J) / T)
                if (m - m_prev).abs().max() < tol:
                    break

        ctx.save_for_backward(m, h, J)
        ctx.T = T
        ctx.damp = damp
        return m

    @staticmethod
    def backward(ctx, grad_out):
        m, h, J = ctx.saved_tensors
        T = ctx.T
        damp = ctx.damp

        # 🔑 Make sure autograd is ON during nested grad calls
        with torch.enable_grad():
            # Re-enable grad on local copies
            m_req = m.detach().requires_grad_(True)     # (B,E)
            h_req = h.detach().requires_grad_(True)     # (B,E)
            J_req = J.detach().requires_grad_(True)     # (E,E)

            def free_energy_local(m_local):
                p = (m_local.clamp(-1+1e-6, 1-1e-6) + 1) * 0.5
                ent = -(p * (p + 1e-12).log() + (1-p) * (1-p + 1e-12).log()).sum(dim=1)  # (B,)
                unary = -(h_req * m_local).sum(dim=1)                                     # (B,)
                pair = -0.5 * (m_local @ J_req @ m_local.transpose(1, 0)).diag()         # (B,)
                return (unary + pair - T * ent).sum()                                     # scalar

            # Build scalar objective on the grad-enabled copies
            F_val = free_energy_local(m_req)
            # Optional sanity check (comment out in production)
            assert F_val.requires_grad, "F_val does not require grad; grad mode likely disabled."

            # ∇_m F with graph for HVPs
            grad_m = autograd.grad(F_val, m_req, create_graph=True)[0]  # (B,E)

            # HVP: v -> H v  (plus damping)
            def Hmv(v):
                gv = (grad_m * v).sum()
                Hv = autograd.grad(gv, m_req, retain_graph=True)[0]
                return Hv + damp * v

            # Conjugate-gradient solve (H + damp I) w = grad_out
            def cg_solve(Hmv, b, iters=100, tol=1e-6):
                x = torch.zeros_like(b)
                r = b.clone()
                p = r.clone()
                rs_old = (r*r).sum()
                for _ in range(iters):
                    Hp = Hmv(p)
                    denom = (p*Hp).sum() + 1e-12
                    alpha = rs_old / denom
                    x = x + alpha * p
                    r = r - alpha * Hp
                    rs_new = (r*r).sum()
                    if torch.sqrt(rs_new) < tol:
                        break
                    p = r + (rs_new/rs_old) * p
                    rs_old = rs_new
                return x

            # If grad_out can be None (shouldn't, but guard anyway)
            if grad_out is None:
                grad_out = torch.zeros_like(m_req)

            w = cg_solve(Hmv, grad_out)

            # φ = (∇_m F)^T w; dL/dx = -∇_x φ
            phi = (grad_m * w).sum()

            dL_dh = -autograd.grad(phi, h_req, retain_graph=True, allow_unused=False)[0]  # (B,E)
            dL_dJ = -autograd.grad(phi, J_req, retain_graph=False, allow_unused=False)[0] # (E,E)

        dL_dT = None
        # Return grads aligned with forward signature: (h, J, T, max_iter, tol, damp)
        return dL_dh, dL_dJ, dL_dT, None, None, None, None

