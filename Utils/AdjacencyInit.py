from preamble import *


def J_normed_diag_gamma(num_classes, num_edges, gamma=1.0, no_diag =False):
    edge_list = list(combinations(range(num_classes), 2))
    num_edges = len(edge_list)
    J = torch.zeros((num_edges, num_edges))

    for a, (i1, j1) in enumerate(edge_list):
        for b, (i2, j2) in enumerate(edge_list):
            if a == b:
                J[a, b] = 0 if no_diag else 1 # gamma  # self-interaction (optional)
                continue
            shared = {i1, j1}.intersection({i2, j2})
            if shared:
                shared_player = list(shared)[0]
                # Determine role of shared player in both edges
                same_role = ((shared_player == i1 and shared_player == i2) or
                            (shared_player == j1 and shared_player == j2))
                J[a, b] = gamma if same_role else -gamma
    # J = J / J.norm(p=2, dim=1, keepdim=True)
    J = J / J.norm(p=2)
    # print(J)
    return J # / 2


def factorize_J(J, d=20):
    # symmetric SVD (equivalent to eigendecomposition for symmetric matrices)
    eigvals, eigvecs = torch.linalg.eigh(J)

    # sort by absolute magnitude (largest first)
    idx = torch.argsort(eigvals.abs(), descending=True)
    eigvals = eigvals[idx][:d]          # (d,)
    eigvecs = eigvecs[:, idx][:, :d]    # (E, d)

    # square-root of eigenvalues (handle negative eigenvalues carefully)
    # If J has both positive and negative curvature:
    #   W captures sign
    #   sqrt(|lambda|) goes into Ä
    lam_abs_sqrt = eigvals.abs().sqrt()     # (d,)
    AE = eigvecs * lam_abs_sqrt             # (E, d)

    # W contains the eigenvalue signs
    W = torch.diag(torch.sign(eigvals))     # (d, d)

    return AE, W



def J_random_normed(num_classes, num_edges):
    edge_list = list(combinations(range(num_classes), 2))
    num_edges = len(edge_list)
    J = torch.randn((num_edges, num_edges))
    J = J / J.norm(p=2)
    return J