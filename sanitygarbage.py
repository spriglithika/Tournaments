import torch
from torch.linalg import svdvals
from Tournament import Tournament

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
t = Tournament(torch.tensor(100), mode='solver').to(device)   # use your real n if different
B = 2
x = torch.rand((B, t.num_edges), device=device)

# prepare the same gathered matrices used in forward
crd = t.crd.to(device)
starts = t.starts.to(device)
vecs = t.vecs.to(device)
idx = t.idx.to(device)  # int64
chis = starts[None, :, :] + x.unsqueeze(-1) * vecs[None, :, :]   # (B, e, d)
verts = torch.cat([crd.unsqueeze(0).expand(B, -1, -1), chis], dim=1)  # (B, n+e, d)

# build gathered rows and matrices
idx_expand = idx.view(1, -1, 1).expand(B, -1, t.euc_dim)
gathered = verts.gather(1, idx_expand)    # (B, n*d*d, d)
matrices = gathered.view(B, t.n * t.euc_dim, t.euc_dim, t.euc_dim)  # (B, n*d, d, d)

# flatten across batch for analysis
flat = matrices.reshape(-1, t.euc_dim, t.euc_dim).to(torch.float32)  # (N, d, d)

# 1) condition number summary
sv = svdvals(flat)  # (N,d)
cond = (sv[:, 0] / sv[:, -1].clamp(min=1e-12))
print('COND min, median, 90p, max:', float(cond.min()), float(cond.median()),
      float(cond.kthvalue(max(1, int(0.9*cond.numel()))).values), float(cond.max()))
print('counts > 1e6:', int((cond > 1e6).sum()), ' >1e8:', int((cond > 1e8).sum()))

# 2) how many unique rows per matrix (detect repetition)
N = flat.shape[0]
d = flat.shape[1]
unique_counts = []
for i in range(N):
    rows = flat[i]            # (d, d)
    # treat rows as flattened vectors; convert to cpu numpy for uniqueness stable test
    rows_np = rows.cpu().numpy()
    # count unique rows (within that matrix)
    uniq = len({tuple(r.round(8)) for r in rows_np})  # round to 1e-8 to group near-equal
    unique_counts.append(uniq)
unique_counts = torch.tensor(unique_counts, dtype=torch.int32)
print('unique rows per matrix: min/median/max:', int(unique_counts.min()), int(unique_counts.median()), int(unique_counts.max()))
print('how many matrices have < d unique rows (i.e. duplicates):', int((unique_counts < d).sum()))

# 3) show top offending matrices by cond
vals, inds = torch.topk(cond, k=min(10, cond.numel()))
print('Top condition values (top 10):')
for v, i in zip(vals.cpu().numpy(), inds.cpu().numpy()):
    print('index', int(i), 'cond', float(v), 'unique_rows', int(unique_counts[int(i)]))
    # show the row indices in verts corresponding to this matrix
    mat_idx = int(i)  # flat matrix index across batch
    # to map back: batch = mat_idx // (t.n * t.euc_dim) ; mat_in_batch = mat_idx % (t.n * t.euc_dim)
    batch = mat_idx // (t.n * t.euc_dim)
    mat_in_batch = mat_idx % (t.n * t.euc_dim)
    start_row = mat_in_batch * t.euc_dim
    # the original idx entries that formed this matrix:
    idx_block = idx[start_row:start_row + t.euc_dim]
    print(' idx block sample:', idx_block[:10].cpu().numpy())
    # show the corresponding verts rows for the first batch sample
    rows = gathered[batch, start_row:start_row + t.euc_dim, :5]  # show first 5 entries of each row
    print(' sample rows (first 5 cols):')
    print(rows.cpu().numpy())
    print('---')