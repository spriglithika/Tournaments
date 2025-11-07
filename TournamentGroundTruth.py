import torch
from itertools import combinations
from tqdm import tqdm
from Tournament import log_symmetric_cross_entropy as lsce

def get_gt_sparse_(num_classes):
    # first we need all permutations of two class labels
    perms = torch.tensor(list(combinations(range(num_classes), 2)), dtype=torch.int32)
    # now we create a tensor to hold the ground truth values
    gt_idx = torch.ones((num_classes, num_classes-1), dtype=torch.int32) * -1
    gt = torch.ones((num_classes, num_classes-1), dtype=torch.float32) * -1
    for i, (a, b) in tqdm(enumerate(perms)): #500, 0, 499
            gt_idx[a, gt[a].argmin()] = i
            gt_idx[b, gt[b].argmin()] = i
            gt[a, gt[a].argmin()] = 0.0
            gt[b, gt[b].argmin()] = 1.0
            # print(gt[a], gt_idx[a])
    return gt, gt_idx, perms

def get_gt_sparse():
    data = torch.load('ImageNetTournGT.pt')
    # Unpack into variables
    return data['gt_vals'], data['gt_idx'], data['perms']


def edge_loss_sparse(x, gt, y):
    gt_vals, gt_idx = gt
    preds = torch.zeros(x.shape[0], gt_vals.shape[-1])
    targets = torch.zeros_like(preds)
    for i, (x,y) in enumerate(zip(x,y)):
        preds[i] = x[gt_idx[y]]
        targets[i] = gt_vals[y]
    
    return lsce(preds, targets)

def get_gt(num_classes):
    # first we need all permutations of two class labels
    perms = torch.tensor(list(combinations(range(num_classes), 2)), dtype=torch.float32)
    # now we create a tensor to hold the ground truth values
    gt = torch.zeros((num_classes, len(perms)), dtype=torch.float32)
    for j in range(num_classes):
        for i, (a, b) in enumerate(perms):
            if a == j:
                gt[j, i] = 1.0
            elif b == j:
                gt[j, i] = -1.0
            else:
                gt[j, i] = 0.0
    return gt, perms

def main():
    num_classes = 1000
    gt_vals, gt_idx, perms = get_gt_sparse_(num_classes)
    torch.save({'gt_vals': gt_vals, 'gt_idx': gt_idx, 'perms': perms}, 'ImageNetTournGT.pt')

if __name__ == '__main__':
    main()