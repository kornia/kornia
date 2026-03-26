import torch.nn.functional as F


def linear_cka(x, y):
    x = x - x.mean(dim=1, keepdim=True)
    y = y - y.mean(dim=1, keepdim=True)

    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)

    return (x * y).sum(dim=1).mean().item()


def linear_similarity(f1, f2):
    """
    Applies CKA layer-wise
    """
    if f1.keys() != f2.keys():
        raise ValueError("Feature dictionaries must have same layers")

    results = {}

    for layer in f1:
        x = f1[layer].reshape(f1[layer].size(0), -1)
        y = f2[layer].reshape(f2[layer].size(0), -1)

        results[layer] = linear_cka(x, y)

    return results