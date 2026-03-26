import torch
import torch.nn.functional as F


def gram_matrix(fmap):
    B, C, H, W = fmap.shape
    fmap = fmap.reshape(B, C, -1)
    G = torch.bmm(fmap, fmap.transpose(1, 2))
    return G / (C * H * W)


def gram_similarity(f1, f2):
    """
    Gram-based similarity (only for 4D feature maps)
    """
    if f1.keys() != f2.keys():
        raise ValueError("Feature dictionaries must have same layers")

    results = {}

    for layer in f1:
        x = f1[layer]
        y = f2[layer]

        if x.dim() != 4 or y.dim() != 4:
            results[layer] = None
            continue

        G1 = gram_matrix(x)
        G2 = gram_matrix(y)

        sim = F.cosine_similarity(
            G1.reshape(G1.size(0), -1),
            G2.reshape(G2.size(0), -1),
            dim=1
        ).mean().item()

        results[layer] = sim

    return results