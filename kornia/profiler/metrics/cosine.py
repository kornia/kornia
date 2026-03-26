import torch.nn.functional as F


def cosine_similarity(f1, f2):
    """
    Layer-wise cosine similarity between feature dictionaries
    """
    if f1.keys() != f2.keys():
        raise ValueError("Feature dictionaries must have same layers")

    results = {}

    for layer in f1:
        x = f1[layer].reshape(f1[layer].size(0), -1)
        y = f2[layer].reshape(f2[layer].size(0), -1)

        sim = F.cosine_similarity(x, y, dim=1).mean().item()
        results[layer] = sim

    return results