def squared_norm(m):
    assert m.shape[-1] == 3
    return m.pow(2).sum(-1, True)
