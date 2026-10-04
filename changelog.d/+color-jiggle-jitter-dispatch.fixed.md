Fixed the eager backward performance of fixed-order `ColorJiggle` and `ColorJitter` by avoiding `torch.cond` dispatch.
