`lovasz_softmax_loss` and `LovaszSoftmaxLoss` now compute the Jaccard gradient from each class foreground mask, correcting multiclass losses and gradients that previously depended on class numbering.
