from torchkde import KernelDensity
import torch


def mutual_information(img_0:torch.Tensor, img_1:torch.Tensor, mask:torch.Tensor, n_bins:int =10)-> torch.Tensor:
    """Mutual information for two single channel images based on gaussian KDE.

    Parameters
    ----------
    img_0 : torch.Tensor
        _description_
    img_1 : torch.Tensor
        _description_
    mask : torch.Tensor
        _description_
    n_bins : int, optional
        _description_, by default 10

    Returns
    -------
    torch.Tensor
        scalar tensor representing the mutual information between the two images
    """
    if img_0.shape != img_1.shape:
        raise ValueError(f"The two input images should have the same shape, received {img_0.shape=} and {img_1.shape=}")
    data_0 = img_0[mask] # flattening here
    data_1 = img_1[mask]

    max_0 = img_0.max().detach()
    min_0 = img_0.min().detach()
    max_1 = img_1.max().detach()
    min_1 = img_1.min().detach()

    step_0 = (max_0 - min_0) / n_bins
    step_1 = (max_1 - min_1) / n_bins
    if step_0 == 0 or step_1 == 0:
        raise ValueError("One of your images contains no information: it has a constant grayscale value.")
    # to allow a uniform treatement in each axis for 2dim kde, we rescale data_1.
    factor = step_0 / step_1
    data_1 = data_1.unsqueeze(-1).float() * factor
    data_0 = data_0.unsqueeze(-1).float()
    joint = torch.cat([data_0, data_1], axis=-1)
    kde_joint = KernelDensity(bandwidth=step_0 / 2, kernel="gaussian")

    kde_joint.fit(joint)
    mesh_0 = torch.arange(min_0 + step_0 / 2, max_0, step_0)
    mesh_1 = torch.arange(min_0 * factor + step_0 / 2, max_0 * factor, step_0)
    mesh_joint = torch.cartesian_prod(mesh_0, mesh_1)

    # log of probabilities
    bin_vals_joint = torch.exp(kde_joint.score_samples(mesh_joint)).reshape((mesh_0.shape[0], mesh_1.shape[0]))
    support_mask = (bin_vals_joint > 0).detach()
    bin_vals_0 = torch.sum(bin_vals_joint, axis=1, keepdim=True).expand(*(support_mask.shape))[support_mask]
    bin_vals_1 = torch.sum(bin_vals_joint, axis=0, keepdim=True).expand(*(support_mask.shape))[support_mask]
    bin_vals_joint = bin_vals_joint[support_mask]
    logs = torch.log(bin_vals_joint / bin_vals_0 / bin_vals_1)
    terms = bin_vals_joint * logs
    return terms.sum()

