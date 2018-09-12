import torch


def image_to_tensor(image):
    """Converts a numpy image to a torch.Tensor image.
    """
    # TODO: add asserts and type checkings
    tensor = torch.from_numpy(image)
    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.permute(2, 0, 1)  # CxHxW


def tensor_to_image(tensor):
    """Converts a torch.Tensor image to a numpy image. In case the tensor is
       in the GPU, it will be copied back to CPU.
    """
    # TODO: add asserts and type checkings
    tensor = torch.squeeze(tensor)
    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, dim=0)
    return tensor.permute(1, 2, 0).contiguous().cpu().detach().numpy()
