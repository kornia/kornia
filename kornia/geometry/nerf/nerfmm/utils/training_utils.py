import os

import numpy as np
import random
import torch


def is_DataParallelModel(model):
    """
    :param model:   nn.Module or torch.nn.DataParallel(model)
    :return:        True/False
    """
    return hasattr(model, 'module')



def set_randomness(args):
    if args.true_rand is False:
        random.seed(args.rand_seed)
        np.random.seed(args.rand_seed)
        torch.manual_seed(args.rand_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)


def save_checkpoint(epoch, model, optimizer, path, ckpt_name='checkpoint'):
    savepath = os.path.join(path, ckpt_name+'.pth')

    if is_DataParallelModel(model):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    state = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)


def load_ckpt_to_net(ckpt_path, net, map_location=None, strict=True):
    if map_location is None:
        ckpt = torch.load(ckpt_path)
    else:
        ckpt = torch.load(ckpt_path, map_location=map_location)

    weights = ckpt['model_state_dict']

    if is_DataParallelModel(net):
        net.module.load_state_dict(weights, strict=strict)
    else:
        net.load_state_dict(weights, strict=strict)

    return net