import argparse
import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchgeometry as dgm


def load_homography(file_name):
    """Loads an homography from text file.
    """
    assert os.path.isfile(file_name), "Invalid file {}".format(file_name)
    return torch.from_numpy(np.loadtxt(file_name)).float()


def load_image(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor
    """
    assert os.path.isfile(file_name), "Invalid file {}".format(file_name)

    # load image with OpenCV
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)

    # convert image to torch tensor
    tensor = dgm.utils.image_to_tensor(img).float() / 255.
    tensor = tensor.view(1, *tensor.shape)  # 1xCxHxW

    return tensor, img


class MyHomography(nn.Module):
    def __init__(self):
        super(MyHomography, self).__init__()
        self.homo = nn.Parameter(torch.Tensor(3, 3))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.eye_(self.homo)

    def forward(self):
        return torch.unsqueeze(self.homo, dim=0)  # 1x3x3


def HomographyRegressionApp():
    # Training settings
    parser = argparse.ArgumentParser(description='Homography Regression with photometric loss.')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='the path to the directory with the input data.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='the path to output the results.')
    parser.add_argument('--num-iterations', type=int, default=1000, metavar='N',
                        help='number of training iterations (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=666, metavar='S',
                        help='random seed (default: 666)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-interval-vis', type=int, default=100, metavar='N',
                        help='how many batches to wait before visual logging training status')
    args = parser.parse_args()

    # define the device to use for inference
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)

    # load the data
    img_src, _ = load_image(os.path.join(args.input_dir, 'img1.ppm'))
    img_dst, _ = load_image(os.path.join(args.input_dir, 'img2.ppm'))
    dst_homo_src_gt = load_homography(os.path.join(args.input_dir, 'H1to2p'))
    
    # instantiate the homography warper from `torchgeometry`
    height, width = img_src.shape[-2:]
    warper = dgm.HomographyWarper(height, width)

    # create the homography as the parameter to be optimized
    dst_homo_src = MyHomography().to(device)

    # create optimizer
    optimizer = optim.Adam(dst_homo_src.parameters(), lr=args.lr)

    # main training loop

    for iter_idx in range(args.num_iterations):
        # send data to device
        img_src, img_dst = img_src.to(device), img_dst.to(device)

        # warp the reference image to the destiny with current homography
        img_src_to_dst = warper(img_src, dst_homo_src())

        # compute the photometric loss
        loss = F.l1_loss(img_src_to_dst, img_dst, reduction='none')

        # propagate the error just for a fixed window
        w_size = 100  # window size
        h_2, w_2 = height // 2, width // 2
        loss = loss[..., h_2 - w_size:h_2 + w_size, w_2 - w_size:w_2 + w_size]
        loss = torch.mean(loss)

        # compute gradient and update optimizer parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_idx % args.log_interval == 0:
            print('Train iteration: {}/{}\tLoss: {:.6}'.format(
                  iter_idx, args.num_iterations, loss.item()))
            print(dst_homo_src.homo)

        if iter_idx % args.log_interval_vis == 0:
            # merge warped and target image for visualization
            img_src_to_dst = warper(img_src, dst_homo_src())
            img_vis = 255. * (0.5 * img_src_to_dst + img_dst)
            # save warped image to disk
            file_name = os.path.join(
                args.output_dir, 'warped_{}.png'.format(iter_idx))
            cv2.imwrite(file_name, dgm.utils.tensor_to_image(img_vis))


if __name__ == "__main__":
     HomographyRegressionApp()
