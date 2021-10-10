import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import kornia as tgm


def load_data(root_path, sequence_name, frame_id):
    # index paths
    file_name = 'frame_%04d' % (frame_id)
    image_file = os.path.join(root_path, 'clean', sequence_name, file_name + '.png')
    depth_file = os.path.join(root_path, 'depth', sequence_name, file_name + '.dpt')
    camera_file = os.path.join(root_path, 'camdata_left', sequence_name, file_name + '.cam')
    # load the actual data
    image_tensor = load_image(image_file)
    depth = load_depth(depth_file)
    # load camera data and create pinhole
    height, width = image_tensor.shape[-2:]
    intrinsics, extrinsics = load_camera_data(camera_file)
    camera = tgm.utils.create_pinhole(intrinsics, extrinsics, height, width)
    return image_tensor, depth, camera


def load_depth(file_name):
    """Load the depth using the sintel SDK and converts to torch.Tensor."""
    if not os.path.isfile(file_name):
        raise AssertionError(f"Invalid file {file_name}")
    import sintel_io

    depth = sintel_io.depth_read(file_name)
    return torch.from_numpy(depth).view(1, 1, *depth.shape).float()


def load_camera_data(file_name):
    """Load the camera data using the sintel SDK and converts to torch.Tensor."""
    if not os.path.isfile(file_name):
        raise AssertionError(f"Invalid file {file_name}")
    import sintel_io

    intrinsic, extrinsic = sintel_io.cam_read(file_name)
    return intrinsic, extrinsic


def load_image(file_name):
    """Load the image with OpenCV and converts to torch.Tensor."""
    if not os.path.isfile(file_name):
        raise AssertionError(f"Invalid file {file_name}")

    # load image with OpenCV
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)

    # convert image to torch tensor
    tensor = tgm.utils.image_to_tensor(img).float() / 255.0
    return tensor.view(1, *tensor.shape)  # 1xCxHxW


def clip_and_convert_tensor(tensor):
    """Convert the input torch.Tensor to OpenCV image,clip it to be between.

    [0, 255] and convert it to unit
    """
    img = tgm.utils.tensor_to_image(255.0 * tensor)  # convert tensor to numpy
    img_cliped = np.clip(img, 0, 255)  # clip and reorder the channels
    img = img_cliped.astype('uint8')  # convert to uint
    return img


class InvDepth(nn.Module):
    def __init__(self, height, width, min_depth=0.50, max_depth=25.0):
        super().__init__()
        self._min_range = 1.0 / max_depth
        self._max_range = 1.0 / min_depth

        self.w = nn.Parameter(self._init_weights(height, width))

    def _init_weights(self, height, width):
        r1 = self._min_range
        r2 = self._min_range + (self._max_range - self._min_range) * 0.1
        w_init = (r1 - r2) * torch.rand(1, 1, height, width) + r2
        return w_init

    def forward(self):
        return self.w.clamp(min=self._min_range, max=self._max_range)


def DepthRegressionApp():
    # data settings
    parser = argparse.ArgumentParser(description='Depth Regression with photometric loss.')
    parser.add_argument('--input-dir', type=str, required=True, help='the path to the directory with the input data.')
    parser.add_argument('--output-dir', type=str, required=True, help='the path to output the results.')
    parser.add_argument(
        '--num-iterations', type=int, default=1000, metavar='N', help='number of training iterations (default: 1000)'
    )
    parser.add_argument('--sequence-name', type=str, default='alley_1', help='the name of the sequence.')
    parser.add_argument('--frame-ref-id', type=int, default=1, help='the id for the reference image in the sequence.')
    parser.add_argument('--frame-i-id', type=int, default=2, help='the id for the image i in the sequence.')
    # optimization parameters
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    # device parameters
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=666, metavar='S', help='random seed (default: 666)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        metavar='N',
        help='how many batches to wait before logging training status',
    )
    parser.add_argument(
        '--log-interval-vis',
        type=int,
        default=100,
        metavar='N',
        help='how many batches to wait before visual logging training status',
    )
    args = parser.parse_args()

    # define the device to use for inference
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)

    # configure sintel SDK path
    root_path = os.path.abspath(args.input_dir)
    sys.path.append(os.path.join(root_path, 'sdk/python'))

    # load the data
    root_dir = os.path.join(root_path, 'training')
    img_ref, _, cam_ref = load_data(root_dir, args.sequence_name, args.frame_ref_id)
    img_i, _, cam_i = load_data(root_dir, args.sequence_name, args.frame_i_id)

    # instantiate the depth warper from `kornia`
    warper = tgm.DepthWarper(cam_i)
    warper.compute_homographies(cam_ref)

    # create the inverse depth as a parameter to be optimized
    height, width = img_ref.shape[-2:]
    inv_depth_ref = InvDepth(height, width).to(device)

    # create optimizer
    optimizer = optim.Adam(inv_depth_ref.parameters(), lr=args.lr)

    # send data to device
    img_ref, img_i = img_ref.to(device), img_i.to(device)

    # main training loop

    for iter_idx in range(args.num_iterations):
        # compute the inverse depth and warp the source image
        img_i_to_ref = warper(inv_depth_ref(), img_i)

        # compute the photometric loss
        loss = F.l1_loss(img_i_to_ref, img_ref, reduction='none')
        loss = torch.mean(loss)

        # compute gradient and update optimizer parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_idx % args.log_interval == 0 or iter_idx == args.num_iterations - 1:
            print(f'Train iteration: {iter_idx}/{args.num_iterations}\tLoss: {loss.item():.6}')

            if iter_idx % args.log_interval_vis == 0:
                # merge warped and target image for  visualization
                img_i_to_ref = warper(inv_depth_ref(), img_i)
                img_both_vis = 0.5 * (img_i_to_ref + img_ref)

                img_both_vis = clip_and_convert_tensor(img_both_vis)
                img_i_to_ref_vis = clip_and_convert_tensor(img_i_to_ref)
                inv_depth_ref_vis = tgm.utils.tensor_to_image(
                    inv_depth_ref() / (inv_depth_ref().max() + 1e-6)
                ).squeeze()
                inv_depth_ref_vis = np.clip(255.0 * inv_depth_ref_vis, 0, 255)
                inv_depth_ref_vis = inv_depth_ref_vis.astype('uint8')

                # save warped image and depth to disk
                def file_name(output_dir, file_name, iter_idx):
                    return os.path.join(output_dir, f"{file_name}_{iter_idx}.png")

                cv2.imwrite(file_name(args.output_dir, "warped", iter_idx), img_i_to_ref_vis)
                cv2.imwrite(file_name(args.output_dir, "warped_both", iter_idx), img_both_vis)
                cv2.imwrite(file_name(args.output_dir, "inv_depth_ref", iter_idx), inv_depth_ref_vis)


if __name__ == "__main__":
    DepthRegressionApp()
