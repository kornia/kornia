import argparse
import os
import cv2
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchgeometry as dgm


def load_depth(file_name):
    """Loads the depth using the syntel SDK and converts to torch.Tensor
    """
    assert os.path.isfile(file_name), "Invalid file {}".format(file_name)
    import sintel_io
    depth = sintel_io.depth_read(file_name)
    return torch.from_numpy(depth).view(1, 1, *depth.shape).float()


def create_pinhole(intrinsic, extrinsic, height, width):
    pinhole = torch.zeros(12)
    pinhole[0] = intrinsic[0, 0]  # fx
    pinhole[1] = intrinsic[1, 1]  # fy
    pinhole[2] = intrinsic[0, 2]  # cx
    pinhole[3] = intrinsic[1, 2]  # cy
    pinhole[4] = height
    pinhole[5] = width
    # TODO: implement in torchgeometry
    rvec = cv2.Rodrigues(extrinsic[:3,:3])[0]
    pinhole[6] = rvec[0, 0]  # rx
    pinhole[7] = rvec[1, 0]  # rx
    pinhole[8] = rvec[2, 0]  # rx
    pinhole[9] = extrinsic[0, 3]   # tx
    pinhole[10] = extrinsic[1, 3]  # ty
    pinhole[11] = extrinsic[2, 3]  # tz
    return pinhole.view(1, -1)


def load_camera_data(file_name):
    """Loads the camera data using the syntel SDK and converts to torch.Tensor.
    """
    assert os.path.isfile(file_name), "Invalid file {}".format(file_name)
    import sintel_io
    intrinsic, extrinsic = sintel_io.cam_read(file_name)
    return intrinsic, extrinsic


def load_image(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor
    """
    assert os.path.isfile(file_name), "Invalid file {}".format(file_name)

    # load image with OpenCV
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)

    # convert image to torch tensor
    tensor = dgm.utils.image_to_tensor(img).float() / 255.
    return tensor.view(1, *tensor.shape)  # 1xCxHxW


def load_data(root_path, sequence_name, frame_id):
    # index paths
    file_name = 'frame_%04d' % (frame_id)
    image_file = os.path.join(root_path, 'clean', sequence_name,
                              file_name + '.png')
    depth_file = os.path.join(root_path, 'depth', sequence_name,
                              file_name + '.dpt')
    camera_file = os.path.join(root_path, 'camdata_left', sequence_name,
                              file_name + '.cam')
    # load the actual data
    image = load_image(image_file)
    depth = load_depth(depth_file)
    camera_data = load_camera_data(camera_file) 
    camera = create_pinhole(*camera_data, *image.shape[-2:])
    return image, depth, camera


def DepthWarperApp():
    parser = argparse.ArgumentParser(description='Warp images by depth application.')
    # data parameters
    parser.add_argument('--input-dir', type=str, required=True,
                        help='the path to the directory with the input data.')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='the path to output the results.')
    parser.add_argument('--sequence-name', type=str, default='alley_1',
                        help='the name of the sequence.')
    parser.add_argument('--frame-source-id', type=int, default=1,
                        help='the id for the source image in the sequence.')
    parser.add_argument('--frame-destination-id', type=int, default=2,
                        help='the id for the destination image in the sequence.')
    # device parameters
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=666, metavar='S',
                        help='random seed (default: 666)')
    args = parser.parse_args()

    # define the device to use for inference
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    torch.manual_seed(args.seed)

    # configure syntel SDK path
    root_path = os.path.abspath(args.input_dir)
    sys.path.append(os.path.join(root_path, 'sdk/python'))

    # load the data
    root_dir = os.path.join(root_path, 'training')
    img_src, depth_src, cam_src = load_data(root_dir, args.sequence_name,
                                            args.frame_source_id)
    img_dst, depth_dst, cam_dst = load_data(root_dir, args.sequence_name,
                                            args.frame_destination_id)
 
    # instantiate the homography warper from `torchgeometry`
    warper = dgm.DepthWarper(cam_src)
    warper.compute_homographies(cam_dst)

    # compute the inverse depth and warp the source image
    inv_depth_src = 1. / depth_src
    img_src_to_dst = warper(inv_depth_src, img_src)

    #import ipdb;ipdb.set_trace()
    img_vis_warped = 0.5 * img_src_to_dst + img_dst

    ## save warped image to disk
    file_name = os.path.join(args.output_dir, \
        'warped_{0}_to_{1}.png'.format(args.frame_source_id, \
                                       args.frame_destination_id))
    cv2.imwrite(file_name, dgm.utils.tensor_to_image(255. * img_vis_warped))


if __name__ == "__main__":
     DepthWarperApp()
