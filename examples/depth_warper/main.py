import argparse
import os
import sys

import cv2
import torch

import kornia as dgm


def load_depth(file_name):
    """Load the depth using the syntel SDK and converts to torch.Tensor."""
    if not os.path.isfile(file_name):
        raise AssertionError(f"Invalid file {file_name}")
    import sintel_io

    depth = sintel_io.depth_read(file_name)
    return torch.from_numpy(depth).view(1, 1, *depth.shape).float()


def load_camera_data(file_name):
    """Load the camera data using the syntel SDK and converts to torch.Tensor."""
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
    tensor = dgm.utils.image_to_tensor(img).float() / 255.0
    return tensor.view(1, *tensor.shape)  # 1xCxHxW


def load_data(root_path, sequence_name, frame_id):
    # index paths
    file_name = 'frame_%04d' % (frame_id)
    image_file = os.path.join(root_path, 'clean', sequence_name, file_name + '.png')
    depth_file = os.path.join(root_path, 'depth', sequence_name, file_name + '.dpt')
    camera_file = os.path.join(root_path, 'camdata_left', sequence_name, file_name + '.cam')
    # load the actual data
    image = load_image(image_file)
    depth = load_depth(depth_file)
    # load camera data and create pinhole
    height, width = image.shape[-2:]
    intrinsics, extrinsics = load_camera_data(camera_file)
    camera = dgm.utils.create_pinhole(intrinsics, extrinsics, height, width)
    return image, depth, camera


def DepthWarperApp():
    parser = argparse.ArgumentParser(description='Warp images by depth application.')
    # data parameters
    parser.add_argument('--input-dir', type=str, required=True, help='the path to the directory with the input data.')
    parser.add_argument('--output-dir', type=str, required=True, help='the path to output the results.')
    parser.add_argument('--sequence-name', type=str, default='alley_1', help='the name of the sequence.')
    parser.add_argument('--frame-ref-id', type=int, default=1, help='the id for the reference image in the sequence.')
    parser.add_argument('--frame-i-id', type=int, default=2, help='the id for the image i in the sequence.')
    # device parameters
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=666, metavar='S', help='random seed (default: 666)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # configure syntel SDK path
    root_path = os.path.abspath(args.input_dir)
    sys.path.append(os.path.join(root_path, 'sdk/python'))

    # load the data
    root_dir = os.path.join(root_path, 'training')
    img_ref, depth_ref, cam_ref = load_data(root_dir, args.sequence_name, args.frame_ref_id)
    img_i, _, cam_i = load_data(root_dir, args.sequence_name, args.frame_i_id)

    # instantiate the homography warper from `kornia`
    warper = dgm.DepthWarper(cam_i)
    warper.compute_homographies(cam_ref)

    # compute the inverse depth and warp the source image
    inv_depth_ref = 1.0 / depth_ref
    img_i_to_ref = warper(inv_depth_ref, img_i)

    # generate occlusion mask
    mask = ((img_ref - img_i_to_ref).mean(1) < 1e-1).float()

    img_vis_warped = 0.5 * img_i_to_ref + img_ref
    img_vis_warped_masked = mask * (0.5 * img_i_to_ref + img_ref)

    # save warped image to disk
    file_name = os.path.join(args.output_dir, f'warped_{args.frame_i_id}_to_{args.frame_ref_id}.png')
    cv2.imwrite(file_name, dgm.utils.tensor_to_image(255.0 * img_vis_warped))
    cv2.imwrite(file_name + 'mask.png', dgm.utils.tensor_to_image(255.0 * mask))
    cv2.imwrite(file_name + 'warpedmask.png', dgm.utils.tensor_to_image(255.0 * img_vis_warped_masked))


if __name__ == "__main__":
    DepthWarperApp()
