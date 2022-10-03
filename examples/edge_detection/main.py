"""Script that finds faces and blurs using FaceDetection and blurring APIs."""
import argparse

import cv2
import torch
import torch.nn as nn

import kornia as K
from kornia.contrib import EdgeDetector


def my_app(args):
    # select the device
    device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')

    # load the image and scale
    img_raw = cv2.imread(args.image_file, cv2.IMREAD_COLOR)

    # preprocess
    img = K.image_to_tensor(img_raw, keepdim=False).to(device)

    def my_preproces(img):
        img = img.float()
        img -= torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1).to(img)
        img = K.color.bgr_to_rgb(img)
        return img / 255.0

    # create the detector and find the faces !
    edge_detection = EdgeDetector().to(device)
    edge_detection.load(args.weights_file)
    edge_detection.preprocess = my_preproces

    with torch.no_grad():
        edges = edge_detection(img)

    # show image

    import pdb

    pdb.set_trace()
    edges = K.enhance.normalize_min_max(edges, 0.0, 255.0)
    img_vis = K.tensor_to_image(edges.byte())

    # save and show image
    cv2.imwrite(args.image_out, img_vis)

    cv2.namedWindow('face_detection', cv2.WINDOW_NORMAL)
    cv2.imshow('face_detection', img_vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face and Landmark Detection')
    parser.add_argument('--image_file', required=True, type=str, help='the image file to be detected.')
    parser.add_argument('--image_out', required=True, type=str, help='the file path to write the output.')
    parser.add_argument('--weights_file', required=True, type=str, help='the file path to the trained weights.')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    args = parser.parse_args()
    my_app(args)
