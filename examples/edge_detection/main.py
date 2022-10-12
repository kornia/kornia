import argparse

import cv2
import torch

import kornia as K
from kornia.contrib import EdgeDetector


def my_app():
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
        img = K.color.bgr_to_rgb(img)
        return img

    # create the detector and find the faces !
    edge_detection = EdgeDetector().to(device)
    edge_detection.preprocess = my_preproces

    with torch.no_grad():
        edges = edge_detection(img)

    # show image

    img_vis = K.tensor_to_image(edges.byte())

    cv2.namedWindow('face_detection', cv2.WINDOW_NORMAL)
    cv2.imshow('face_detection', img_vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face and Landmark Detection')
    parser.add_argument('--image_file', required=True, type=str, help='the image file to be detected.')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    args = parser.parse_args()
    my_app()
