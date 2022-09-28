"""Script that finds faces and blurs using FaceDetection and blurring APIs."""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

import kornia as K
from kornia.contrib import FaceDetector, FaceDetectorResult, FaceKeypoint


def draw_keypoint(img: np.ndarray, det: FaceDetectorResult, kpt_type: FaceKeypoint) -> np.ndarray:
    kpt = det.get_keypoint(kpt_type).int().tolist()
    return cv2.circle(img, kpt, 2, (255, 0, 0), 2)


def scale_image(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = 1.0 * size / w
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def apply_blur_face(img: torch.Tensor, img_vis: np.ndarray, det: FaceDetectorResult):
    # crop the face
    x1, y1 = det.xmin.int(), det.ymin.int()
    x2, y2 = det.xmax.int(), det.ymax.int()
    roi = img[..., y1:y2, x1:x2]

    # apply blurring and put back to the visualisation image
    roi = K.filters.gaussian_blur2d(roi, (21, 21), (35.0, 35.0))
    roi = K.color.rgb_to_bgr(roi)
    img_vis[y1:y2, x1:x2] = K.tensor_to_image(roi)


def my_app(args):
    # select the device
    device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')

    # load the image and scale
    img_raw = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    img_raw = scale_image(img_raw, args.image_size)

    # preprocess
    img = K.image_to_tensor(img_raw, keepdim=False).to(device)
    img = K.color.bgr_to_rgb(img.float())

    # create the detector and find the faces !
    face_detection = FaceDetector().to(device)

    with torch.no_grad():
        dets = face_detection(img)
    dets = [FaceDetectorResult(o) for o in dets]

    # show image

    img_vis = img_raw.copy()

    for b in dets:
        if b.score < args.vis_threshold:
            continue

        # draw face bounding box
        img_vis = cv2.rectangle(img_vis, b.top_left.int().tolist(), b.bottom_right.int().tolist(), (0, 255, 0), 4)

        if args.blur_faces:
            apply_blur_face(img, img_vis, b)

        if args.vis_keypoints:
            # draw facial keypoints
            img_vis = draw_keypoint(img_vis, b, FaceKeypoint.EYE_LEFT)
            img_vis = draw_keypoint(img_vis, b, FaceKeypoint.EYE_RIGHT)
            img_vis = draw_keypoint(img_vis, b, FaceKeypoint.NOSE)
            img_vis = draw_keypoint(img_vis, b, FaceKeypoint.MOUTH_LEFT)
            img_vis = draw_keypoint(img_vis, b, FaceKeypoint.MOUTH_RIGHT)

            # draw the text score
            cx = int(b.xmin)
            cy = int(b.ymin + 12)
            img_vis = cv2.putText(img_vis, f"{b.score:.2f}", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    # save and show image
    cv2.imwrite(args.image_out, img_vis)

    cv2.namedWindow('face_detection', cv2.WINDOW_NORMAL)
    cv2.imshow('face_detection', img_vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face and Landmark Detection')
    parser.add_argument('--image_file', required=True, type=str, help='the image file to be detected.')
    parser.add_argument('--image_out', required=True, type=str, help='the file path to write the output.')
    parser.add_argument('--image_size', default=320, type=int, help='the image size to process.')
    parser.add_argument('--vis_threshold', default=0.8, type=float, help='visualization_threshold')
    parser.add_argument('--vis_keypoints', dest='vis_keypoints', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--blur_faces', dest='blur_faces', action='store_true')
    args = parser.parse_args()
    my_app(args)
