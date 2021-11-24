import argparse

import cv2
import kornia as K
import numpy as np

from kornia.contrib import FaceDetection, FaceKeypoint, FaceDetectionResults


def draw_keypoint(img: np.ndarray, det: FaceDetectionResults, kpt_type: FaceKeypoint) -> np.ndarray:
    kpt = det.get_keypoint(kpt_type).int().tolist()
    return cv2.circle(img, kpt, 2, (255, 0, 0), 2)


def my_app(args):
    img_raw = cv2.imread(args.image_file, cv2.IMREAD_COLOR)
    h, w = img_raw.shape[:2]
    scale = 1. * 320 / w
    img_raw = cv2.resize(img_raw, (int(w * scale), int(h * scale)))

    # preprocess
    img = K.image_to_tensor(img_raw, keepdim=False).float()
    img = K.color.bgr_to_rgb(img)

    # detect
    face_detection = FaceDetection(pretrained=True)
    dets = face_detection(img)

    # show image

    img_vis = img_raw.copy()

    for b in dets:
        if b.score < 0.8:
            continue

        # draw face bounding box
        img_vis = cv2.rectangle(
            img_vis, b.top_left.int().tolist(), b.bottom_right.int().tolist(), (0, 255, 0), 2)

        # draw facial keypoints
        img_vis = draw_keypoint(img_vis, b, FaceKeypoint.EYE_LEFT)
        img_vis = draw_keypoint(img_vis, b, FaceKeypoint.EYE_RIGHT)
        img_vis = draw_keypoint(img_vis, b, FaceKeypoint.NOSE)
        img_vis = draw_keypoint(img_vis, b, FaceKeypoint.MOUTH_LEFT)
        img_vis = draw_keypoint(img_vis, b, FaceKeypoint.MOUTH_RIGHT)

        # draw the text score
        cx = int(b.xmin)
        cy = int(b.ymin + 12)
        img_vis = cv2.putText(
            img_vis, f"{b.score:.2f}", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

    cv2.namedWindow('face_detection', cv2.WINDOW_NORMAL)
    cv2.imshow('face_detection', img_vis)
    cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face and Landmark Detection')
    parser.add_argument('--image_file', default='', type=str, help='the image file to be detected')
    args = parser.parse_args()
    my_app(args)
