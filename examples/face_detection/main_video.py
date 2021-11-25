import cv2
import numpy as np
import torch

import kornia as K
from kornia.contrib import FaceDetector, FaceDetectorResult, FaceKeypoint


def draw_keypoint(img: np.ndarray, det: FaceDetectorResult, kpt_type: FaceKeypoint) -> np.ndarray:
    kpt = det.get_keypoint(kpt_type).int().tolist()
    return cv2.circle(img, kpt, 2, (255, 0, 0), 2)


def my_app():
    # define a video capture object
    cap = cv2.VideoCapture(0)

    face_detection = FaceDetector(pretrained=True)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    while(True):

        # Capture the video frame
        # by frame
        _, frame = cap.read()

        h, w = frame.shape[:2]
        scale = 1. * 320 / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # preprocess
        img = K.image_to_tensor(frame, keepdim=False).float()
        img = K.color.bgr_to_rgb(img)

        # detect !
        with torch.no_grad():
            dets = face_detection(img)

        # show image

        frame_vis = frame.copy()

        for b in dets:
            if b.score < 0.8:
                continue

            # draw face bounding box
            frame_vis = cv2.rectangle(
                frame_vis, b.top_left.int().tolist(), b.bottom_right.int().tolist(), (0, 255, 0), 2)

            # draw facial keypoints
            frame_vis = draw_keypoint(frame_vis, b, FaceKeypoint.EYE_LEFT)
            frame_vis = draw_keypoint(frame_vis, b, FaceKeypoint.EYE_RIGHT)
            frame_vis = draw_keypoint(frame_vis, b, FaceKeypoint.NOSE)
            frame_vis = draw_keypoint(frame_vis, b, FaceKeypoint.MOUTH_LEFT)
            frame_vis = draw_keypoint(frame_vis, b, FaceKeypoint.MOUTH_RIGHT)

            # draw the text score
            cx = int(b.xmin)
            cy = int(b.ymin + 12)
            frame_vis = cv2.putText(
                frame_vis, f"{b.score:.2f}", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # Display the resulting frame
        cv2.imshow('frame', frame_vis)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    my_app()
