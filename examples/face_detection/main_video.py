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


def my_app():
    # select the device
    device = torch.device('cpu')
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True

    # create the video capture object
    cap = cv2.VideoCapture(0)

    # compute scale
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: h/w: {height}/{width} fps:{fps}")

    scale = 1.0 * args.image_size / width
    w, h = int(width * scale), int(height * scale)

    # create the video writer object
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(args.video_out, fourcc, fps, (w, h))

    # create the detector object
    face_detection = FaceDetector().to(device)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    draw_keypoints: bool = False

    while True:
        # Capture the video frame
        # by frame
        _, frame = cap.read()

        start = cv2.getTickCount()

        # preprocess
        frame = scale_image(frame, args.image_size)
        img = K.image_to_tensor(frame, keepdim=False).to(device)
        img = K.color.bgr_to_rgb(img.float())

        # detect !
        with torch.no_grad():
            dets = face_detection(img)
        dets = [FaceDetectorResult(o) for o in dets[0]]

        fps: float = cv2.getTickFrequency() / (cv2.getTickCount() - start)

        # show image

        frame_vis = frame.copy()

        frame_vis = cv2.putText(frame_vis, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        for b in dets:
            if b.score < args.vis_threshold:
                continue

            # draw face bounding box
            line_thickness = 2
            line_length = 10

            x1, y1 = b.top_left.int().tolist()
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1 + line_length, y1), (0, 255, 0), thickness=line_thickness)
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1, y1 + line_length), (0, 255, 0), thickness=line_thickness)

            x1, y1 = b.top_right.int().tolist()
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1 - line_length, y1), (0, 255, 0), thickness=line_thickness)
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1, y1 + line_length), (0, 255, 0), thickness=line_thickness)

            x1, y1 = b.bottom_right.int().tolist()
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1 - line_length, y1), (0, 255, 0), thickness=line_thickness)
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1, y1 - line_length), (0, 255, 0), thickness=line_thickness)

            x1, y1 = b.bottom_left.int().tolist()
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1 + line_length, y1), (0, 255, 0), thickness=line_thickness)
            frame_vis = cv2.line(frame_vis, (x1, y1), (x1, y1 - line_length), (0, 255, 0), thickness=line_thickness)

            if draw_keypoints:
                # draw facial keypoints
                frame_vis = draw_keypoint(frame_vis, b, FaceKeypoint.EYE_LEFT)
                frame_vis = draw_keypoint(frame_vis, b, FaceKeypoint.EYE_RIGHT)
                frame_vis = draw_keypoint(frame_vis, b, FaceKeypoint.NOSE)
                frame_vis = draw_keypoint(frame_vis, b, FaceKeypoint.MOUTH_LEFT)
                frame_vis = draw_keypoint(frame_vis, b, FaceKeypoint.MOUTH_RIGHT)

                # draw the text score and FPS
                pt = b.top_left.int().tolist()

                frame_vis = cv2.putText(
                    frame_vis, f"{b.score:.2f}", (pt[0], pt[1] - 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)
                )

        # write the processed frame
        out.write(frame_vis)

        # Display the resulting frame
        cv2.imshow('frame', frame_vis)

        # the 's' button is set as the
        # switching button to draw the face keypoints
        if cv2.waitKey(1) == ord('s'):
            draw_keypoints = not draw_keypoints

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) == ord('q'):
            break

    # After the loop release the cap and writing objects
    cap.release()
    out.release()

    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face and Landmark Detection')
    parser.add_argument('--video_out', required=True, type=str, help='the file path to write the output.')
    parser.add_argument('--image_size', default=320, type=int, help='the image size to process.')
    parser.add_argument('--vis_threshold', default=0.8, type=float, help='visualization_threshold')
    parser.add_argument('--vis_keypoints', dest='vis_keypoints', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    args = parser.parse_args()
    my_app()
