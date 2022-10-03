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
        torch.backends.cudnn.benchmark = True

    # create the video capture object
    cap = cv2.VideoCapture(0)

    # compute scale
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: h/w: {h}/{w} fps:{fps}")

    # create the video writer object
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(args.video_out, fourcc, fps, (w, h))

    # create the detector object
    edge_detection = EdgeDetector().to(device)
    edge_detection.load(args.weights_file)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    while True:

        # Capture the video frame
        # by frame
        _, frame = cap.read()

        start = cv2.getTickCount()

        # preprocess
        img = K.image_to_tensor(frame, keepdim=False).to(device)
        img = K.color.bgr_to_rgb(img.float())

        # detect !
        with torch.no_grad():
            edges = edge_detection(img)

        fps: float = cv2.getTickFrequency() / (cv2.getTickCount() - start)

        # show image

        frame_vis = K.tensor_to_image(edges.byte())

        frame_vis = cv2.putText(frame_vis, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

        # write the processed frame
        out.write(frame_vis)

        # Display the resulting frame
        cv2.imshow('frame', frame_vis)

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
    parser.add_argument('--weights_file', required=True, type=str, help='the file path to the trained weights.')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    args = parser.parse_args()
    my_app()
