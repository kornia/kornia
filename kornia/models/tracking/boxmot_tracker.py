import os
from pathlib import Path
from typing import Union

from kornia.core import Tensor
from kornia.core.external import boxmot
from kornia.core.external import numpy as np
from kornia.models.detection.base import ObjectDetector
from kornia.models.detection.rtdetr import RTDETRDetectorBuilder
from kornia.utils.image import tensor_to_image

__all__ = ["BoxMotTracker"]


class BoxMotTracker:
    """BoxMotTracker is a module that wraps a detector and a tracker model.

    This module uses BoxMot library for tracking.

    Args:
        detector: ObjectDetector: The detector model.
        tracker_model_name: The name of the tracker model. Valid options are:
            - "BoTSORT"
            - "DeepOCSORT"
            - "OCSORT"
            - "HybridSORT"
            - "ByteTrack"
            - "StrongSORT"
            - "ImprAssoc"
        tracker_model_weights: Path to the model weights for ReID (Re-Identification).
        device: Device on which to run the model (e.g., 'cpu' or 'cuda').
        fp16: Whether to use half-precision (fp16) for faster inference on compatible devices.
        per_class: Whether to perform per-class tracking
        track_high_thresh: High threshold for detection confidence.
            Detections above this threshold are used in the first association round.
        track_low_thresh: Low threshold for detection confidence.
            Detections below this threshold are ignored.
        new_track_thresh: Threshold for creating a new track.
            Detections above this threshold will be considered as potential new tracks.
        track_buffer: Number of frames to keep a track alive after it was last detected.
        match_thresh: Threshold for the matching step in data association.
        proximity_thresh: Threshold for IoU (Intersection over Union) distance in first-round association.
        appearance_thresh: Threshold for appearance embedding distance in the ReID module.
        cmc_method: Method for correcting camera motion. Options include "sof" (simple optical flow).
        frame_rate: Frame rate of the video being processed. Used to scale the track buffer size.
        fuse_first_associate: Whether to fuse appearance and motion information during the first association step.
        with_reid: Whether to use ReID (Re-Identification) features for association.
    """

    def __init__(
        self,
        detector: Union[ObjectDetector, str] = "rtdetr_r18vd",
        tracker_model_name: str = "BoTSORT",
        tracker_model_weights: str = "osnet_x0_25_msmt17.pt",
        device: str = "cpu",
        fp16: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(detector, str):
            if detector.startswith("rtdetr"):
                detector = RTDETRDetectorBuilder.build(model_name=detector)
            else:
                raise ValueError(
                    f"Detector `{detector}` not available. You may pass an ObjectDetector instance instead."
                )
        self.detector = detector
        os.makedirs(".kornia_hub/models/boxmot", exist_ok=True)
        self.tracker = getattr(boxmot, tracker_model_name)(
            model_weights=Path(os.path.join(".kornia_hub/models/boxmot", tracker_model_weights)),
            device=device,
            fp16=fp16,
            **kwargs,
        )

    def update(self, image: Tensor) -> None:
        """Update the tracker with a new image.

        Args:
            image: The input image.
        """

        if not (image.ndim == 4 and image.shape[0] == 1) and not image.ndim == 3:
            raise ValueError(f"Input tensor must be of shape (1, 3, H, W) or (3, H, W). Got {image.shape}")

        if image.ndim == 3:
            image = image.unsqueeze(0)

        detections: Union[Tensor, list[Tensor]] = self.detector(image)

        detections = detections[0].cpu().numpy()  # Batch size is 1

        detections = np.array(  # type: ignore
            [
                detections[:, 2],  # type: ignore
                detections[:, 3],  # type: ignore
                detections[:, 2] + detections[:, 4],  # type: ignore
                detections[:, 3] + detections[:, 5],  # type: ignore
                detections[:, 1],  # type: ignore
                detections[:, 0],  # type: ignore
            ]
        ).T

        if detections.shape[0] == 0:  # type: ignore
            # empty N X (x, y, x, y, conf, cls)
            detections = np.empty((0, 6))  # type: ignore

        frame_raw = (tensor_to_image(image) * 255).astype(np.uint8)  # type: ignore

        # --> M X (x, y, x, y, id, conf, cls, ind)
        return self.tracker.update(detections, frame_raw)  # type: ignore

    def visualize(self, image: Tensor, show_trajectories: bool = True) -> np.ndarray:  # type: ignore
        """Visualize the results of the tracker.

        Args:
            image: The input image.
            show_trajectories: Whether to show the trajectories.

        Returns:
            The image with the results of the tracker.
        """
        frame_raw = (tensor_to_image(image) * 255).astype(np.uint8)
        self.update(image)
        self.tracker.plot_results(frame_raw, show_trajectories=show_trajectories)

        return frame_raw
