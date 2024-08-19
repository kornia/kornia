"""Command-line interface for Kornia."""

import argparse
import logging
from pathlib import Path

from kornia.contrib.face_detection import YuFaceDetectNet

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def export_onnx_model_resolver(model_name: str, output_path: Path) -> None:
    """Resolve the configuration for exporting a model to ONNX format."""
    if "yunet" in model_name:
        onnx_model_path = output_path / "yunet.onnx"
        res = YuFaceDetectNet("test", pretrained=True).to_onnx(
            image_shape={"channels": 3, "height": 320, "width": 320},
            onnx_model_path=onnx_model_path,
            input_names=["images"],
            output_names=["loc", "conf", "iou"],
            dynamic_axes={"images": {0: "B"}},
        )
        if res:
            logger.info("Model exported to %s", onnx_model_path)
    else:
        raise ValueError(f"Model {model_name} not supported")


def main() -> None:
    """Main function for the Kornia CLI."""
    parser = argparse.ArgumentParser(description="Kornia CLI")

    subparsers = parser.add_subparsers(dest="command")

    # Create a subparser for the 'export' command
    export_parser = subparsers.add_parser("export", help="Export a model to different formats")
    export_parser.add_argument("--model", type=str, required=True, help="Model name to export")
    export_parser.add_argument(
        "--format",
        type=str,
        required=True,
        choices=["onnx"],
        help="Format to export the model",
    )
    export_parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path to save the exported model",
    )

    args = parser.parse_args()

    # Handle the 'export' command
    if args.command == "export":
        if args.format == "onnx":
            logger.info("Exporting model %s to ONNX format", args.model)
            export_onnx_model_resolver(args.model, args.output_path)
        else:
            logger.error("Format %s not supported", args.format)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
