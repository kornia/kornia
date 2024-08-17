"""Command-line interface for Kornia."""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch

from kornia.contrib import FaceDetector

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExportOnnxModelConfig:
    """Configuration for exporting a model to ONNX format."""

    model: torch.nn.Module
    img: torch.Tensor
    output_path: Path
    input_names: list[str]
    output_names: list[str]
    dynamic_axes: dict[str, dict[int, str]]


def export_onnx_model(config: ExportOnnxModelConfig) -> None:
    """Export a model to ONNX format."""
    torch.onnx.export(
        config.model,
        config.img,
        config.output_path,
        input_names=config.input_names,
        output_names=config.output_names,
        dynamic_axes=config.dynamic_axes,
    )

    if config.output_path.exists():
        logger.info("Model exported to %s", config.output_path)
    else:
        logger.error("Error exporting model to %s", config.output_path)


def export_onnx_model_config_resolver(model_name: str, output_path: Path) -> ExportOnnxModelConfig:
    """Resolve the configuration for exporting a model to ONNX format."""
    if model_name == "kornia/yunet":
        face_detector = FaceDetector()
        config = ExportOnnxModelConfig(
            model=face_detector.model,
            img=torch.rand(1, 3, 320, 320),
            output_path=output_path / "yunet.onnx",
            input_names=["images"],
            output_names=["loc", "conf", "iou"],
            dynamic_axes={"images": {0: "B"}},
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

    return config


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
            config: ExportOnnxModelConfig = export_onnx_model_config_resolver(args.model, args.output_path)
            export_onnx_model(config)
        else:
            logger.error("Format %s not supported", args.format)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
