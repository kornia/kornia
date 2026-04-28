"""Pydantic v2 parameter models for the seven rf-detr-critical transforms.

These models describe the constructor parameters for each transform and provide:
- ``model_validate`` — strict validation
- ``model_json_schema`` — JSON Schema export for agent skill / IDE autocomplete
- ``model_dump`` — round-trip via dict

The transform classes themselves are NOT modified by this PR. These models
are descriptors only; downstream tooling (catalog, serialization, scaffold)
uses them.
"""
from __future__ import annotations

try:
    from pydantic import BaseModel, Field, ConfigDict, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[misc, assignment]


if PYDANTIC_AVAILABLE:

    class _Base(BaseModel):
        model_config = ConfigDict(extra="forbid", frozen=True)

    class RandomHorizontalFlipParams(_Base):
        p: float = Field(0.5, ge=0.0, le=1.0)
        same_on_batch: bool = False
        keepdim: bool = False
        p_batch: float = Field(1.0, ge=0.0, le=1.0)

    class RandomVerticalFlipParams(_Base):
        p: float = Field(0.5, ge=0.0, le=1.0)
        same_on_batch: bool = False
        keepdim: bool = False
        p_batch: float = Field(1.0, ge=0.0, le=1.0)

    class RandomRotationParams(_Base):
        degrees: tuple[float, float] | float = Field(...)
        resample: str = "BILINEAR"
        same_on_batch: bool = False
        align_corners: bool = True
        p: float = Field(0.5, ge=0.0, le=1.0)
        keepdim: bool = False

        @field_validator("degrees")
        @classmethod
        def _check_degrees(cls, v):
            if isinstance(v, tuple):
                if len(v) != 2 or v[0] > v[1]:
                    raise ValueError("degrees tuple must be (low, high) with low <= high")
            elif isinstance(v, (int, float)):
                if v < 0:
                    raise ValueError("degrees scalar must be >= 0")
            return v

    class RandomAffineParams(_Base):
        degrees: tuple[float, float] | float = 0.0
        translate: tuple[float, float] | None = None
        scale: tuple[float, float] | None = None
        shear: tuple[float, float] | float | None = None
        resample: str = "BILINEAR"
        align_corners: bool = True
        p: float = Field(0.5, ge=0.0, le=1.0)
        keepdim: bool = False

    class ColorJiggleParams(_Base):
        brightness: tuple[float, float] | float = 0.0
        contrast: tuple[float, float] | float = 0.0
        saturation: tuple[float, float] | float = 0.0
        hue: tuple[float, float] | float = 0.0
        same_on_batch: bool = False
        p: float = Field(1.0, ge=0.0, le=1.0)
        keepdim: bool = False

    class RandomGaussianBlurParams(_Base):
        kernel_size: tuple[int, int] | int = Field(...)
        sigma: tuple[float, float] = Field(...)
        border_type: str = "reflect"
        same_on_batch: bool = False
        p: float = Field(0.5, ge=0.0, le=1.0)
        keepdim: bool = False
        separable: bool = True

        @field_validator("kernel_size")
        @classmethod
        def _check_kernel(cls, v):
            ks = v if isinstance(v, tuple) else (v, v)
            for k in ks:
                if k < 1 or k % 2 == 0:
                    raise ValueError(f"kernel_size must be odd and >= 1; got {ks}")
            return v

    class RandomGaussianNoiseParams(_Base):
        mean: float = 0.0
        std: float = Field(1.0, ge=0.0)
        same_on_batch: bool = False
        p: float = Field(0.5, ge=0.0, le=1.0)
        keepdim: bool = False


    PARAMS_BY_NAME: dict[str, type[BaseModel]] = {
        "RandomHorizontalFlip": RandomHorizontalFlipParams,
        "RandomVerticalFlip": RandomVerticalFlipParams,
        "RandomRotation": RandomRotationParams,
        "RandomAffine": RandomAffineParams,
        "ColorJiggle": ColorJiggleParams,
        "ColorJitter": ColorJiggleParams,  # alias — same shape
        "RandomGaussianBlur": RandomGaussianBlurParams,
        "RandomGaussianNoise": RandomGaussianNoiseParams,
    }

    __all__ = [
        "RandomHorizontalFlipParams", "RandomVerticalFlipParams",
        "RandomRotationParams", "RandomAffineParams",
        "ColorJiggleParams", "RandomGaussianBlurParams",
        "RandomGaussianNoiseParams", "PARAMS_BY_NAME",
        "PYDANTIC_AVAILABLE",
    ]

else:
    # Pydantic unavailable: stub minimal no-op equivalents so import doesn't break
    PARAMS_BY_NAME = {}
    __all__ = ["PARAMS_BY_NAME", "PYDANTIC_AVAILABLE"]
