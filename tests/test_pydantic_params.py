"""Tests for the Pydantic Params models for the rf-detr seven."""
import pytest
from kornia.augmentations.params import PARAMS_BY_NAME, PYDANTIC_AVAILABLE

if not PYDANTIC_AVAILABLE:
    pytest.skip("pydantic not installed", allow_module_level=True)


def test_params_dict_has_all_seven_plus_alias():
    expected = {
        "RandomHorizontalFlip", "RandomVerticalFlip", "RandomRotation",
        "RandomAffine", "ColorJiggle", "ColorJitter",
        "RandomGaussianBlur", "RandomGaussianNoise",
    }
    assert expected.issubset(set(PARAMS_BY_NAME.keys()))


@pytest.mark.parametrize("name", list(PARAMS_BY_NAME.keys()))
def test_each_params_class_emits_json_schema(name):
    cls = PARAMS_BY_NAME[name]
    schema = cls.model_json_schema()
    assert "properties" in schema
    assert "title" in schema


def test_horizontal_flip_validates_p_range():
    cls = PARAMS_BY_NAME["RandomHorizontalFlip"]
    cls(p=0.5)        # OK
    with pytest.raises(Exception):
        cls(p=1.5)
    with pytest.raises(Exception):
        cls(p=-0.1)


def test_random_rotation_requires_degrees():
    cls = PARAMS_BY_NAME["RandomRotation"]
    cls(degrees=15.0)
    cls(degrees=(-15.0, 15.0))
    with pytest.raises(Exception):
        cls()  # missing required


def test_gaussian_blur_kernel_size_must_be_odd():
    cls = PARAMS_BY_NAME["RandomGaussianBlur"]
    cls(kernel_size=3, sigma=(0.1, 2.0))     # OK
    cls(kernel_size=(5, 5), sigma=(0.1, 2.0))  # OK
    with pytest.raises(Exception):
        cls(kernel_size=4, sigma=(0.1, 2.0))


def test_gaussian_noise_std_nonnegative():
    cls = PARAMS_BY_NAME["RandomGaussianNoise"]
    cls(std=0.0)  # OK
    cls(std=0.5)
    with pytest.raises(Exception):
        cls(std=-0.1)


def test_color_jiggle_jitter_alias():
    a = PARAMS_BY_NAME["ColorJiggle"]
    b = PARAMS_BY_NAME["ColorJitter"]
    assert a is b


def test_round_trip_dict():
    cls = PARAMS_BY_NAME["RandomHorizontalFlip"]
    d = cls(p=0.7).model_dump()
    cls(**d)  # round-trip
