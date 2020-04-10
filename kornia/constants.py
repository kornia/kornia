from enum import Enum


class Resample(Enum):
    NEAREST = 0
    BILINEAR = 1
    BICUBIC = 2
