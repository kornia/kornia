# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from kornia.augmentation._2d.geometric.affine import RandomAffine
from kornia.augmentation._2d.geometric.center_crop import CenterCrop
from kornia.augmentation._2d.geometric.crop import RandomCrop
from kornia.augmentation._2d.geometric.elastic_transform import RandomElasticTransform
from kornia.augmentation._2d.geometric.fisheye import RandomFisheye
from kornia.augmentation._2d.geometric.horizontal_flip import RandomHorizontalFlip
from kornia.augmentation._2d.geometric.pad import PadTo
from kornia.augmentation._2d.geometric.perspective import RandomPerspective
from kornia.augmentation._2d.geometric.resize import LongestMaxSize, Resize, SmallestMaxSize
from kornia.augmentation._2d.geometric.resized_crop import RandomResizedCrop
from kornia.augmentation._2d.geometric.rotation import RandomRotation, RandomRotation90
from kornia.augmentation._2d.geometric.shear import RandomShear
from kornia.augmentation._2d.geometric.thin_plate_spline import RandomThinPlateSpline
from kornia.augmentation._2d.geometric.translate import RandomTranslate
from kornia.augmentation._2d.geometric.vertical_flip import RandomVerticalFlip
