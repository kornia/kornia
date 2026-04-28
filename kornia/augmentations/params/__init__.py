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

"""kornia.augmentations.params — Pydantic models describing transform parameters.

Currently scopes the seven rf-detr-critical transforms (PR-PV).
Mass migration to all transforms ships in a follow-up.
"""

from kornia.augmentations.params.rfdetr_seven import PARAMS_BY_NAME, PYDANTIC_AVAILABLE

if PYDANTIC_AVAILABLE:
    from kornia.augmentations.params.rfdetr_seven import (
        ColorJiggleParams,
        RandomAffineParams,
        RandomGaussianBlurParams,
        RandomGaussianNoiseParams,
        RandomHorizontalFlipParams,
        RandomRotationParams,
        RandomVerticalFlipParams,
    )

__all__ = ["PARAMS_BY_NAME", "PYDANTIC_AVAILABLE"]
