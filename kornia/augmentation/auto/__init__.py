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

"""Choose a policy that samples image augmentation operations automatically.

Each policy records the operations it chose in ``_params``, so a call can be replayed. To chain transforms of your
own choice, use ``kornia.augmentation.ImageSequential``.

Which class to use:

- :class:`AutoAugment`: choose one searched, dataset-specific sub-policy.
- :class:`RandAugment`: choose a fixed number of operations at one magnitude.
- :class:`TrivialAugment`: choose one operation, drawing magnitudes only for operations that have one.
- :class:`PolicyAugmentBase`: build a custom policy selection class.
- :class:`PolicySequential`: run the operations within one custom sub-policy.
"""

from .autoaugment import AutoAugment
from .base import PolicyAugmentBase
from .operations import PolicySequential
from .rand_augment import RandAugment
from .trivial_augment import TrivialAugment

__all__ = ["AutoAugment", "PolicyAugmentBase", "PolicySequential", "RandAugment", "TrivialAugment"]
