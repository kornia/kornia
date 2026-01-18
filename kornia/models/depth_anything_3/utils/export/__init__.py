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

# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from kornia.models.depth_anything_3.specs import Prediction
from kornia.models.depth_anything_3.utils.export.gs import export_to_gs_ply, export_to_gs_video

from .colmap import export_to_colmap
from .depth_vis import export_to_depth_vis
from .feat_vis import export_to_feat_vis
from .glb import export_to_glb
from .npz import export_to_mini_npz, export_to_npz


def export(
    prediction: Prediction,
    export_format: str,
    export_dir: str,
    **kwargs,
):
    if "-" in export_format:
        export_formats = export_format.split("-")
        for export_format in export_formats:
            export(prediction, export_format, export_dir, **kwargs)
        return  # Prevent falling through to single-format handling

    if export_format == "glb":
        export_to_glb(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "mini_npz":
        export_to_mini_npz(prediction, export_dir)
    elif export_format == "npz":
        export_to_npz(prediction, export_dir)
    elif export_format == "feat_vis":
        export_to_feat_vis(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "depth_vis":
        export_to_depth_vis(prediction, export_dir)
    elif export_format == "gs_ply":
        export_to_gs_ply(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "gs_video":
        export_to_gs_video(prediction, export_dir, **kwargs.get(export_format, {}))
    elif export_format == "colmap":
        export_to_colmap(prediction, export_dir, **kwargs.get(export_format, {}))
    else:
        raise ValueError(f"Unsupported export format: {export_format}")


__all__ = [
    export,
]
