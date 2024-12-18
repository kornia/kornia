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

import argparse
import os

import diffusers
import torch

models = {
    "sold2_wireframe": ("torchhub", "http://cmp.felk.cvut.cz/~mishkdmy/models/sold2_wireframe.pth"),
    "stabilityai/stable-diffusion-2-1": ("diffusers", "StableDiffusionPipeline"),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("WeightsDownloader")
    parser.add_argument("--target_directory", "-t", required=False, default="target_directory")

    args = parser.parse_args()

    torch.hub.set_dir(args.target_directory)
    # For HuggingFace model caching
    os.environ["HF_HOME"] = args.target_directory

    for name, (src, path) in models.items():
        if src == "torchhub":
            print(f"Downloading weights of `{name}` from `{path}`. Caching to dir `{args.target_directory}`")
            torch.hub.load_state_dict_from_url(path, model_dir=args.target_directory, map_location=torch.device("cpu"))
        elif src == "diffusers":
            print(f"Downloading `{name}` from diffusers. Caching to dir `{args.target_directory}`")
            if path == "StableDiffusionPipeline":
                diffusers.StableDiffusionPipeline.from_pretrained(
                    name, cache_dir=args.target_directory, device_map="balanced"
                )

    raise SystemExit(0)
