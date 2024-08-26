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
