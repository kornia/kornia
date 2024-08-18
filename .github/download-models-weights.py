import argparse

import torch

fonts = {
    "sold2_wireframe": "http://cmp.felk.cvut.cz/~mishkdmy/models/sold2_wireframe.pth",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("WeightsDownloader")
    parser.add_argument("--target_directory", "-t", required=False, default="target_directory")

    args = parser.parse_args()

    torch.hub.set_dir(args.target_directory)

    for name, url in fonts.items():
        print(f"Downloading weights of `{name}` from `url`. Caching to dir `{args.target_directory}`")
        torch.hub.load_state_dict_from_url(url, model_dir=args.target_directory, map_location=torch.device("cpu"))

    raise SystemExit(0)
