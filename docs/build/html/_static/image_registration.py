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

import os

import cv2
import imageio
import torch

import kornia as K
import kornia.geometry as KG


def load_timg(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor."""
    assert os.path.isfile(file_name), f"Invalid file {file_name}"  # nosec
    # load image with OpenCV
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    # convert image to torch tensor
    tensor = K.image_to_tensor(img, None).float() / 255.0
    return K.color.bgr_to_rgb(tensor)


registrator = KG.ImageRegistrator("similarity")

img1 = K.resize(load_timg("/Users/oldufo/datasets/stewart/MR-CT/CT.png"), (400, 600))
img2 = K.resize(load_timg("/Users/oldufo/datasets/stewart/MR-CT/MR.png"), (400, 600))
model, intermediate = registrator.register(img1, img2, output_intermediate_models=True)

video_writer = imageio.get_writer("medical_registration.gif", fps=2)

timg_dst_first = img1.clone()
timg_dst_first[0, 0, :, :] = img2[0, 0, :, :]
video_writer.append_data(K.tensor_to_image((timg_dst_first * 255.0).byte()))

with torch.no_grad():
    for m in intermediate:
        timg_dst = KG.homography_warp(img1, m, img2.shape[-2:])
        timg_dst[0, 0, :, :] = img2[0, 0, :, :]
        video_writer.append_data(K.tensor_to_image((timg_dst_first * 255.0).byte()))
video_writer.close()
