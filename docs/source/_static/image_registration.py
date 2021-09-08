import cv2
import imageio
import matplotlib.pyplot as plt

import kornia as K
import kornia.geometry as KG


def load_timg(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor."""
    assert os.path.isfile(file_name), f"Invalid file {file_name}"
    # load image with OpenCV
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    # convert image to torch tensor
    tensor = K.image_to_tensor(img, None).float() / 255.
    return K.color.bgr_to_rgb(tensor)

registrator = KG.ImageRegistrator('similarity')

img1 = K.resize(load_timg('/Users/oldufo/datasets/stewart/MR-CT/CT.png'), (400,600))
img2 = K.resize(load_timg('/Users/oldufo/datasets/stewart/MR-CT/MR.png'), (400,600))
model, interm = registrator.register(img1, img2, output_intermediate_models=True)

video_writer = imageio.get_writer('medical_registration.gif', fps=2)

timg_dst_first = img1.clone()
timg_dst_first[0,0,:,:] = img2[0,0,:,:]
video_writer.append_data((K.tensor_to_image(timg_dst_first)*255.).astype(np.uint8))

with torch.no_grad():
    for m in interm:
        timg_dst = KG.homography_warp(img1, m, img2.shape[-2:])
        timg_dst[0,0,:,:] = img2[0,0,:,:]
        video_writer.append_data((K.tensor_to_image(timg_dst)*255.).astype(np.uint8))
video_writer.close()
