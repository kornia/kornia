import cv2
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchgeometry as tgm


input_dir = '/data/depth/'
root_dir = os.path.join(input_dir, 'training')
sequence_name = 'alley_1'
frame_i_id = 3
frame_ref_id = 1
EPS = 1e-6
learning_rate = 1e-3
num_iterations = 400  
log_interval = 100  # print log every 200 iterations
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Using ', device)

def load_data(root_path, sequence_name, frame_id):
    # index paths                                                                                                                        
    file_name = 'frame_%04d' % (frame_id)
    image_file = os.path.join(root_path, 'clean', sequence_name,
                              file_name + '.png')
    depth_file = os.path.join(root_path, 'depth', sequence_name,
                              file_name + '.dpt')
    camera_file = os.path.join(root_path, 'camdata_left', sequence_name,
                               file_name + '.cam')
    # load the actual data                                                                                                               
    image_tensor = load_image(image_file)
    depth = load_depth(depth_file)
    # load camera data and create pinhole                                                                                                
    height, width = image_tensor.shape[-2:]
    intrinsics, extrinsics = load_camera_data(camera_file)
    camera = tgm.utils.create_pinhole(intrinsics, extrinsics, height, width)
    return image_tensor, depth, camera

def load_depth(file_name):
    """Loads the depth using the sintel SDK and converts to torch.Tensor                                                                 
    """
    assert os.path.isfile(file_name), "Invalid file {}".format(file_name)
    import sintel_io
    depth = sintel_io.depth_read(file_name)
    return torch.from_numpy(depth).view(1, 1, *depth.shape).float()

def load_camera_data(file_name):
    """Loads the camera data using the sintel SDK and converts to torch.Tensor.                                                          
    """
    assert os.path.isfile(file_name), "Invalid file {}".format(file_name)
    import sintel_io
    intrinsic, extrinsic = sintel_io.cam_read(file_name)
    return intrinsic, extrinsic

def load_image(file_name):
    """Loads the image with OpenCV and converts to torch.Tensor                                                                          
    """
    assert os.path.isfile(file_name), "Invalid file {}".format(file_name)

    # load image with OpenCV                                                                                                             
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)

    # convert image to torch tensor                                                                                                      
    tensor = tgm.utils.image_to_tensor(img).float() / 255.
    return tensor.view(1, *tensor.shape)  # 1xCxHxW 

def clip_and_convert_tensor(tensor):
    """convert the input torch.Tensor to OpenCV image,clip it to be between
    [0, 255] and convert it to unit
    """
    img = tgm.utils.tensor_to_image(255. * tensor) # convert tensor to numpy
    img_cliped = np.clip(img, 0, 255)[:,:,::-1] # clip and reorder the channels
    img = img_cliped.astype('uint') # convert to uint
    return img





# configure sintel SDK path                                                                                                          
root_path = os.path.abspath(input_dir)
sys.path.append(os.path.join(root_path, 'sdk/python'))


img_ref, depth_ref, cam_ref = load_data(root_dir, sequence_name, frame_ref_id)
img_i, _, cam_i = load_data(root_dir, sequence_name, frame_i_id)

class MyDepth(nn.Module):
    def __init__(self, height, width):
        super(MyDepth, self).__init__()
        self.depth = nn.Parameter(torch.Tensor(1, height, width))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.uniform_(self.depth)

    def forward(self):
        return torch.unsqueeze(self.depth, dim=0)  # 1xheightxwidth


height, width = img_ref.shape[2], img_ref.shape[3]
depth_ref = MyDepth(height, width).to(device)
optimizer = optim.Adam(depth_ref.parameters(), lr=learning_rate)
# send data to device
img_ref, img_i = img_ref.to(device), img_i.to(device)

warper = tgm.DepthWarper(cam_i)
warper.compute_homographies(cam_ref)




for iter_idx in range(num_iterations):

    # compute the inverse depth and warp the source image                                                                                
    inv_depth_ref = 1. / depth_ref()
    img_i_to_ref = warper(inv_depth_ref, img_i)

    # compute the photometric loss
    loss = F.l1_loss(img_i_to_ref, img_ref, reduction='none')

    # propagate the error just for a fixed window
    w_size = 100  # window size
    h_2, w_2 = height // 2, width // 2
    loss = loss[..., h_2 - w_size:h_2 + w_size, w_2 - w_size:w_2 + w_size]
    loss = torch.mean(loss)

    # compute gradient and update optimizer parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if iter_idx % log_interval == 0 or iter_idx == num_iterations-1:
        print('Train iteration: {}/{}\tLoss: {:.6}'.format(
        iter_idx, num_iterations, loss.item()))
        # merge warped and target image for visualization                                                                                
        inv_depth_ref = 1. / depth_ref()
        img_i_to_ref = warper(inv_depth_ref, img_i)
        
        img_vis = 255. * 0.5 * (img_i_to_ref + img_ref)
           
        # fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        # fig.set_figheight(15); fig.set_figwidth(15)
        # ax1.imshow(tgm.utils.tensor_to_image(img_vis)[:,:,::-1])
        # ax1.set_title('merge warped and ref image')
        # est_depth = 1 / (inv_depth_ref + EPS) 
        # ax2.imshow(tgm.utils.tensor_to_image(est_depth)[:,:,::-1])
        # ax2.set_title('Depth')
        # plt.show()
