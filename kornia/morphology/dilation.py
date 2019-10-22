import torch
import torch.nn as nn
import torch.nn.functional as F

class Dilation(nn.Module):
    def __init__(self, structuring_element: torch.Tensor) -> None:
        super(Dilation, self).__init__()
        self.structuring_element = structuring_element

    def forward(self, img: torch.tensor):
        return dilation(img, self.structuring_element)
        
 

def dilation(img: torch.Tensor, structuring_element: torch.Tensor):
    if not torch.is_tensor(img):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(img)}")
    if not torch.is_tensor(structuring_element):
        raise TypeError(f"Structuring element type is not a torch.Tensor. Got {type(structuring_element)}")
    img_shape = img.shape
    if not (len(img_shape) == 3 or len(img_shape) == 4):
        raise ValueError(f"Expected input tensor to be of ndim 3 or 4, but got {len(img_shape)}")
    if len(img_shape) == 3:
        # unsqueeze introduces a batch dimension
        img = img.unsqueeze(0)
    else:
        if(img_shape[1] != 1):
            raise ValueError(f"Expected a single channel image, but got {img_shape[1]} channels")
    if len(structuring_element.shape) != 2:
        raise ValueError(f"Expected structuring element tensor to be of ndim=2, but got {len(structuring_element.shape)}")
    
    # Check if the input image is a binary containing only 0, 1
    unique_vals = torch.unique(img)
    if len(unique_vals) > 2:
        raise ValueError(f"Expected only 2 unique values in the tensor, since it should be binary, but got {len(torch.unique(img))}")
    if not ((unique_vals == 0) + (unique_vals == 1)).all():
        raise ValueError("Expected image to contain only 1's and 0's since it should be a binary image")

    # Convert structuring_element from shape [a, b] to [1, 1, a, b]
    structuring_element = structuring_element.unsqueeze(0).unsqueeze(0)
    
    se_shape = structuring_element.shape
    conv1 = F.conv2d(img, structuring_element, padding = (se_shape[2]//2, se_shape[2]//2))
    return (conv1 > 0).float()



   