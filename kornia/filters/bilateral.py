import numpy as np
import math

# gaussian filter
def gfunc(x,y,sigma):
    return (math.exp(-(x**2 + y**2)/(2*(sigma**2))))/(2*3.14*(sigma**2))

def gaussFilter(size, sigma):
    out = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            out[i,j] = gfunc(i-size[0]//2,j-size[1]//2, sigma )
    return out/np.sum(out)

def bfunc(i,j,fw,fh,image,sigma1, sigma2, bilateralWFilter):
    imgwork = image[i - fh//2:i+1 + fh//2, j - fw//2:j+1 + fw//2, :]
    
    bilateralIFilter = ((imgwork - image[i, j,:])**2 )/(2*(sigma1**2))
    
    bilateralFilter = np.exp(-1*bilateralIFilter)*bilateralWFilter
    bilateralFilter = bilateralFilter/np.sum(bilateralFilter,axis=(0,1))
    return np.sum(np.multiply(imgwork, bilateralFilter),axis=(0,1))

def bilateral_blur(image,  kernel_size, sigma):
        size = image.shape
        fw, fh = kernel_size[0], kernel_size[1]
        sigma1, sigma2 = sigma[0], sigma[1]
        bilateral1 = 2*3.14*sigma2*sigma2*gaussFilter((fw,fh), sigma2)
        if len(image.shape) < 3  or image.shape[2] == 1:
            bilateralWFilter = np.resize(bilateral1,(*bilateral1.shape,1))
        else:
            bilateralWFilter = np.stack([bilateral1, bilateral1, bilateral1], axis=2)
        out = np.zeros((size[0]-2*fw +1,size[1]-2*fh +1,size[2]))
        for i in range(size[0]-2*fh +1):
            for j in range(size[1]-2*fw +1):
                out[i,j,:] = bfunc(i+fw-1, j+fh-1, fw, fh, image, sigma1, sigma2, bilateralWFilter)
        
        if id == 1:
            return np.resize(out, (out.shape[0], out.shape[1])).astype(np.uint8)
        else:
            return out.astype(np.uint8)

class BilateralFilter():
    def __init__(self, kernel_size, sigma):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __repr__(self):
        return (
            self.__class__.name__
            + '(kernel_size='
            + str(self.kernel_size)
            + ', '
            + 'sigma='
            + str(self.sigma)
             + ')'
        )
    def forward(self, image):
        "input image must be in numpy array"
        return bilateral_blur(image, self.kernel_size, self.sigma)
  


