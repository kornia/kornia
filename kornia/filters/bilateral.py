import math

import torch
import torch.nn as nn

import kornia


def gauss(x, sigma): #Function for Gaussian
  return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))
def calc(i,j, d, sigs, arr): #Defining boundaries
    r=d//2
    val=arr[i][j] # The pixel under consideration
    xtl=i-r
    if(xtl<0):
        xtl=0
    xbl=i+r
    if(xbl>=len(arr)):
        xbl=len(arr)-1
    ytl=j-r
    if(ytl<0):
        ytl=0
    ytr=j+r
    if(ytr>=len(arr[0])):
        ytr=len(arr[0])-1
    extract=arr[xtl:xbl+1, ytl: ytr+1] #get the window enclosed by boundary

    hold=torch.empty(xbl+1-xtl, ytr+1-ytl) #Initialize a window with the same size as extract

    hold.fill_(val) #Fill the window with same value as the pixel ubnder consideration

    hold=torch.subtract(extract, hold)
    hold.apply_(lambda x: gauss(x, sigs))

    new_hold=torch.mul(extract, hold)
    sm=torch.sum(new_hold)
    sum_deno=sm /torch.sum(hold)
    return sum_deno

def bilateral(
        input: torch.Tensor,
        kernel_size: int,
        sigma: int) -> torch.Tensor:

    arr=input[0][0] # The required part from the converted tensor
    store=torch.zeros(arr.size()[0], arr.size()[1]) #An empty tensor to store the results
    for i in range(0, len(arr)):
        for j in range(0, len(arr[i])):
            store[i][j]=calc(i,j,kernel_size,sigma,arr)
    return store
class Bilateralfilter(nn.module):
    def __init__(self, kernel_size: int,
                 sigma: int) -> None:
        super().__init__()
        self.kernel_size: int = kernel_size
        self.sigma: int = sigma
        self.border_type = border_type

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return bilateral(input, self.kernel_size, self.sigma)
