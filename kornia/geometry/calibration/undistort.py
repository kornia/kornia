import torch


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/distortion_model.hpp#L75
def inverseTiltProjection(taux, tauy):
    Rx = torch.tensor([[1,0,0],[0,torch.cos(taux),torch.sin(taux)],[0,-torch.sin(taux),torch.cos(taux)]])
    Ry = torch.tensor([[torch.cos(tauy),0,-torch.sin(tauy)],[0,1,0],[torch.sin(tauy),0,torch.cos(tauy)]])

    R = Ry @ Rx
    invPz = torch.tensor([[1/R[2,2],0,R[0,2]/R[2,2]],[0,1/R[2,2],R[1,2]/R[2,2]],[0,0,1]])

    return R.T @ invPz

# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L265
def undistort_points(points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    '''
    Compensate for lens distortion 2D image points.
    
    Args:
		pts: (n,2) input image points.
		K: (3,3) intrinsic camera matrix.
		dist: distortion coefficients (k1,k2,p1,p2[,k3[,k4,k5,k6[,s1,s2,s3,s4[,tx,ty]]]]). This is
		a vector with 4, 5, 8, 12 or 14 elements with shape: (1,n), (n,1) or (n,)
	
	Returns:
		torch.Tensor: Undistorted 2D points with shape (n,2).
    '''
    assert points.dim() >= 2 and points.shape[-1] == 2
    assert K.shape == (3, 3)
    assert dist.numel() in [4,5,8,12,14]
    assert dist.dim() == 1 or (dist.dim() == 2 and (dist.shape[0] == 1 or dist.shape[1] == 1))

    dist = dist.squeeze()
    n = 14 - dist.numel()
    if n != 0:
        dist = torch.cat([dist,torch.zeros(n)])


    x = (points[:,0] - K[0,2])/K[0,0]
    y = (points[:,1] - K[1,2])/K[1,1]
    
    if dist[12] != 0 or dist[13] != 0:
        invTilt = inverseTiltProjection(dist[12], dist[13])

        pointsUntilt = invTilt @ torch.stack([x,y,torch.ones(x.shape,dtype=x.dtype)],0)
        x = pointsUntilt[0]/pointsUntilt[2]
        y = pointsUntilt[1]/pointsUntilt[2]

    x0, y0 = x, y
    for _ in range(5):
        r2 = x*x + y*y

        inv_rad_poly = (1 + dist[5]*r2 + dist[6]*r2*r2 + dist[7]*r2**3)/(1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2**3)
        deltaX = 2*dist[2]*x*y + dist[3]*(r2 + 2*x*x) + dist[8]*r2 + dist[9]*r2*r2
        deltaY = dist[2]*(r2 + 2*y*y) + 2*dist[3]*x*y + dist[10]*r2 + dist[11]*r2*r2

        x = (x0 - deltaX)*inv_rad_poly
        y = (y0 - deltaY)*inv_rad_poly
    
    x = x*K[0,0] + K[0,2]
    y = y*K[1,1] + K[1,2]

    return torch.stack([x,y], 1)
