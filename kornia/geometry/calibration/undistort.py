import torch


# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/distortion_model.hpp#L75
def inverseTiltProjection(taux, tauy):
    r"""Estimate the inverse of the tilt projection matrix
    
    Args:
        taux (torch.Tensor): Rotation angle in radians around the :math:`x`-axis.
        tauy (torch.Tensor): Rotation angle in radians around the :math:`y`-axis.

    Returns:
        torch.Tensor: Inverse tilt projection matrix.
    """

    if not torch.is_tensor(taux): taux = torch.tensor(taux)
    if not torch.is_tensor(tauy): tauy = torch.tensor(tauy)

    Rx = torch.tensor([[1,0,0],[0,torch.cos(taux),torch.sin(taux)],[0,-torch.sin(taux),torch.cos(taux)]])
    Ry = torch.tensor([[torch.cos(tauy),0,-torch.sin(tauy)],[0,1,0],[torch.sin(tauy),0,torch.cos(tauy)]])

    R = Ry @ Rx
    invPz = torch.tensor([[1/R[2,2],0,R[0,2]/R[2,2]],[0,1/R[2,2],R[1,2]/R[2,2]],[0,0,1]])

    return R.T @ invPz

# Based on https://github.com/opencv/opencv/blob/master/modules/calib3d/src/undistort.dispatch.cpp#L265
def undistort_points(points: torch.Tensor, K: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    r"""Compensate for lens distortion a set of 2D image points. Radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are cover in this function.
    
    Args:
        points (torch.Tensor): Input image points with shape :math:`(N, 2)`.
        K (torch.Tensor): Intrinsic camera matrix with shape :math:`(3, 3)`.
        dist (torch.Tensor): Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(1, n)`, :math:`(n, 1)` or :math:`(n,)`

    Returns:
        torch.Tensor: Undistorted 2D points with shape :math:`(N, 2)`.
    """
    assert points.dim() >= 2 and points.shape[-1] == 2
    assert K.shape == (3, 3)
    assert dist.numel() in [4,5,8,12,14]
    assert dist.dim() == 1 or (dist.dim() == 2 and (dist.shape[0] == 1 or dist.shape[1] == 1))
	
    dist = dist.squeeze()
    n = 14 - dist.numel()
    if n != 0:
        dist = torch.cat([dist,torch.zeros(n)])

	
	# Convert 2D points from pixels to normalized camera coordinates
    x: torch.Tensor = (points[:,0] - K[0,2])/K[0,0]
    y: torch.Tensor = (points[:,1] - K[1,2])/K[1,1]
    
    # Compensate for tilt distortion
    if dist[12] != 0 or dist[13] != 0:
        invTilt = inverseTiltProjection(dist[12], dist[13])

        pointsUntilt = invTilt @ torch.stack([x,y,torch.ones(x.shape,dtype=x.dtype)],0)
        x = pointsUntilt[0]/pointsUntilt[2]
        y = pointsUntilt[1]/pointsUntilt[2]
	
	# Iteratively undistort points
    x0, y0 = x, y
    for _ in range(5):
        r2 = x*x + y*y

        inv_rad_poly = (1 + dist[5]*r2 + dist[6]*r2*r2 + dist[7]*r2**3)/(1 + dist[0]*r2 + dist[1]*r2*r2 + dist[4]*r2**3)
        deltaX = 2*dist[2]*x*y + dist[3]*(r2 + 2*x*x) + dist[8]*r2 + dist[9]*r2*r2
        deltaY = dist[2]*(r2 + 2*y*y) + 2*dist[3]*x*y + dist[10]*r2 + dist[11]*r2*r2

        x = (x0 - deltaX)*inv_rad_poly
        y = (y0 - deltaY)*inv_rad_poly
    
    # Covert points from normalized camera coordinates to pixel coordinates
    x = x*K[0,0] + K[0,2]
    y = y*K[1,1] + K[1,2]

    return torch.stack([x,y], 1)