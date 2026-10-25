import torch
from kornia.morphology import dilation, erosion

torch.manual_seed(123)

def check(name, fn, **kw):
    try:
        x = kw.pop('x')
        k = kw.pop('k')
        out_m = fn(x,k,engine='monotonic',**kw)
        out_u = fn(x,k,engine='unfold',**kw)
        print(f'{name}: ok={torch.allclose(out_m,out_u,atol=1e-5)} maxdiff={(out_m-out_u).abs().max().item()} shape_m={out_m.shape} shape_u={out_u.shape}')
        return out_m,out_u
    except Exception as e:
        print(f'{name}: ERR {type(e).__name__}: {e}')
        return None,None

x = torch.randn(1,1,16,16,dtype=torch.float32)
k3 = torch.ones(3,3)
k2 = torch.ones(2,2)
k5 = torch.ones(5,5)

# 1) even kernel
check('dilate k2 default origin', dilation, x=x, k=k2)
check('erode k2 default origin', erosion, x=x, k=k2)

# 2) off-center origin
check('dilate k3 origin [0,0]', dilation, x=x, k=k3, origin=[0,0])
check('dilate k3 origin [0,1]', dilation, x=x, k=k3, origin=[0,1])

# 3) border types
for bt in ['constant','reflect','replicate','circular']:
    check(f'dilate k3 border={bt}', dilation, x=x, k=k3, border_type=bt, border_value=0.0)

# 4) geodesic default
check('dilate k3 geodesic', dilation, x=x, k=k3, border_type='geodesic')

# 5) dtype promotion
x32 = torch.randn(1,1,12,12,dtype=torch.float32)
k64 = torch.ones(3,3,dtype=torch.float64)
check('dilate float32 img float64 ker', dilation, x=x32, k=k64)

# 6) kernel values !=1
k2v = torch.ones(3,3)*2.0
check('dilate kernel all 2.0', dilation, x=x, k=k2v)

# 7) non-square rectangular
krect = torch.ones(3,5)
check('dilate rect 3x5', dilation, x=x, k=krect)
