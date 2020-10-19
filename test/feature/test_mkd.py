import pytest
import kornia.testing as utils  # test utils

from torch.testing import assert_allclose
from torch.autograd import gradcheck
from kornia.feature.mkd import *


@pytest.mark.parametrize("ps", [5, 13, 25])
def test_get_grid_dict(ps):
    grid_dict = get_grid_dict(ps)
    param_keys = ['x', 'y', 'phi', 'rho']
    assert set(grid_dict.keys()) == set(param_keys)
    for k in param_keys:
        assert grid_dict[k].shape == (ps, ps)


@pytest.mark.parametrize("d1,d2",
                         [(1, 1), (1, 2), (2, 1), (5, 6)])
def test_get_kron_order(d1, d2):
    out = get_kron_order(d1, d2)
    assert out.shape == (d1 * d2, 2)


class TestMKDGradients:
    @pytest.mark.parametrize("ps", [5, 13, 25])
    def test_shape(self, ps, device):
        inp = torch.ones(1, 1, ps, ps, device=device)
        gradients = MKDGradients().to(device)
        out = gradients(inp)
        assert out.shape == (1, 2, ps, ps)

    @pytest.mark.parametrize("bs", [1, 5, 13])
    def test_batch_shape(self, bs, device):
        inp = torch.ones(bs, 1, 15, 15, device=device)
        gradients = MKDGradients().to(device)
        out = gradients(inp)
        assert out.shape == (bs, 2, 15, 15)

    def test_print(self, device):
        gradients = MKDGradients().to(device)
        gradients.__repr__()

    def test_toy(self, device):
        patch = torch.ones(1, 1, 6, 6, device=device).float()
        patch[0, 0, :, 3:] = 0
        gradients = MKDGradients().to(device)
        out = gradients(patch)
        expected = torch.Tensor([0, 0, 1., 1., 0, 0], device=device)
        expected_mags = expected.unsqueeze(0).repeat(6, 1)
        expected_oris = expected_mags * 0
        assert_allclose(out[0, 0, :, :], expected_mags, atol=1e-3, rtol=1e-3)
        assert_allclose(out[0, 1, :, :], expected_oris, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 13, 13
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def grad_describe(patches):
            return MKDGradients()(patches)
        assert gradcheck(grad_describe, (patches),
                         raise_exception=True, nondet_tol=1e-4)


class TestVonMisesKernel:
    @ pytest.mark.parametrize("ps", [5, 13, 25])
    def test_shape(self, ps, device):
        inp = torch.ones(1, 1, ps, ps, device=device)
        vm = VonMisesKernel(patch_size=ps,
                            coeffs=[0.38214156, 0.48090413]).to(device)
        out = vm(inp)
        assert out.shape == (1, 3, ps, ps)

    @ pytest.mark.parametrize("bs", [1, 5, 13])
    def test_batch_shape(self, bs, device):
        inp = torch.ones(bs, 1, 15, 15, device=device)
        vm = VonMisesKernel(patch_size=15,
                            coeffs=[0.38214156, 0.48090413]).to(device)
        out = vm(inp)
        assert out.shape == (bs, 3, 15, 15)

    @ pytest.mark.parametrize("coeffs", COEFFS.values())
    def test_coeffs(self, coeffs, device):
        inp = torch.ones(1, 1, 15, 15, device=device)
        vm = VonMisesKernel(patch_size=15,
                            coeffs=coeffs).to(device)
        out = vm(inp)
        assert out.shape == (1, 2 * len(coeffs) - 1, 15, 15)

    def test_print(self, device):
        vm = VonMisesKernel(patch_size=32,
                            coeffs=[0.38214156, 0.48090413]).to(device)
        vm.__repr__()

    def test_toy(self, device):
        patch = torch.ones(1, 1, 6, 6, device=device).float()
        patch[0, 0, :, 3:] = 0
        vm = VonMisesKernel(patch_size=6,
                            coeffs=[0.38214156, 0.48090413]).to(device)
        out = vm(patch)
        expected = torch.ones_like(out[0, 0, :, :], device=device)
        assert_allclose(out[0, 0, :, :], expected * 0.6182, atol=1e-3, rtol=1e-3)

        expected = torch.Tensor([0.3747, 0.3747, 0.3747,
                                 0.6935, 0.6935, 0.6935], device=device)
        expected = expected.unsqueeze(0).repeat(6, 1)
        assert_allclose(out[0, 1, :, :], expected, atol=1e-3, rtol=1e-3)

        expected = torch.Tensor([0.5835, 0.5835, 0.5835,
                                 0.0000, 0.0000, 0.0000], device=device)
        expected = expected.unsqueeze(0).repeat(6, 1)
        assert_allclose(out[0, 2, :, :], expected, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, ps = 1, 1, 13
        patches = torch.rand(batch_size, channels, ps, ps, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def vm_describe(patches, ps=13):
            return VonMisesKernel(patch_size=ps,
                                  coeffs=[0.38214156, 0.48090413]).double()(patches.double())
        assert gradcheck(vm_describe, (patches, ps),
                         raise_exception=True, nondet_tol=1e-4)


class TestEmbedGradients:

    @pytest.mark.parametrize("ps,relative", [(5, True), (13, True), (25, True),
                                             (5, False), (13, False), (25, False)])
    def test_shape(self, ps, relative, device):
        inp = torch.ones(1, 2, ps, ps, device=device)
        emb_grads = EmbedGradients(patch_size=ps,
                                   relative=relative).to(device)
        out = emb_grads(inp)
        assert out.shape == (1, 7, ps, ps)

    @pytest.mark.parametrize("bs", [1, 5, 13])
    def test_batch_shape(self, bs, device):
        inp = torch.ones(bs, 2, 15, 15, device=device)
        emb_grads = EmbedGradients(patch_size=15,
                                   relative=True).to(device)
        out = emb_grads(inp)
        assert out.shape == (bs, 7, 15, 15)

    def test_print(self, device):
        emb_grads = EmbedGradients(patch_size=15,
                                   relative=True).to(device)
        emb_grads.__repr__()

    def test_toy(self, device):
        grads = torch.ones(1, 2, 6, 6, device=device).float()
        grads[0, 0, :, 3:] = 0
        emb_grads = EmbedGradients(patch_size=6,
                                   relative=True).to(device)
        out = emb_grads(grads)
        expected = torch.ones_like(out[0, 0, :, :3], device=device)
        assert_allclose(out[0, 0, :, :3], expected * .3787, atol=1e-3, rtol=1e-3)
        assert_allclose(out[0, 0, :, 3:], expected * 0, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, ps = 1, 2, 13
        patches = torch.rand(batch_size, channels, ps, ps, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def emb_grads_describe(patches, ps=13):
            return EmbedGradients(patch_size=ps,
                                  relative=True).double()(patches.double())
        assert gradcheck(emb_grads_describe, (patches, ps),
                         raise_exception=True, nondet_tol=1e-4)


@pytest.mark.parametrize("kerneltype,d,ps", [('cart', 9, 9), ('polar', 25, 9),
                                             ('cart', 9, 16), ('polar', 25, 16)])
def test_spatial_kernel_embedding(kerneltype, ps, d):
    grids = get_grid_dict(ps)
    spatial_kernel = spatial_kernel_embedding(kerneltype, grids)
    assert spatial_kernel.shape == (d, ps, ps)


class TestExplicitSpacialEncoding:

    @pytest.mark.parametrize("kerneltype,ps,in_dims", [('cart', 9, 3),
                                                       ('polar', 9, 3), ('cart', 13, 7), ('polar', 13, 7)])
    def test_shape(self, kerneltype, ps, in_dims, device):
        inp = torch.ones(1, in_dims, ps, ps, device=device)
        ese = ExplicitSpacialEncoding(kerneltype=kerneltype,
                                      fmap_size=ps,
                                      in_dims=in_dims).to(device)
        out = ese(inp)
        d_ = 9 if kerneltype == 'cart' else 25
        assert out.shape == (1, d_ * in_dims)

    @pytest.mark.parametrize("kerneltype,bs", [('cart', 1), ('cart', 5), ('cart', 13),
                                               ('polar', 1), ('polar', 5), ('polar', 13)])
    def test_batch_shape(self, kerneltype, bs, device):
        inp = torch.ones(bs, 7, 15, 15, device=device)
        ese = ExplicitSpacialEncoding(kerneltype=kerneltype,
                                      fmap_size=15,
                                      in_dims=7).to(device)
        out = ese(inp)
        d_ = 9 if kerneltype == 'cart' else 25
        assert out.shape == (bs, d_ * 7)

    @pytest.mark.parametrize("kerneltype", ['cart', 'polar'])
    def test_print(self, kerneltype, device):
        ese = ExplicitSpacialEncoding(kerneltype=kerneltype,
                                      fmap_size=15,
                                      in_dims=7).to(device)
        ese.__repr__()

    def test_toy(self, device):
        inp = torch.ones(1, 2, 6, 6, device=device).float()
        inp[0, 0, :, :] = 0
        cart_ese = ExplicitSpacialEncoding(kerneltype='cart',
                                           fmap_size=6,
                                           in_dims=2).to(device)
        out = cart_ese(inp)
        out_part = out[:, :9]
        expected = torch.zeros_like(out_part, device=device)
        assert_allclose(out_part, expected, atol=1e-3, rtol=1e-3)

        polar_ese = ExplicitSpacialEncoding(kerneltype='polar',
                                            fmap_size=6,
                                            in_dims=2).to(device)
        out = polar_ese(inp)
        out_part = out[:, :25]
        expected = torch.zeros_like(out_part, device=device)
        assert_allclose(out_part, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("kerneltype", ['cart', 'polar'])
    def test_gradcheck(self, kerneltype, device):
        batch_size, channels, ps = 1, 2, 13
        patches = torch.rand(batch_size, channels, ps, ps, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def explicit_spatial_describe(patches, ps=13):
            return ExplicitSpacialEncoding(kerneltype=kerneltype,
                                           fmap_size=ps,
                                           in_dims=2)(patches)
        assert gradcheck(explicit_spatial_describe, (patches, ps),
                         raise_exception=True, nondet_tol=1e-4)


class TestWhitening:
    @pytest.mark.parametrize("kerneltype,xform,output_dims", [('cart', None, 3),
                                                              ('polar', None, 3), ('cart', 'lw', 7), ('polar', 'lw', 7),
                                                              ('cart', 'pca', 9), ('polar', 'pca', 9)])
    def test_shape(self, kerneltype, xform, output_dims, device):
        in_dims = 63 if kerneltype == 'cart' else 175
        wh = Whitening(xform=xform,
                       whitening_model=None,
                       in_dims=in_dims,
                       output_dims=output_dims).to(device)
        inp = torch.ones(1, in_dims, device=device)
        out = wh(inp)
        assert out.shape == (1, output_dims)

    @pytest.mark.parametrize("bs", [1, 3, 7])
    def test_batch_shape(self, bs, device):
        wh = Whitening(xform='lw',
                       whitening_model=None,
                       in_dims=175,
                       output_dims=128).to(device)
        inp = torch.ones(bs, 175, device=device)
        out = wh(inp)
        assert out.shape == (bs, 128)

    def test_print(self, device):
        wh = Whitening(xform='lw',
                       whitening_model=None,
                       in_dims=175,
                       output_dims=128).to(device)
        wh.__repr__()

    def test_toy(self, device):
        wh = Whitening(xform='lw',
                       whitening_model=None,
                       in_dims=175,
                       output_dims=175).to(device)
        inp = torch.ones(1, 175, device=device).float()
        out = wh(inp)
        expected = torch.ones_like(inp, device=device) * 0.0756
        assert_allclose(out, expected, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, in_dims = 1, 175
        patches = torch.rand(batch_size, in_dims, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def whitening_describe(patches, in_dims=175):
            return Whitening(xform='lw',
                             whitening_model=None,
                             in_dims=in_dims).double()(patches.double())
        assert gradcheck(whitening_describe, (patches, in_dims),
                         raise_exception=True, nondet_tol=1e-4)


class TestMKDDescriptor:
    dims = {'cart': 63, 'polar': 175, 'concat': 238}

    @pytest.mark.parametrize("ps,kerneltype", [(9, 'concat'),
                                               (9, 'cart'), (9, 'polar'), (32, 'concat'), (32, 'cart'), (32, 'polar')])
    def test_shape(self, ps, kerneltype, device):
        mkd = MKDDescriptor(patch_size=ps,
                            kerneltype=kerneltype,
                            whitening=None).to(device)
        inp = torch.ones(1, 1, ps, ps, device=device)
        out = mkd(inp)
        assert out.shape == (1, self.dims[kerneltype])

    @pytest.mark.parametrize("ps,kerneltype,whitening", [(9, 'concat', 'lw'),
                                                         (9, 'cart', 'lw'), (9, 'polar', 'lw'), (9, 'concat', 'pcawt'),
                                                         (9, 'cart', 'pcawt'), (9, 'polar', 'pcawt')])
    def test_whitened_shape(self, ps, kerneltype, whitening, device):
        mkd = MKDDescriptor(patch_size=ps,
                            kerneltype=kerneltype,
                            whitening=whitening).to(device)
        inp = torch.ones(1, 1, ps, ps, device=device)
        out = mkd(inp)
        output_dims = min(self.dims[kerneltype], 128)
        assert out.shape == (1, output_dims)

    @pytest.mark.parametrize("bs", [1, 3, 7])
    def test_batch_shape(self, bs, device):
        mkd = MKDDescriptor(patch_size=19,
                            kerneltype='concat',
                            whitening=None).to(device)
        inp = torch.ones(bs, 1, 19, 19, device=device)
        out = mkd(inp)
        assert out.shape == (bs, 238)

    def test_print(self, device):
        mkd = MKDDescriptor(patch_size=32,
                            whitening='lw',
                            training_set='liberty',
                            output_dims=128).to(device)
        mkd.__repr__()

    def test_toy(self, device):
        inp = torch.ones(1, 1, 6, 6, device=device).float()
        inp[0, 0, :, :] = 0
        mkd = MKDDescriptor(patch_size=6,
                            kerneltype='concat',
                            whitening=None).to(device)
        out = mkd(inp)
        out_part = out[0, -28:]
        expected = torch.zeros_like(out_part, device=device)
        assert_allclose(out_part, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("whitening", [None, 'lw', 'pca'])
    def test_gradcheck(self, whitening, device):
        batch_size, channels, ps = 1, 1, 19
        patches = torch.rand(batch_size, channels, ps, ps, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def mkd_describe(patches, patch_size=19):
            return MKDDescriptor(patch_size=patch_size,
                                 kerneltype='concat',
                                 whitening=whitening).double()(patches.double())
        assert gradcheck(mkd_describe, (patches, ps),
                         raise_exception=True, nondet_tol=1e-4)


class TestSimpleKD:
    dims = {'cart': 63, 'polar': 175}

    @pytest.mark.parametrize("ps,kerneltype", [(9, 'cart'), (9, 'polar'),
                                               (32, 'cart'), (32, 'polar')])
    def test_shape(self, ps, kerneltype, device):
        skd = SimpleKD(patch_size=ps,
                       kerneltype=kerneltype).to(device)
        inp = torch.ones(1, 1, ps, ps, device=device)
        out = skd(inp)
        assert out.shape == (1, min(128, self.dims[kerneltype]))

    @pytest.mark.parametrize("bs", [1, 3, 7])
    def test_batch_shape(self, bs, device):
        skd = SimpleKD(patch_size=19,
                       kerneltype='polar').to(device)
        inp = torch.ones(bs, 1, 19, 19, device=device)
        out = skd(inp)
        assert out.shape == (bs, 128)

    def test_print(self, device):
        skd = SimpleKD(patch_size=19, kerneltype='polar').to(device)
        skd.__repr__()

    def test_gradcheck(self, device):
        batch_size, channels, ps = 1, 1, 19
        patches = torch.rand(batch_size, channels, ps, ps, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def skd_describe(patches, patch_size=19):
            return SimpleKD(patch_size=patch_size, kerneltype='polar',
                            whitening='lw').double()(patches.double())
        assert gradcheck(skd_describe, (patches, ps),
                         raise_exception=True, nondet_tol=1e-4)
