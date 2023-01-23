from math import pi

import pytest
import torch
from torch.autograd import gradcheck

import kornia.testing as utils  # test utils
from kornia.feature.mkd import (
    COEFFS,
    EmbedGradients,
    ExplicitSpacialEncoding,
    MKDDescriptor,
    MKDGradients,
    SimpleKD,
    VonMisesKernel,
    Whitening,
    get_grid_dict,
    get_kron_order,
    spatial_kernel_embedding,
)
from kornia.testing import assert_close


@pytest.mark.parametrize("ps", [5, 13, 25])
def test_get_grid_dict(ps):
    grid_dict = get_grid_dict(ps)
    param_keys = ['x', 'y', 'phi', 'rho']
    assert set(grid_dict.keys()) == set(param_keys)
    for k in param_keys:
        assert grid_dict[k].shape == (ps, ps)


@pytest.mark.parametrize("d1,d2", [(1, 1), (1, 2), (2, 1), (5, 6)])
def test_get_kron_order(d1, d2):
    out = get_kron_order(d1, d2)
    assert out.shape == (d1 * d2, 2)


class TestMKDGradients:
    @pytest.mark.parametrize("ps", [5, 13, 25])
    def test_shape(self, ps, device):
        inp = torch.ones(1, 1, ps, ps).to(device)
        gradients = MKDGradients().to(device)
        out = gradients(inp)
        assert out.shape == (1, 2, ps, ps)

    @pytest.mark.parametrize("bs", [1, 5, 13])
    def test_batch_shape(self, bs, device):
        inp = torch.ones(bs, 1, 15, 15).to(device)
        gradients = MKDGradients().to(device)
        out = gradients(inp)
        assert out.shape == (bs, 2, 15, 15)

    def test_print(self, device):
        gradients = MKDGradients().to(device)
        gradients.__repr__()

    def test_toy(self, device):
        patch = torch.ones(1, 1, 6, 6).to(device).float()
        patch[0, 0, :, 3:] = 0
        gradients = MKDGradients().to(device)
        out = gradients(patch)
        expected_mags_1 = torch.Tensor([0, 0, 1.0, 1.0, 0, 0]).to(device)
        expected_mags = expected_mags_1.unsqueeze(0).repeat(6, 1)
        expected_oris_1 = torch.Tensor([-pi, -pi, 0, 0, -pi, -pi]).to(device)
        expected_oris = expected_oris_1.unsqueeze(0).repeat(6, 1)
        assert_close(out[0, 0, :, :], expected_mags, atol=1e-3, rtol=1e-3)
        assert_close(out[0, 1, :, :], expected_oris, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 13, 13
        patches = torch.rand(batch_size, channels, height, width).to(device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def grad_describe(patches):
            mkd_grads = MKDGradients()
            mkd_grads.to(device)
            return mkd_grads(patches)

        assert gradcheck(grad_describe, (patches), raise_exception=True, nondet_tol=1e-4, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 13, 13
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = MKDGradients().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(MKDGradients().to(patches.device, patches.dtype).eval())
        assert_close(model(patches), model_jit(patches))


class TestVonMisesKernel:
    @pytest.mark.parametrize("ps", [5, 13, 25])
    def test_shape(self, ps, device):
        inp = torch.ones(1, 1, ps, ps).to(device)
        vm = VonMisesKernel(patch_size=ps, coeffs=[0.38214156, 0.48090413]).to(device)
        out = vm(inp)
        assert out.shape == (1, 3, ps, ps)

    @pytest.mark.parametrize("bs", [1, 5, 13])
    def test_batch_shape(self, bs, device):
        inp = torch.ones(bs, 1, 15, 15).to(device)
        vm = VonMisesKernel(patch_size=15, coeffs=[0.38214156, 0.48090413]).to(device)
        out = vm(inp)
        assert out.shape == (bs, 3, 15, 15)

    @pytest.mark.parametrize("coeffs", COEFFS.values())
    def test_coeffs(self, coeffs, device):
        inp = torch.ones(1, 1, 15, 15).to(device)
        vm = VonMisesKernel(patch_size=15, coeffs=coeffs).to(device)
        out = vm(inp)
        assert out.shape == (1, 2 * len(coeffs) - 1, 15, 15)

    def test_print(self, device):
        vm = VonMisesKernel(patch_size=32, coeffs=[0.38214156, 0.48090413]).to(device)
        vm.__repr__()

    def test_toy(self, device):
        patch = torch.ones(1, 1, 6, 6).float().to(device)
        patch[0, 0, :, 3:] = 0
        vm = VonMisesKernel(patch_size=6, coeffs=[0.38214156, 0.48090413]).to(device)
        out = vm(patch)
        expected = torch.ones_like(out[0, 0, :, :]).to(device)
        assert_close(out[0, 0, :, :], expected * 0.6182, atol=1e-3, rtol=1e-3)

        expected = torch.Tensor([0.3747, 0.3747, 0.3747, 0.6935, 0.6935, 0.6935]).to(device)
        expected = expected.unsqueeze(0).repeat(6, 1)
        assert_close(out[0, 1, :, :], expected, atol=1e-3, rtol=1e-3)

        expected = torch.Tensor([0.5835, 0.5835, 0.5835, 0.0000, 0.0000, 0.0000]).to(device)
        expected = expected.unsqueeze(0).repeat(6, 1)
        assert_close(out[0, 2, :, :], expected, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, ps = 1, 1, 13
        patches = torch.rand(batch_size, channels, ps, ps).to(device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def vm_describe(patches, ps=13):
            vmkernel = VonMisesKernel(patch_size=ps, coeffs=[0.38214156, 0.48090413]).double()
            vmkernel.to(device)
            return vmkernel(patches.double())

        assert gradcheck(vm_describe, (patches, ps), raise_exception=True, nondet_tol=1e-4, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 13, 13
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = VonMisesKernel(patch_size=13, coeffs=[0.38214156, 0.48090413]).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(
            VonMisesKernel(patch_size=13, coeffs=[0.38214156, 0.48090413]).to(patches.device, patches.dtype).eval()
        )
        assert_close(model(patches), model_jit(patches))


class TestEmbedGradients:
    @pytest.mark.parametrize("ps,relative", [(5, True), (13, True), (25, True), (5, False), (13, False), (25, False)])
    def test_shape(self, ps, relative, device):
        inp = torch.ones(1, 2, ps, ps).to(device)
        emb_grads = EmbedGradients(patch_size=ps, relative=relative).to(device)
        out = emb_grads(inp)
        assert out.shape == (1, 7, ps, ps)

    @pytest.mark.parametrize("bs", [1, 5, 13])
    def test_batch_shape(self, bs, device):
        inp = torch.ones(bs, 2, 15, 15).to(device)
        emb_grads = EmbedGradients(patch_size=15, relative=True).to(device)
        out = emb_grads(inp)
        assert out.shape == (bs, 7, 15, 15)

    def test_print(self, device):
        emb_grads = EmbedGradients(patch_size=15, relative=True).to(device)
        emb_grads.__repr__()

    def test_toy(self, device):
        grads = torch.ones(1, 2, 6, 6).float().to(device)
        grads[0, 0, :, 3:] = 0
        emb_grads = EmbedGradients(patch_size=6, relative=True).to(device)
        out = emb_grads(grads)
        expected = torch.ones_like(out[0, 0, :, :3]).to(device)
        assert_close(out[0, 0, :, :3], expected * 0.3787, atol=1e-3, rtol=1e-3)
        assert_close(out[0, 0, :, 3:], expected * 0, atol=1e-3, rtol=1e-3)

    # TODO: review this test implementation
    # @pytest.mark.xfail(reason="RuntimeError: Jacobian mismatch for output 0 with respect to input 0,")
    def test_gradcheck(self, device):
        batch_size, channels, ps = 1, 2, 13
        patches = torch.rand(batch_size, channels, ps, ps).to(device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def emb_grads_describe(patches, ps=13):
            emb_grads = EmbedGradients(patch_size=ps, relative=True).double()
            emb_grads.to(device)
            return emb_grads(patches.double())

        assert gradcheck(emb_grads_describe, (patches, ps), raise_exception=True, nondet_tol=1e-4, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 2, 13, 13
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = EmbedGradients(patch_size=W, relative=True).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(
            EmbedGradients(patch_size=W, relative=True).to(patches.device, patches.dtype).eval()
        )
        assert_close(model(patches), model_jit(patches))


@pytest.mark.parametrize("kernel_type,d,ps", [('cart', 9, 9), ('polar', 25, 9), ('cart', 9, 16), ('polar', 25, 16)])
def test_spatial_kernel_embedding(kernel_type, ps, d):
    grids = get_grid_dict(ps)
    spatial_kernel = spatial_kernel_embedding(kernel_type, grids)
    assert spatial_kernel.shape == (d, ps, ps)


class TestExplicitSpacialEncoding:
    @pytest.mark.parametrize(
        "kernel_type,ps,in_dims", [('cart', 9, 3), ('polar', 9, 3), ('cart', 13, 7), ('polar', 13, 7)]
    )
    def test_shape(self, kernel_type, ps, in_dims, device):
        inp = torch.ones(1, in_dims, ps, ps).to(device)
        ese = ExplicitSpacialEncoding(kernel_type=kernel_type, fmap_size=ps, in_dims=in_dims).to(device)
        out = ese(inp)
        d_ = 9 if kernel_type == 'cart' else 25
        assert out.shape == (1, d_ * in_dims)

    @pytest.mark.parametrize(
        "kernel_type,bs", [('cart', 1), ('cart', 5), ('cart', 13), ('polar', 1), ('polar', 5), ('polar', 13)]
    )
    def test_batch_shape(self, kernel_type, bs, device):
        inp = torch.ones(bs, 7, 15, 15).to(device)
        ese = ExplicitSpacialEncoding(kernel_type=kernel_type, fmap_size=15, in_dims=7).to(device)
        out = ese(inp)
        d_ = 9 if kernel_type == 'cart' else 25
        assert out.shape == (bs, d_ * 7)

    @pytest.mark.parametrize("kernel_type", ['cart', 'polar'])
    def test_print(self, kernel_type, device):
        ese = ExplicitSpacialEncoding(kernel_type=kernel_type, fmap_size=15, in_dims=7).to(device)
        ese.__repr__()

    def test_toy(self, device):
        inp = torch.ones(1, 2, 6, 6).to(device).float()
        inp[0, 0, :, :] = 0
        cart_ese = ExplicitSpacialEncoding(kernel_type='cart', fmap_size=6, in_dims=2).to(device)
        out = cart_ese(inp)
        out_part = out[:, :9]
        expected = torch.zeros_like(out_part).to(device)
        assert_close(out_part, expected, atol=1e-3, rtol=1e-3)

        polar_ese = ExplicitSpacialEncoding(kernel_type='polar', fmap_size=6, in_dims=2).to(device)
        out = polar_ese(inp)
        out_part = out[:, :25]
        expected = torch.zeros_like(out_part).to(device)
        assert_close(out_part, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("kernel_type", ['cart', 'polar'])
    def test_gradcheck(self, kernel_type, device):
        batch_size, channels, ps = 1, 2, 13
        patches = torch.rand(batch_size, channels, ps, ps).to(device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def explicit_spatial_describe(patches, ps=13):
            ese = ExplicitSpacialEncoding(kernel_type=kernel_type, fmap_size=ps, in_dims=2)
            ese.to(device)
            return ese(patches)

        assert gradcheck(
            explicit_spatial_describe, (patches, ps), raise_exception=True, nondet_tol=1e-4, fast_mode=True
        )

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        B, C, H, W = 2, 2, 13, 13
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = (
            ExplicitSpacialEncoding(kernel_type='cart', fmap_size=W, in_dims=2).to(patches.device, patches.dtype).eval()
        )
        model_jit = torch.jit.script(
            ExplicitSpacialEncoding(kernel_type='cart', fmap_size=W, in_dims=2).to(patches.device, patches.dtype).eval()
        )
        assert_close(model(patches), model_jit(patches))


class TestWhitening:
    @pytest.mark.parametrize(
        "kernel_type,xform,output_dims",
        [
            ('cart', None, 3),
            ('polar', None, 3),
            ('cart', 'lw', 7),
            ('polar', 'lw', 7),
            ('cart', 'pca', 9),
            ('polar', 'pca', 9),
        ],
    )
    def test_shape(self, kernel_type, xform, output_dims, device):
        in_dims = 63 if kernel_type == 'cart' else 175
        wh = Whitening(xform=xform, whitening_model=None, in_dims=in_dims, output_dims=output_dims).to(device)
        inp = torch.ones(1, in_dims).to(device)
        out = wh(inp)
        assert out.shape == (1, output_dims)

    @pytest.mark.parametrize("bs", [1, 3, 7])
    def test_batch_shape(self, bs, device):
        wh = Whitening(xform='lw', whitening_model=None, in_dims=175, output_dims=128).to(device)
        inp = torch.ones(bs, 175).to(device)
        out = wh(inp)
        assert out.shape == (bs, 128)

    def test_print(self, device):
        wh = Whitening(xform='lw', whitening_model=None, in_dims=175, output_dims=128).to(device)
        wh.__repr__()

    def test_toy(self, device):
        wh = Whitening(xform='lw', whitening_model=None, in_dims=175, output_dims=175).to(device)
        inp = torch.ones(1, 175).to(device).float()
        out = wh(inp)
        expected = torch.ones_like(inp).to(device) * 0.0756
        assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, in_dims = 1, 175
        patches = torch.rand(batch_size, in_dims).to(device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def whitening_describe(patches, in_dims=175):
            wh = Whitening(xform='lw', whitening_model=None, in_dims=in_dims).double()
            wh.to(device)
            return wh(patches.double())

        assert gradcheck(whitening_describe, (patches, in_dims), raise_exception=True, nondet_tol=1e-4, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, in_dims = 1, 175
        patches = torch.rand(batch_size, in_dims).to(device)
        model = Whitening(xform='lw', whitening_model=None, in_dims=in_dims).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(
            Whitening(xform='lw', whitening_model=None, in_dims=in_dims).to(patches.device, patches.dtype).eval()
        )
        assert_close(model(patches), model_jit(patches))


class TestMKDDescriptor:
    dims = {'cart': 63, 'polar': 175, 'concat': 238}

    @pytest.mark.parametrize(
        "ps,kernel_type", [(9, 'concat'), (9, 'cart'), (9, 'polar'), (32, 'concat'), (32, 'cart'), (32, 'polar')]
    )
    def test_shape(self, ps, kernel_type, device):
        mkd = MKDDescriptor(patch_size=ps, kernel_type=kernel_type, whitening=None).to(device)
        inp = torch.ones(1, 1, ps, ps).to(device)
        out = mkd(inp)
        assert out.shape == (1, self.dims[kernel_type])

    @pytest.mark.parametrize(
        "ps,kernel_type,whitening",
        [
            (9, 'concat', 'lw'),
            (9, 'cart', 'lw'),
            (9, 'polar', 'lw'),
            (9, 'concat', 'pcawt'),
            (9, 'cart', 'pcawt'),
            (9, 'polar', 'pcawt'),
        ],
    )
    def test_whitened_shape(self, ps, kernel_type, whitening, device):
        mkd = MKDDescriptor(patch_size=ps, kernel_type=kernel_type, whitening=whitening).to(device)
        inp = torch.ones(1, 1, ps, ps).to(device)
        out = mkd(inp)
        output_dims = min(self.dims[kernel_type], 128)
        assert out.shape == (1, output_dims)

    @pytest.mark.parametrize("bs", [1, 3, 7])
    def test_batch_shape(self, bs, device):
        mkd = MKDDescriptor(patch_size=19, kernel_type='concat', whitening=None).to(device)
        inp = torch.ones(bs, 1, 19, 19).to(device)
        out = mkd(inp)
        assert out.shape == (bs, 238)

    def test_print(self, device):
        mkd = MKDDescriptor(patch_size=32, whitening='lw', training_set='liberty', output_dims=128).to(device)
        mkd.__repr__()

    def test_toy(self, device):
        inp = torch.ones(1, 1, 6, 6).to(device).float()
        inp[0, 0, :, :] = 0
        mkd = MKDDescriptor(patch_size=6, kernel_type='concat', whitening=None).to(device)
        out = mkd(inp)
        out_part = out[0, -28:]
        expected = torch.zeros_like(out_part).to(device)
        assert_close(out_part, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("whitening", [None, 'lw', 'pca'])
    def test_gradcheck(self, whitening, device):
        batch_size, channels, ps = 1, 1, 19
        patches = torch.rand(batch_size, channels, ps, ps).to(device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def mkd_describe(patches, patch_size=19):
            mkd = MKDDescriptor(patch_size=patch_size, kernel_type='concat', whitening=whitening).double()
            mkd.to(device)
            return mkd(patches.double())

        assert gradcheck(mkd_describe, (patches, ps), raise_exception=True, nondet_tol=1e-4, fast_mode=True)

    @pytest.mark.skip("neither dict, nor nn.ModuleDict works")
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels, ps = 1, 1, 19
        patches = torch.rand(batch_size, channels, ps, ps).to(device)
        kt = 'concat'
        wt = 'lw'
        model = MKDDescriptor(patch_size=ps, kernel_type=kt, whitening=wt).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(
            MKDDescriptor(patch_size=ps, kernel_type=kt, whitening=wt).to(patches.device, patches.dtype).eval()
        )
        assert_close(model(patches), model_jit(patches))


class TestSimpleKD:
    dims = {'cart': 63, 'polar': 175}

    @pytest.mark.parametrize("ps,kernel_type", [(9, 'cart'), (9, 'polar'), (32, 'cart'), (32, 'polar')])
    def test_shape(self, ps, kernel_type, device):
        skd = SimpleKD(patch_size=ps, kernel_type=kernel_type).to(device)
        inp = torch.ones(1, 1, ps, ps).to(device)
        out = skd(inp)
        assert out.shape == (1, min(128, self.dims[kernel_type]))

    @pytest.mark.parametrize("bs", [1, 3, 7])
    def test_batch_shape(self, bs, device):
        skd = SimpleKD(patch_size=19, kernel_type='polar').to(device)
        inp = torch.ones(bs, 1, 19, 19).to(device)
        out = skd(inp)
        assert out.shape == (bs, 128)

    def test_print(self, device):
        skd = SimpleKD(patch_size=19, kernel_type='polar').to(device)
        skd.__repr__()

    def test_gradcheck(self, device):
        batch_size, channels, ps = 1, 1, 19
        patches = torch.rand(batch_size, channels, ps, ps).to(device)
        patches = utils.tensor_to_gradcheck_var(patches)  # to var

        def skd_describe(patches, patch_size=19):
            skd = SimpleKD(patch_size=ps, kernel_type='polar', whitening='lw').double()
            skd.to(device)
            return skd(patches.double())

        assert gradcheck(skd_describe, (patches, ps), raise_exception=True, nondet_tol=1e-4, fast_mode=True)

    @pytest.mark.jit
    def test_jit(self, device, dtype):
        batch_size, channels, ps = 1, 1, 19
        patches = torch.rand(batch_size, channels, ps, ps).to(device)
        model = SimpleKD(patch_size=ps, kernel_type='polar', whitening='lw').to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(
            SimpleKD(patch_size=ps, kernel_type='polar', whitening='lw').to(patches.device, patches.dtype).eval()
        )
        assert_close(model(patches), model_jit(patches))
