# LICENSE HEADER MANAGED BY add-license-header
#
# Copyright 2018 Kornia Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from math import pi
from unittest.mock import patch

import pytest
import torch

from kornia.core.exceptions import ShapeError
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

from testing.base import BaseTester, supports_replicate_padding


@pytest.fixture(autouse=True)
def mock_torch_hub_load_state_dict_from_url():
    with patch("torch.hub.load_state_dict_from_url") as mock_load:

        def side_effect(url, **kwargs):
            # Create a dummy model structure depending on the kernel type in the URL
            if "cart" in url:
                in_dims = 63
            elif "polar" in url:
                in_dims = 175
            elif "concat" in url:
                in_dims = 238
            else:
                in_dims = 256

            dummy_algo = {
                "mean": torch.zeros(in_dims),
                "eigvecs": torch.eye(in_dims),
                "eigvals": torch.ones(in_dims),
            }
            dummy_training_set = {"lw": dummy_algo, "pca": dummy_algo}
            return {
                "liberty": dummy_training_set,
                "notredame": dummy_training_set,
                "yosemite": dummy_training_set,
            }

        mock_load.side_effect = side_effect
        yield mock_load


@pytest.mark.parametrize("ps", [5, 13, 25])
def test_get_grid_dict(ps):
    grid_dict = get_grid_dict(ps)
    param_keys = ["x", "y", "phi", "rho"]
    assert set(grid_dict.keys()) == set(param_keys)
    for k in param_keys:
        assert grid_dict[k].shape == (ps, ps)


@pytest.mark.parametrize("d1,d2", [(1, 1), (1, 2), (2, 1), (5, 6)])
def test_get_kron_order(d1, d2):
    out = get_kron_order(d1, d2)
    assert out.shape == (d1 * d2, 2)


def test_get_kron_order_values_and_dtype():
    d1, d2 = 2, 3
    expected_output = torch.tensor(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
        ],
        dtype=torch.int64,
    )

    actual_output = get_kron_order(d1, d2)
    assert actual_output.dtype == torch.int64
    assert torch.equal(actual_output, expected_output)


class TestMKDGradients(BaseTester):
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
        self.assert_close(out[0, 0, :, :], expected_mags, atol=1e-3, rtol=1e-3)
        self.assert_close(out[0, 1, :, :], expected_oris, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, height, width = 1, 1, 13, 13
        patches = torch.rand(batch_size, channels, height, width, device=device, dtype=torch.float64)

        def grad_describe(patches):
            mkd_grads = MKDGradients()
            mkd_grads.to(device)
            return mkd_grads(patches)

        self.gradcheck(grad_describe, (patches), nondet_tol=1e-4)


class TestVonMisesKernel(BaseTester):
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
        self.assert_close(out[0, 0, :, :], expected * 0.6182, atol=1e-3, rtol=1e-3)

        expected = torch.Tensor([0.3747, 0.3747, 0.3747, 0.6935, 0.6935, 0.6935]).to(device)
        expected = expected.unsqueeze(0).repeat(6, 1)
        self.assert_close(out[0, 1, :, :], expected, atol=1e-3, rtol=1e-3)

        expected = torch.Tensor([0.5835, 0.5835, 0.5835, 0.0000, 0.0000, 0.0000]).to(device)
        expected = expected.unsqueeze(0).repeat(6, 1)
        self.assert_close(out[0, 2, :, :], expected, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, channels, ps = 1, 1, 13
        patches = torch.rand(batch_size, channels, ps, ps, device=device, dtype=torch.float64)

        def vm_describe(patches, ps=13):
            vmkernel = VonMisesKernel(patch_size=ps, coeffs=[0.38214156, 0.48090413]).double()
            vmkernel.to(device)
            return vmkernel(patches.double())

        self.gradcheck(vm_describe, (patches, ps), nondet_tol=1e-4)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 1, 13, 13
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = VonMisesKernel(patch_size=13, coeffs=[0.38214156, 0.48090413]).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(
            VonMisesKernel(patch_size=13, coeffs=[0.38214156, 0.48090413]).to(patches.device, patches.dtype).eval()
        )
        self.assert_close(model(patches), model_jit(patches))


class TestEmbedGradients(BaseTester):
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
        self.assert_close(out[0, 0, :, :3], expected * 0.3787, atol=1e-3, rtol=1e-3)
        self.assert_close(out[0, 0, :, 3:], expected * 0, atol=1e-3, rtol=1e-3)

    # TODO: review this test implementation
    # @pytest.mark.xfail(reason="RuntimeError: Jacobian mismatch for output 0 with respect to input 0,")
    def test_gradcheck(self, device):
        batch_size, channels, ps = 1, 2, 13
        patches = torch.rand(batch_size, channels, ps, ps, device=device, dtype=torch.float64)

        def emb_grads_describe(patches, ps=13):
            emb_grads = EmbedGradients(patch_size=ps, relative=True).double()
            emb_grads.to(device)
            return emb_grads(patches.double())

        self.gradcheck(emb_grads_describe, (patches, ps), nondet_tol=1e-4)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 2, 13, 13
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = EmbedGradients(patch_size=W, relative=True).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(
            EmbedGradients(patch_size=W, relative=True).to(patches.device, patches.dtype).eval()
        )
        self.assert_close(model(patches), model_jit(patches))


@pytest.mark.parametrize("kernel_type,d,ps", [("cart", 9, 9), ("polar", 25, 9), ("cart", 9, 16), ("polar", 25, 16)])
def test_spatial_kernel_embedding(kernel_type, ps, d):
    grids = get_grid_dict(ps)
    spatial_kernel = spatial_kernel_embedding(kernel_type, grids)
    assert spatial_kernel.shape == (d, ps, ps)


class TestExplicitSpacialEncoding(BaseTester):
    @pytest.mark.parametrize(
        "kernel_type,ps,in_dims", [("cart", 9, 3), ("polar", 9, 3), ("cart", 13, 7), ("polar", 13, 7)]
    )
    def test_shape(self, kernel_type, ps, in_dims, device):
        inp = torch.ones(1, in_dims, ps, ps).to(device)
        ese = ExplicitSpacialEncoding(kernel_type=kernel_type, fmap_size=ps, in_dims=in_dims).to(device)
        out = ese(inp)
        d_ = 9 if kernel_type == "cart" else 25
        assert out.shape == (1, d_ * in_dims)

    @pytest.mark.parametrize(
        "kernel_type,bs", [("cart", 1), ("cart", 5), ("cart", 13), ("polar", 1), ("polar", 5), ("polar", 13)]
    )
    def test_batch_shape(self, kernel_type, bs, device):
        inp = torch.ones(bs, 7, 15, 15).to(device)
        ese = ExplicitSpacialEncoding(kernel_type=kernel_type, fmap_size=15, in_dims=7).to(device)
        out = ese(inp)
        d_ = 9 if kernel_type == "cart" else 25
        assert out.shape == (bs, d_ * 7)

    @pytest.mark.parametrize("kernel_type", ["cart", "polar"])
    def test_print(self, kernel_type, device):
        ese = ExplicitSpacialEncoding(kernel_type=kernel_type, fmap_size=15, in_dims=7).to(device)
        ese.__repr__()

    @pytest.mark.parametrize("kernel_type", ["cart", "polar"])
    def test_exception(self, kernel_type, device):
        ese = ExplicitSpacialEncoding(kernel_type=kernel_type, fmap_size=15, in_dims=7).to(device)
        # a wrong channel count must raise rather than return a descriptor of the wrong width
        with pytest.raises(ValueError, match="we expect Bx7xHxW"):
            ese(torch.ones(2, 5, 15, 15, device=device))
        with pytest.raises(ValueError, match="we expect Bx7xHxW"):
            ese(torch.ones(7, 15, 15, device=device))

    def test_toy(self, device):
        inp = torch.ones(1, 2, 6, 6).to(device).float()
        inp[0, 0, :, :] = 0
        cart_ese = ExplicitSpacialEncoding(kernel_type="cart", fmap_size=6, in_dims=2).to(device)
        out = cart_ese(inp)
        out_part = out[:, :9]
        expected = torch.zeros_like(out_part).to(device)
        self.assert_close(out_part, expected, atol=1e-3, rtol=1e-3)

        polar_ese = ExplicitSpacialEncoding(kernel_type="polar", fmap_size=6, in_dims=2).to(device)
        out = polar_ese(inp)
        out_part = out[:, :25]
        expected = torch.zeros_like(out_part).to(device)
        self.assert_close(out_part, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("kernel_type", ["cart", "polar"])
    def test_gradcheck(self, kernel_type, device):
        batch_size, channels, ps = 1, 2, 13
        patches = torch.rand(batch_size, channels, ps, ps, device=device, dtype=torch.float64)

        def explicit_spatial_describe(patches, ps=13):
            ese = ExplicitSpacialEncoding(kernel_type=kernel_type, fmap_size=ps, in_dims=2)
            ese.to(device)
            return ese(patches)

        self.gradcheck(explicit_spatial_describe, (patches, ps), nondet_tol=1e-4)

    def test_jit(self, device, dtype):
        B, C, H, W = 2, 2, 13, 13
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = (
            ExplicitSpacialEncoding(kernel_type="cart", fmap_size=W, in_dims=2).to(patches.device, patches.dtype).eval()
        )
        model_jit = torch.jit.script(
            ExplicitSpacialEncoding(kernel_type="cart", fmap_size=W, in_dims=2).to(patches.device, patches.dtype).eval()
        )
        self.assert_close(model(patches), model_jit(patches))


class TestWhitening(BaseTester):
    @pytest.mark.parametrize(
        "kernel_type,xform,output_dims",
        [
            ("cart", None, 3),
            ("polar", None, 3),
            ("cart", "lw", 7),
            ("polar", "lw", 7),
            ("cart", "pca", 9),
            ("polar", "pca", 9),
        ],
    )
    def test_shape(self, kernel_type, xform, output_dims, device):
        in_dims = 63 if kernel_type == "cart" else 175
        wh = Whitening(xform=xform, whitening_model=None, in_dims=in_dims, output_dims=output_dims).to(device)
        inp = torch.ones(1, in_dims).to(device)
        out = wh(inp)
        assert out.shape == (1, output_dims)

    @pytest.mark.parametrize("bs", [1, 3, 7])
    def test_batch_shape(self, bs, device):
        wh = Whitening(xform="lw", whitening_model=None, in_dims=175, output_dims=128).to(device)
        inp = torch.ones(bs, 175).to(device)
        out = wh(inp)
        assert out.shape == (bs, 128)

    def test_print(self, device):
        wh = Whitening(xform="lw", whitening_model=None, in_dims=175, output_dims=128).to(device)
        wh.__repr__()

    def test_toy(self, device):
        wh = Whitening(xform="lw", whitening_model=None, in_dims=175, output_dims=175).to(device)
        inp = torch.ones(1, 175).to(device).float()
        out = wh(inp)
        expected = torch.ones_like(inp).to(device) * 0.0756
        self.assert_close(out, expected, atol=1e-3, rtol=1e-3)

    def test_gradcheck(self, device):
        batch_size, in_dims = 1, 175
        patches = torch.rand(batch_size, in_dims, device=device, dtype=torch.float64)

        def whitening_describe(patches, in_dims=175):
            wh = Whitening(xform="lw", whitening_model=None, in_dims=in_dims).double()
            wh.to(device)
            return wh(patches.double())

        self.gradcheck(whitening_describe, (patches, in_dims), nondet_tol=1e-4)

    def test_jit(self, device, dtype):
        batch_size, in_dims = 1, 175
        patches = torch.rand(batch_size, in_dims).to(device)
        model = Whitening(xform="lw", whitening_model=None, in_dims=in_dims).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(
            Whitening(xform="lw", whitening_model=None, in_dims=in_dims).to(patches.device, patches.dtype).eval()
        )
        self.assert_close(model(patches), model_jit(patches))

    @pytest.mark.parametrize("output_dims", [20, 40, 41, 50])
    def test_pcaws_shrinkage_does_not_depend_on_output_dims(self, device, dtype, output_dims):
        # pcaws replaces every kept eigenvalue by lambda' = (1 - alpha) * lambda + alpha, a shrinkage towards 1 with
        # weight alpha = eigvals[keval=40] of the full spectrum. alpha belongs to the whitening model, so keeping
        # fewer components only drops columns; output_dims <= 40 used to raise IndexError instead.
        in_dims, keval = 50, 40
        eigvals = 1.0 / torch.arange(1, in_dims + 1, dtype=torch.float64)  # decaying, alpha = 1 / 41
        eigvecs = torch.linspace(-1.0, 1.0, in_dims * in_dims, dtype=torch.float64).view(in_dims, in_dims)
        model = {"pca": {"mean": torch.zeros(in_dims), "eigvecs": eigvecs.float(), "eigvals": eigvals.float()}}
        wh = Whitening(xform="pcaws", whitening_model=model, in_dims=in_dims, output_dims=output_dims, keval=keval)
        wh = wh.to(device, dtype)
        alpha = eigvals[keval]
        expected = eigvecs[:, :output_dims] * ((1 - alpha) * eigvals[:output_dims] + alpha).pow(-0.5)
        self.assert_close(wh.evecs.data, expected.to(device, dtype))
        assert wh(torch.rand(3, in_dims, device=device, dtype=dtype)).shape == (3, output_dims)

    @pytest.mark.parametrize("keval", [40, -1, -31])
    def test_pcaws_needs_the_shrinkage_eigenvalue(self, device, keval):
        # A whitening model with no eigenvalue at index keval has no shrinkage weight: say so instead of IndexError.
        # A negative keval has no meaning for a spectrum; it used to index the one truncated to output_dims.
        in_dims = 30
        model = {"pca": {"mean": torch.zeros(in_dims), "eigvecs": torch.eye(in_dims), "eigvals": torch.ones(in_dims)}}
        with pytest.raises(ValueError, match=rf"keval={keval} .*30 eigenvalues \(0 <= keval < 30\)"):
            Whitening(xform="pcaws", whitening_model=model, in_dims=in_dims, output_dims=20, keval=keval)


class TestMKDDescriptor(BaseTester):
    dims = {"cart": 63, "polar": 175, "concat": 238}

    @pytest.mark.parametrize(
        "ps,kernel_type", [(9, "concat"), (9, "cart"), (9, "polar"), (32, "concat"), (32, "cart"), (32, "polar")]
    )
    def test_shape(self, ps, kernel_type, device):
        mkd = MKDDescriptor(patch_size=ps, kernel_type=kernel_type, whitening=None).to(device)
        inp = torch.ones(1, 1, ps, ps).to(device)
        out = mkd(inp)
        assert out.shape == (1, self.dims[kernel_type])

    @pytest.mark.parametrize(
        "ps,kernel_type,whitening",
        [
            (9, "concat", "lw"),
            (9, "cart", "lw"),
            (9, "polar", "lw"),
            (9, "concat", "pcawt"),
            (9, "cart", "pcawt"),
            (9, "polar", "pcawt"),
        ],
    )
    def test_whitened_shape(self, ps, kernel_type, whitening, device):
        mkd = MKDDescriptor(patch_size=ps, kernel_type=kernel_type, whitening=whitening).to(device)
        inp = torch.ones(1, 1, ps, ps).to(device)
        out = mkd(inp)
        output_dims = min(self.dims[kernel_type], 128)
        assert out.shape == (1, output_dims)

    def test_state_dict_from_before_the_moduledict_loads_strictly(self, device):
        # The stages used to live in a plain dict, so a saved state dict has no `feats.*` keys.
        # Those buffers are derived from the constructor arguments, so a strict load fills them in.
        mkd = MKDDescriptor(patch_size=19, kernel_type="concat", whitening=None).to(device)
        old = {k: v for k, v in mkd.state_dict().items() if not k.startswith("feats.")}
        assert len(old) < len(mkd.state_dict())
        fresh = MKDDescriptor(patch_size=19, kernel_type="concat", whitening=None).to(device)
        fresh.load_state_dict(old, strict=True)
        inp = torch.rand(2, 1, 19, 19, device=device)
        self.assert_close(fresh(inp), mkd(inp))
        # A saved `feats.*` key is still honoured over the freshly computed buffer.
        new = mkd.state_dict()
        key = next(k for k in new if k.startswith("feats."))
        new[key] = torch.zeros_like(new[key])
        fresh.load_state_dict(new, strict=True)
        assert bool((fresh.state_dict()[key] == 0).all())

    def test_partial_stage_state_dict_fails_strictly(self, device):
        # The legacy fill is for a state dict with *no* `feats.*` keys. One saved by this release
        # that lost a single stage buffer is a corrupt checkpoint, and a strict load has to say so
        # rather than silently fill the hole from the live instance.
        mkd = MKDDescriptor(patch_size=19, kernel_type="concat", whitening=None).to(device)
        partial = mkd.state_dict()
        key = next(k for k in partial if k.startswith("feats."))
        del partial[key]
        fresh = MKDDescriptor(patch_size=19, kernel_type="concat", whitening=None).to(device)
        with pytest.raises(RuntimeError, match=key.replace(".", r"\.")):
            fresh.load_state_dict(partial, strict=True)
        # non-strict still loads, as it does for any missing key
        fresh.load_state_dict(partial, strict=False)

    @pytest.mark.parametrize("bs", [1, 3, 7])
    def test_batch_shape(self, bs, device):
        mkd = MKDDescriptor(patch_size=19, kernel_type="concat", whitening=None).to(device)
        inp = torch.ones(bs, 1, 19, 19).to(device)
        out = mkd(inp)
        assert out.shape == (bs, 238)

    def test_print(self, device):
        mkd = MKDDescriptor(patch_size=32, whitening="lw", training_set="liberty", output_dims=128).to(device)
        mkd.__repr__()

    def test_toy(self, device):
        inp = torch.ones(1, 1, 6, 6).to(device).float()
        inp[0, 0, :, :] = 0
        mkd = MKDDescriptor(patch_size=6, kernel_type="concat", whitening=None).to(device)
        out = mkd(inp)
        out_part = out[0, -28:]
        expected = torch.zeros_like(out_part).to(device)
        self.assert_close(out_part, expected, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("whitening", [None, "lw", "pca"])
    def test_gradcheck(self, whitening, device):
        batch_size, channels, ps = 1, 1, 19
        patches = torch.rand(batch_size, channels, ps, ps, device=device, dtype=torch.float64)

        def mkd_describe(patches, patch_size=19):
            mkd = MKDDescriptor(patch_size=patch_size, kernel_type="concat", whitening=whitening).double()
            mkd.to(device)
            return mkd(patches.double())

        self.gradcheck(mkd_describe, (patches, ps), nondet_tol=1e-4)

    @pytest.mark.skip("neither dict, nor nn.ModuleDict works")
    def test_jit(self, device, dtype):
        batch_size, channels, ps = 1, 1, 19
        patches = torch.rand(batch_size, channels, ps, ps).to(device)
        kt = "concat"
        wt = "lw"
        model = MKDDescriptor(patch_size=ps, kernel_type=kt, whitening=wt).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(
            MKDDescriptor(patch_size=ps, kernel_type=kt, whitening=wt).to(patches.device, patches.dtype).eval()
        )
        self.assert_close(model(patches), model_jit(patches))

    @pytest.mark.parametrize("output_dims", [20, 40])
    def test_pcaws_with_at_most_40_output_dims(self, device, dtype, output_dims):
        # pcaws used to index the eigenvalue at keval=40 after truncating the spectrum to output_dims (IndexError).
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on this device")
        mkd = MKDDescriptor(patch_size=19, kernel_type="concat", whitening="pcaws", output_dims=output_dims)
        out = mkd.to(device, dtype)(torch.rand(2, 1, 19, 19, device=device, dtype=dtype))
        assert out.shape == (2, output_dims)
        assert out.dtype == dtype

    @pytest.mark.parametrize(
        ("shape", "message"),
        [
            ((2, 1, 23, 19), "dimension 2: expected 19, got 23"),
            ((2, 1, 19, 23), "dimension 3: expected 19, got 23"),
            ((2, 1, 32, 32), "dimension 2: expected 19, got 32"),
            ((2, 3, 19, 19), "dimension 1: expected 1, got 3"),
            ((1, 19, 19), "expected 4 dimensions, got 3"),
            ((1, 1, 1, 19, 19), "expected 4 dimensions, got 5"),
        ],
        ids=["height", "width", "both", "channels", "rank3", "rank5"],
    )
    def test_patches_must_be_grayscale_and_patch_size(self, device, dtype, shape, message):
        # The spatial encoding is precomputed for patch_size x patch_size grayscale patches. Another size used to fail
        # deep inside with torch's broadcast RuntimeError, and three channels with a message about five.
        mkd = MKDDescriptor(patch_size=19, kernel_type="concat", whitening=None).to(device, dtype)
        with pytest.raises(ShapeError, match=message):
            mkd(torch.rand(*shape, device=device, dtype=dtype))
        if supports_replicate_padding(device, dtype):
            assert mkd(torch.rand(0, 1, 19, 19, device=device, dtype=dtype)).shape == (0, 238)


class TestSimpleKD(BaseTester):
    dims = {"cart": 63, "polar": 175}

    @pytest.mark.parametrize("ps,kernel_type", [(9, "cart"), (9, "polar"), (32, "cart"), (32, "polar")])
    def test_shape(self, ps, kernel_type, device):
        skd = SimpleKD(patch_size=ps, kernel_type=kernel_type).to(device)
        inp = torch.ones(1, 1, ps, ps).to(device)
        out = skd(inp)
        assert out.shape == (1, min(128, self.dims[kernel_type]))

    @pytest.mark.parametrize("bs", [1, 3, 7])
    def test_batch_shape(self, bs, device):
        skd = SimpleKD(patch_size=19, kernel_type="polar").to(device)
        inp = torch.ones(bs, 1, 19, 19).to(device)
        out = skd(inp)
        assert out.shape == (bs, 128)

    def test_print(self, device):
        skd = SimpleKD(patch_size=19, kernel_type="polar").to(device)
        skd.__repr__()

    def test_gradcheck(self, device):
        batch_size, channels, ps = 1, 1, 19
        patches = torch.rand(batch_size, channels, ps, ps, device=device, dtype=torch.float64)

        def skd_describe(patches, patch_size=19):
            skd = SimpleKD(patch_size=ps, kernel_type="polar", whitening="lw").double()
            skd.to(device)
            return skd(patches.double())

        self.gradcheck(skd_describe, (patches, ps), nondet_tol=1e-4)


class TestConventionsMKDDescriptor(BaseTester):
    """Pin for the frame of MKD's kernels within the patch."""

    @pytest.mark.parametrize("kernel_type", ["concat", "polar", "cart"])
    def test_convention_mkd_kernels_are_centred_on_the_patch_centre(self, device, dtype, kernel_type):
        # The kernel grid spans [-1, 1] across the patch_size pixels, so its origin is pixel (patch_size - 1) / 2, as in
        # manyids2/mkd_pytorch (between two pixels at the even 32). Mirroring a patch about that centre maps the grid
        # onto itself and flips only the signs of the odd harmonics, so every entry keeps its magnitude; a grid half a
        # pixel off the centre does not. whitening=None needs no download and keeps the raw entries.
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        torch.manual_seed(0)
        patches = torch.rand(4, 1, 32, 32, device=device, dtype=dtype)
        mkd = MKDDescriptor(patch_size=32, kernel_type=kernel_type, whitening=None).to(device, dtype)
        desc = mkd(patches)
        for mirrored in (patches.flip(-1), patches.flip(-2)):
            out = mkd(mirrored)
            self.assert_close(out.abs(), desc.abs())
            assert (out - desc).abs().max() > 0.02  # a mirrored random patch changes signs: the pin is not vacuous

    @pytest.mark.parametrize("kernel_type", ["concat", "polar", "cart"])
    def test_convention_mkd_flat_patch_is_unit_norm_except_in_float16_without_whitening(
        self, device, dtype, kernel_type
    ):
        # A flat patch has zero gradients, so its magnitudes are the sqrt of the 1e-8 added to them. In float16 the
        # 1e-8 underflows to 0 and, with whitening=None, the descriptor is all zero; in the other dtypes it is a unit
        # vector. whitening=None needs no download.
        if not supports_replicate_padding(device, dtype):
            pytest.skip(f"no replicate-padding kernel for {dtype} on {device.type}")
        mkd = MKDDescriptor(patch_size=19, kernel_type=kernel_type, whitening=None).to(device, dtype)
        out = mkd(torch.full((2, 1, 19, 19), 0.5, device=device, dtype=dtype))
        expected = 0.0 if dtype == torch.float16 else 1.0
        self.assert_close(out.norm(dim=1), torch.full((2,), expected, device=device, dtype=dtype))
