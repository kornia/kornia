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

import pytest
import torch

import kornia

from testing.base import BaseTester


class TestZCA(BaseTester):
    @pytest.mark.parametrize("dim", [0, 1, -1])
    @pytest.mark.parametrize("sample_count", [0, 1])
    def test_unbiased_sample_count(self, device, dtype, dim, sample_count):
        """Unbiased covariance is undefined with fewer than two samples (gh-5313)."""
        shape = (sample_count, 3) if dim == 0 else (3, sample_count)
        data = torch.ones(shape, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="at least two samples"):
            kornia.enhance.zca_mean(data, dim=dim, unbiased=True)

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_biased_single_sample(self, device, dtype, dim):
        """The population covariance of one sample is zero and remains supported."""
        data = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)
        if dim != 0:
            data = data.t()
        actual = kornia.enhance.zca_whiten(data, dim=dim, unbiased=False)
        self.assert_close(actual, torch.zeros_like(data))

    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_biased_empty_sample_axis(self, device, dtype, dim):
        """The biased covariance divides by N, so an empty sample axis is rejected as well."""
        shape = (0, 3) if dim == 0 else (3, 0)
        data = torch.ones(shape, device=device, dtype=dtype)
        with pytest.raises(ValueError, match="at least one sample"):
            kornia.enhance.zca_mean(data, dim=dim, unbiased=False)

    @pytest.mark.parametrize("unbiased", [True, False])
    @pytest.mark.parametrize("dim", [0, 1, -1])
    def test_two_samples(self, device, dtype, dim, unbiased):
        """Two samples are the fewest that unbiased whitening accepts; the divisor is N - 1 = 1, or N = 2 if biased."""
        data = torch.tensor([[2.0, 1.0], [0.0, 3.0]], device=device, dtype=dtype)
        # The centred samples (1, -1) and (-1, 1) have scatter eigenvalue 4 along (1, -1) and 0 across it, so with
        # eps = 1 the output is the centred data divided by sqrt(4 / divisor + 1).
        expected = (
            torch.tensor([[1.0, -1.0], [-1.0, 1.0]], device=device, dtype=dtype) / (5.0 if unbiased else 3.0) ** 0.5
        )
        if dim != 0:
            data, expected = data.t(), expected.t()
        actual = kornia.enhance.ZCAWhitening(dim=dim, unbiased=unbiased, eps=1.0)(data, include_fit=True)
        self.assert_close(actual, expected)

    @pytest.mark.parametrize("shape", [(3, 5, 2), (3, 3, 2)])
    @pytest.mark.parametrize("dim", [0, 1, -2, -1])
    def test_inverse_sample_axis(self, device, dtype, shape, dim):
        """Inverse uses the fitted sample axis, including equal-sized axis aliases (gh-5311)."""
        data = torch.arange(30 if shape[1] == 5 else 18, device=device, dtype=dtype).reshape(shape) / 100
        zca = kornia.enhance.ZCAWhitening(dim=dim, compute_inv=True, eps=1.0).fit(data)
        # A held-out query also exercises the fitted mean, rather than refitting.
        query = (10 * data).cos() / 4
        actual = zca.inverse_transform(zca(query))
        assert actual.shape == query.shape
        self.assert_close(actual, query)

    @pytest.mark.parametrize("unbiased", [True, False])
    def test_zca_unbiased(self, unbiased, device, dtype):
        data = torch.tensor([[0, 1], [1, 0], [-1, 0], [0, -1]], device=device, dtype=dtype)

        unbiased_val = 1.5 if unbiased else 2.0

        expected = torch.sqrt(unbiased_val * torch.abs(data)) * torch.sign(data)

        zca = kornia.enhance.ZCAWhitening(unbiased=unbiased).fit(data)

        actual = zca(data)

        self.assert_close(actual, expected, low_tolerance=True)

    @pytest.mark.parametrize("dim", [0, 1])
    def test_dim_args(self, dim, device, dtype):
        if "xla" in device.type:
            pytest.skip("buggy with XLA devices.")

        if dtype == torch.float16:
            pytest.skip("not work for half-precision")

        data = torch.tensor([[0, 1], [1, 0], [-1, 0], [0, -1]], device=device, dtype=dtype)

        if dim == 1:
            expected = torch.tensor(
                [
                    [-0.35360718, 0.35360718],
                    [0.35351562, -0.35351562],
                    [-0.35353088, 0.35353088],
                    [0.35353088, -0.35353088],
                ],
                device=device,
                dtype=dtype,
            )
        elif dim == 0:
            expected = torch.tensor(
                [[0.0, 1.2247448], [1.2247448, 0.0], [-1.2247448, 0.0], [0.0, -1.2247448]], device=device, dtype=dtype
            )

        zca = kornia.enhance.ZCAWhitening(dim=dim)
        actual = zca(data, True)

        self.assert_close(actual, expected, low_tolerance=True)

    @pytest.mark.parametrize("input_shape,eps", [((15, 2, 2, 2), 1e-6), ((10, 4), 0.1), ((20, 3, 2, 2), 1e-3)])
    def test_identity(self, input_shape, eps, device, dtype):
        """Assert that data can be recovered by the inverse transform."""
        data = torch.randn(*input_shape, device=device, dtype=dtype)

        zca = kornia.enhance.ZCAWhitening(compute_inv=True, eps=eps).fit(data)

        data_w = zca(data)

        data_hat = zca.inverse_transform(data_w)

        self.assert_close(data, data_hat, low_tolerance=True)

    def test_grad_zca_individual_transforms(self, device):
        """Check if the gradients of the transforms are correct w.r.t to the input data."""
        if device.type == "mps":
            pytest.skip("MPS does not support float64 required for gradcheck")
        data = torch.tensor([[2, 0], [0, 1], [-2, 0], [0, -1]], device=device, dtype=torch.float64)

        def zca_T(x):
            return kornia.enhance.zca_mean(x)[0]

        def zca_mu(x):
            return kornia.enhance.zca_mean(x)[1]

        def zca_T_inv(x):
            return kornia.enhance.zca_mean(x, return_inverse=True)[2]

        self.gradcheck(zca_T, (data,))
        self.gradcheck(zca_mu, (data,))
        self.gradcheck(zca_T_inv, (data,))

    def test_grad_zca_with_fit(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 required for gradcheck")
        data = torch.tensor([[2, 0], [0, 1], [-2, 0], [0, -1]], device=device, dtype=torch.float64)

        def zca_fit(x):
            zca = kornia.enhance.ZCAWhitening(detach_transforms=False)
            return zca(x, include_fit=True)

        self.gradcheck(zca_fit, (data,))

    def test_grad_detach_zca(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64 required for gradcheck")
        data = torch.tensor([[1, 0], [0, 1], [-2, 0], [0, -1]], device=device, dtype=torch.float64)

        zca = kornia.enhance.ZCAWhitening()

        zca.fit(data)

        self.gradcheck(zca, (data,))

    def test_fitted_state_is_serialized(self, device, dtype):
        data = torch.randn(8, 3, device=device, dtype=dtype)
        zca = kornia.enhance.ZCAWhitening().fit(data)

        state_dict = zca.state_dict()

        assert "mean_vector" in state_dict
        assert "transform_matrix" in state_dict
        assert "transform_inv" in state_dict
        assert state_dict["mean_vector"].shape == zca.mean_vector.shape
        assert state_dict["transform_matrix"].shape == zca.transform_matrix.shape
        # Without compute_inv the inverse is an empty placeholder, made on the data's device and dtype.
        assert state_dict["transform_inv"].shape == (0,)
        assert all(tensor.device == device and tensor.dtype == dtype for tensor in state_dict.values())

    def test_unfitted_state_is_empty(self):
        zca = kornia.enhance.ZCAWhitening()

        assert zca.state_dict() == {}

    def test_fitted_state_follows_dtype_conversion(self, device):
        if device.type == "mps":
            pytest.skip("MPS does not support float64")
        data = torch.randn(8, 3, device=device, dtype=torch.float32)
        zca = kornia.enhance.ZCAWhitening().fit(data)

        expected = zca(data)
        zca.double()

        actual = zca(data.double())

        assert zca.mean_vector.dtype == torch.float64
        assert zca.transform_matrix.dtype == torch.float64
        assert zca.transform_inv.dtype == torch.float64
        self.assert_close(actual, expected.double(), low_tolerance=True)

    def test_fitted_state_follows_device_conversion(self, device, dtype):
        data = torch.randn(8, 3, device=device, dtype=dtype)
        zca = kornia.enhance.ZCAWhitening(compute_inv=True).fit(data)

        # The meta device needs no hardware, so every test leg moves the fit off the device it was made on.
        zca.to("meta")

        assert zca.mean_vector.device.type == "meta"
        assert zca.transform_matrix.device.type == "meta"
        assert zca.transform_inv.device.type == "meta"

        output = zca(data.to("meta"))

        assert output.device.type == "meta"

    def test_fitted_state_round_trip(self, device, dtype):
        data = torch.randn(8, 3, device=device, dtype=dtype)

        zca = kornia.enhance.ZCAWhitening(compute_inv=True).fit(data)
        expected = zca(data)

        loaded = kornia.enhance.ZCAWhitening(compute_inv=True)
        result = loaded.load_state_dict(zca.state_dict())

        assert result.missing_keys == []
        assert result.unexpected_keys == []
        assert loaded.fitted
        self.assert_close(loaded(data), expected, low_tolerance=True)
        self.assert_close(loaded.inverse_transform(expected), zca.inverse_transform(expected), low_tolerance=True)

    def test_checkpoint_without_fitted_state_keeps_the_fit(self, device, dtype):
        data = torch.randn(8, 3, device=device, dtype=dtype)
        model = torch.nn.Sequential(kornia.enhance.ZCAWhitening().fit(data))
        expected = model(data)

        # A checkpoint saved before the fitted state was persisted: no ZCA keys and module version 1.
        checkpoint = torch.nn.Sequential(kornia.enhance.ZCAWhitening()).state_dict()
        checkpoint._metadata["0"]["version"] = 1
        result = model.load_state_dict(checkpoint)

        assert result.missing_keys == []
        assert model[0].fitted
        self.assert_close(model(data), expected)

        # A plain dict, e.g. one rebuilt with renamed keys, carries no version and is read the same way.
        assert model.load_state_dict({}).missing_keys == []
        self.assert_close(model(data), expected)

        # An unfitted module stays unfitted.
        unfitted = torch.nn.Sequential(kornia.enhance.ZCAWhitening())
        result = unfitted.load_state_dict(checkpoint)
        assert result.missing_keys == []
        assert result.unexpected_keys == []
        assert not unfitted[0].fitted

        # A checkpoint of this version saved from an unfitted module lacks the keys, and strict loading says so.
        with pytest.raises(RuntimeError, match="Missing key"):
            model.load_state_dict(torch.nn.Sequential(kornia.enhance.ZCAWhitening()).state_dict())

    def test_unfitted_state_round_trip(self):
        zca = kornia.enhance.ZCAWhitening()
        loaded = kornia.enhance.ZCAWhitening()

        result = loaded.load_state_dict(zca.state_dict())

        assert result.missing_keys == []
        assert result.unexpected_keys == []
        assert not loaded.fitted

    def test_not_fitted(self, device, dtype):
        data = torch.rand(10, 2, device=device, dtype=dtype)
        zca = kornia.enhance.ZCAWhitening()
        with pytest.raises(RuntimeError):
            zca(data)

    def test_not_fitted_inv(self, device, dtype):
        data = torch.rand(10, 2, device=device, dtype=dtype)
        zca = kornia.enhance.ZCAWhitening()
        with pytest.raises(RuntimeError):
            zca.inverse_transform(data)

    def test_jit(self, device, dtype):
        data = torch.rand(10, 3, 1, 2, device=device, dtype=dtype)
        zca = kornia.enhance.ZCAWhitening().fit(data)
        zca_jit = kornia.enhance.ZCAWhitening().fit(data)
        zca_jit = torch.jit.script(zca_jit)
        self.assert_close(zca_jit(data), zca(data))

    @pytest.mark.parametrize("unbiased", [True, False])
    def test_zca_whiten_func_unbiased(self, unbiased, device, dtype):
        data = torch.tensor([[0, 1], [1, 0], [-1, 0], [0, -1]], device=device, dtype=dtype)

        unbiased_val = 1.5 if unbiased else 2.0

        expected = torch.sqrt(unbiased_val * torch.abs(data)) * torch.sign(data)

        actual = kornia.enhance.zca_whiten(data, unbiased=unbiased)

        self.assert_close(actual, expected, low_tolerance=True)
