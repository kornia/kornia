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

from typing import List, Optional, Tuple

import torch
from torch import nn

from kornia.core.utils import _autocast_disabled, _torch_svd_cast

__all__ = ["ZCAWhitening", "linear_transform", "zca_mean", "zca_whiten"]


class ZCAWhitening(nn.Module):
    r"""Whiten data with a ZCA transform fitted on a data set.

    ``zca = ZCAWhitening(dim)`` is fitted with ``zca.fit(x)``, after which ``zca(x)`` returns ``x`` centered and
    whitened. ``dim`` is the sample axis and all other axes are flattened into
    one feature vector, so a ``(5, 3, 2, 2)`` tensor has :math:`N = 5` samples of :math:`D = 12` features.

    Convention:
        - The mean :math:`\mathbf{\mu}` and covariance :math:`\mathbf{\Sigma}` come from the flattened data
          :math:`\mathbf{X} \in \mathbb{R}^{N \times D}`. With :math:`U` the eigenvectors and :math:`S` the
          eigenvalues of :math:`\Sigma`, the transform is

          .. math::

              \mathbf{X}_{\text{zca}} = (\mathbf{X} - \mathbf{\mu})(U(S + \epsilon I)^{-1/2}U^T)^T

        - The output has the shape of the input. Call ``fit`` or pass ``include_fit=True`` before transforming.
        - Float16 and bfloat16 inputs are fitted and transformed in float32, and only the transformed output is rounded
          to the input dtype. The fitted buffers stay float32; casting the module to half precision discards that
          accuracy.
        - The fitted ``mean_vector``, ``transform_matrix`` and ``transform_inv`` are buffers: ``state_dict`` saves them
          and ``.to()`` moves and casts them. Loading a fitted state into an unfitted module fits it, and the loaded
          tensors keep the device and dtype they were saved with, so move the module after loading. A checkpoint saved
          before the fitted state was persisted holds none of it, and loading it keeps the current fit.

    Args:
        dim: sample axis.
        eps: value added to the eigenvalues before the inverse square root.
        unbiased: divide the covariance by :math:`N - 1` if ``True``, by :math:`N` otherwise.
        detach_transforms: detach the fitted tensors from the graph, so gradients do not flow through the fit.
        compute_inv: also compute the inverse transform that ``inverse_transform`` needs.

    Shape:
        - x: tensor ``(D_0, ..., D_dim, ..., D_N)`` whose axis ``dim`` indexes the samples.
        - x_whiten: same shape as ``x``.

    .. note::
       The fit takes an SVD of the covariance, whose backward step yields NaNs if the singular values are not unique,
       see :func:`torch.svd`. A working example is in the `ZCA whitening tutorial
       <https://colab.sandbox.google.com/github/kornia/tutorials/blob/master/source/zca_whitening.ipynb>`__.

    See Also:
        - :func:`zca_whiten`: fits and whitens in one call.
        - :func:`zca_mean`: computes the transform and mean this module stores.
        - :func:`linear_transform`: applies a given transform and mean.

    Examples:
        >>> x = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype = torch.float32)
        >>> zca = ZCAWhitening().fit(x)
        >>> x_whiten = zca(x)
        >>> zca = ZCAWhitening()
        >>> x_whiten = zca(x, include_fit = True) # Includes the fitting step
        >>> x_whiten = zca(x) # Can run now without the fitting set
        >>> # Enable backprop through ZCA fitting process
        >>> zca = ZCAWhitening(detach_transforms = False)
        >>> x_whiten = zca(x, include_fit = True) # Includes the fitting step

    References:
        [1] `Stanford PCA & ZCA whitening tutorial <http://ufldl.stanford.edu/tutorial/unsupervised/PCAWhitening/>`_

    """

    # Version 2 persists the fitted tensors; checkpoints of earlier versions hold none of them.
    _version: int = 2

    def __init__(
        self,
        dim: int = 0,
        eps: float = 1e-6,
        unbiased: bool = True,
        detach_transforms: bool = True,
        compute_inv: bool = False,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.eps = eps
        self.unbiased = unbiased
        self.detach_transforms = detach_transforms
        self.compute_inv = compute_inv

        self.fitted = False

        self.register_buffer("mean_vector", None)
        self.register_buffer("transform_matrix", None)
        self.register_buffer("transform_inv", None)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        version = local_metadata.get("version")
        for name in ("mean_vector", "transform_matrix", "transform_inv"):
            key = prefix + name
            current = self._buffers[name]
            if key in state_dict:
                if current is None:
                    # An unfitted module has no tensor to copy into: take the saved one's shape, dtype and device.
                    self._buffers[name] = torch.empty_like(state_dict[key])
            elif current is not None and (version is None or version < 2):
                # An older checkpoint did not save the fit: keep the current one, so strict loading still works.
                state_dict[key] = current

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        self.fitted = self.transform_matrix is not None

    def fit(self, x: torch.Tensor) -> "ZCAWhitening":
        r"""Fit ZCA whitening matrices to the data.

        Args:
            x: Input data.

        Returns:
            Returns a fitted ZCAWhiten object instance.

        Raises:
            ValueError: If the sample dimension has fewer than two entries with ``unbiased=True``, or none with
                ``unbiased=False``.

        """
        T, mean, T_inv = zca_mean(x, self.dim, self.unbiased, self.eps, self.compute_inv)

        self.mean_vector = mean
        self.transform_matrix = T
        if T_inv is None:
            self.transform_inv = T.new_empty(0)
        else:
            self.transform_inv = T_inv

        if self.detach_transforms:
            self.mean_vector = self.mean_vector.detach()
            self.transform_matrix = self.transform_matrix.detach()
            self.transform_inv = self.transform_inv.detach()

        self.fitted = True

        return self

    def forward(self, x: torch.Tensor, include_fit: bool = False) -> torch.Tensor:
        r"""Apply the whitening transform to the data.

        Args:
            x: Input data.
            include_fit: Indicates whether to fit the data as part of the forward pass.

        Returns:
            The transformed data.

        """
        if include_fit:
            self.fit(x)

        if not self.fitted:
            raise RuntimeError("Needs to be fitted first before running. Please call fit or set include_fit to True.")

        return linear_transform(x, self.transform_matrix, self.mean_vector, self.dim)

    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        r"""Apply the inverse transform to the whitened data.

        Uses the same sample dimension specified by ``dim`` when fitting.

        Args:
            x: Whitened data.

        Returns:
            Original data.

        """
        if not self.fitted:
            raise RuntimeError("Needs to be fitted first before running. Please call fit or set include_fit to True.")

        if not self.compute_inv:
            raise RuntimeError("Did not compute inverse ZCA. Please set compute_inv to True")

        if self.transform_inv is None:
            raise TypeError("The transform inverse should be a torch.Tensor. Got None.")

        if x.dtype in (torch.float16, torch.bfloat16) and not torch.jit.is_scripting():
            # The transformed mean is part of the fit too: autocasting it loses accuracy before linear_transform.
            with _autocast_disabled(x.device.type):
                mean_inv = -self.mean_vector.float().mm(self.transform_matrix.float())
        else:
            mean_inv = -self.mean_vector.mm(self.transform_matrix)

        return linear_transform(x, self.transform_inv, mean_inv, self.dim)


def zca_mean(
    inp: torch.Tensor, dim: int = 0, unbiased: bool = True, eps: float = 1e-6, return_inverse: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fit the ZCA whitening matrix and mean vector of a data set.

    ``T, mean, T_inv = zca_mean(inp, dim)`` returns the ZCA matrix ``T`` with shape ``(D, D)``, the mean vector with
    shape ``(1, D)`` and, if ``return_inverse=True``, the inverse matrix ``T_inv`` (else ``None``). ``D`` is the product
    of the sizes of all axes except ``dim``, the sample axis. Apply the result with :func:`linear_transform`.

    Convention:
        - See :class:`ZCAWhitening` for the covariance and the transform.
        - The fit runs with autocast disabled (not under TorchScript). Float16 and bfloat16 inputs are fitted in float32
          and return float32 tensors, to avoid cancellation when applying a rank-deficient fit.

    Args:
        inp: data tensor with shape ``(D_0, ..., D_dim, ..., D_N)``.
        dim: sample axis.
        unbiased: divide the covariance by :math:`N - 1` if ``True``, by :math:`N` otherwise.
        eps: value added to the eigenvalues before the inverse square root.
        return_inverse: also return the inverse transform.

    Returns:
        Tuple ``(T, mean, T_inv)``.

    Raises:
        ValueError: If the sample dimension has fewer than two entries with ``unbiased=True``, or none with
            ``unbiased=False``.

    See Also:
        - :class:`ZCAWhitening`: stores the fit and applies it.
        - :func:`zca_whiten`: fits and whitens in one call.
        - :func:`linear_transform`: applies the result.

    Examples:
        >>> x = torch.tensor([[0,1],[1,0],[-1,0],[0,-1]], dtype = torch.float32)
        >>> transform_matrix, mean_vector,_ = zca_mean(x) # Returns transformation matrix and data mean
        >>> x = torch.rand(3,20,2,2)
        >>> transform_matrix, mean_vector, inv_transform = zca_mean(x, dim = 1, return_inverse = True)
        >>> # transform_matrix.size() equals (12,12) and the mean vector.size equal (1,12)

    """
    if not isinstance(inp, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(inp)}")

    if not isinstance(eps, float):
        raise TypeError(f"eps type is not a float. Got{type(eps)}")

    if not isinstance(unbiased, bool):
        raise TypeError(f"unbiased type is not bool. Got{type(unbiased)}")

    if not isinstance(dim, int):
        raise TypeError(f"Argument 'dim' must be of type int. Got {type(dim)}")

    if not isinstance(return_inverse, bool):
        raise TypeError(f"Argument return_inverse must be of type bool {type(return_inverse)}")

    inp_size = inp.size()

    if dim >= len(inp_size) or dim < -len(inp_size):
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-len(inp_size)},{len(inp_size) - 1}], but got {dim}"
        )

    if dim < 0:
        dim = len(inp_size) + dim

    # Plain Python index arithmetic: tensor round-trips here would read data under graph capture.
    new_order: List[int] = [dim]
    num_features: int = 1
    for i in range(len(inp_size)):
        if i != dim:
            new_order.append(i)
            num_features *= inp_size[i]

    inp_permute = inp.permute(new_order)

    N = inp_size[dim]

    # The covariance divides by N - 1 (unbiased) or by N (biased), so fewer samples leave it undefined.
    if unbiased and N < 2:
        raise ValueError(f"Unbiased covariance requires at least two samples, got {N}.")
    if N < 1:
        raise ValueError("Covariance requires at least one sample, got 0.")

    if torch.jit.is_scripting():
        # A scripted autocast block does not leave an enclosing autocast region, so none is entered here.
        T, mean, T_inv = _zca_mean(inp_permute, num_features, unbiased, eps, return_inverse)
    elif inp.device.type == "meta":
        # Meta tensors have no autocast backend and carry only shape information.
        T, mean, T_inv = _zca_mean(inp_permute, num_features, unbiased, eps, return_inverse)
    else:
        # Autocast would run the scatter in float16 again, for float32 inputs too, and overflow it.
        with torch.autocast(device_type=inp.device.type, enabled=False):
            T, mean, T_inv = _zca_mean(inp_permute, num_features, unbiased, eps, return_inverse)

    return T, mean, T_inv


def _zca_mean(
    inp_permute: torch.Tensor, num_features: int, unbiased: bool, eps: float, return_inverse: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fit ZCA in the working dtype after validating and moving the sample axis."""
    N = inp_permute.shape[0]
    half_input = inp_permute.dtype in (torch.float16, torch.bfloat16)
    if half_input:
        inp_permute = inp_permute.float()

    mean: torch.Tensor = torch.mean(inp_permute, dim=0, keepdim=True)

    mean = mean.reshape((1, num_features))

    inp_center_flat: torch.Tensor = inp_permute.reshape((N, num_features)) - mean

    divisor = float(N - 1 if unbiased else N)
    if half_input:
        # Bfloat16 shares float32's exponent range, so divide before the scatter can overflow.
        inp_center_flat = inp_center_flat / divisor**0.5
    cov = inp_center_flat.t().mm(inp_center_flat)

    if not half_input:
        cov = cov / divisor

    U, S, _ = _torch_svd_cast(cov)

    S = S.reshape(-1, 1)
    S_inv_root: torch.Tensor = torch.rsqrt(S + eps)
    T: torch.Tensor = (U).mm(S_inv_root * U.t())

    T_inv: Optional[torch.Tensor] = None
    if return_inverse:
        T_inv = (U).mm(torch.sqrt(S + eps) * U.t())

    return T, mean, T_inv


def zca_whiten(inp: torch.Tensor, dim: int = 0, unbiased: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """Whiten data with a ZCA transform fitted on the same data.

    ``out = zca_whiten(inp, dim)`` fits ZCA on ``inp`` and returns ``inp`` centered and whitened, with its shape. To
    reuse the fit on other data, use :class:`ZCAWhitening`.

    Convention:
        - See :class:`ZCAWhitening` for the sample axis and the covariance normalization.

    Args:
        inp: data tensor with shape ``(D_0, ..., D_dim, ..., D_N)``.
        dim: sample axis.
        unbiased: divide the covariance by :math:`N - 1` if ``True``, by :math:`N` otherwise.
        eps: value added to the eigenvalues before the inverse square root.

    Returns:
        Whitened tensor with the shape of ``inp``.

    Raises:
        ValueError: If the sample dimension has fewer than two entries with ``unbiased=True``, or none with
            ``unbiased=False``.

    See Also:
        - :class:`ZCAWhitening`: fits once and whitens other data, and inverts the transform.
        - :func:`zca_mean`: returns the transform and mean without applying them.

    Examples:
        >>> x = torch.tensor([[0,1],[1,0],[-1,0]], dtype = torch.float32)
        >>> zca_whiten(x).round(decimals=3)
        tensor([[ 0.0000,  1.1550],
                [ 1.0000, -0.5770],
                [-1.0000, -0.5770]])

    """
    if not isinstance(inp, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(inp)}")

    if not isinstance(eps, float):
        raise TypeError(f"eps type is not a float. Got{type(eps)}")

    if not isinstance(unbiased, bool):
        raise TypeError(f"unbiased type is not bool. Got{type(unbiased)}")

    if not isinstance(dim, int):
        raise TypeError(f"Argument 'dim' must be of type int. Got {type(dim)}")

    transform, mean, _ = zca_mean(inp, dim, unbiased, eps, False)

    return linear_transform(inp, transform, mean, dim)


def linear_transform(
    inp: torch.Tensor, transform_matrix: torch.Tensor, mean_vector: torch.Tensor, dim: int = 0
) -> torch.Tensor:
    """Center data and multiply it by a transform matrix.

    ``out = linear_transform(inp, transform_matrix, mean_vector, dim)`` flattens all axes of ``inp`` except the sample
    axis ``dim`` into one feature vector per sample, computes ``(X - mean_vector) @ transform_matrix`` and restores the
    shape of ``inp``.

    Convention:
        Half inputs are centered and multiplied in float32 with autocast disabled (outside TorchScript), then rounded to
        the input dtype. Keep the fitted transform and mean in float32 to preserve their accuracy.

    Args:
        inp: data tensor with shape ``(D_0, ..., D_dim, ..., D_N)``.
        transform_matrix: matrix with shape ``(D, D)``, ``D`` being the product of the sizes of all axes except ``dim``.
        mean_vector: mean with shape ``(1, D)``.
        dim: sample axis.

    Returns:
        Transformed tensor with the shape of ``inp``.

    See Also:
        - :func:`zca_mean`: computes the whitening matrix and mean to pass here.
        - :class:`ZCAWhitening`: stores them and applies this transform.

    Example:
        >>> # Example where dim = 3
        >>> inp = torch.ones((10,3,4,5))
        >>> transform_mat = torch.ones((10*3*4,10*3*4))
        >>> mean = 2*torch.ones((1,10*3*4))
        >>> out = linear_transform(inp, transform_mat, mean, 3)
        >>> print(out.shape, out.unique())  # a (10,3,4,5) tensor of -120s
        torch.Size([10, 3, 4, 5]) tensor([-120.])

        >>> # Example where dim = 0
        >>> inp = torch.ones((10,2))
        >>> transform_mat = torch.ones((2,2))
        >>> mean = torch.zeros((1,2))
        >>> out = linear_transform(inp, transform_mat, mean)
        >>> print(out.shape, out.unique()) # a (10,2) tensor of 2s
        torch.Size([10, 2]) tensor([2.])

    """
    inp_size = inp.size()

    if dim >= len(inp_size) or dim < -len(inp_size):
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-len(inp_size)},{len(inp_size) - 1}], but got {dim}"
        )

    if dim < 0:
        dim = len(inp_size) + dim

    # Plain Python index arithmetic: tensor round-trips here would read data under graph capture.
    new_order: List[int] = [dim]
    num_features: int = 1
    for i in range(len(inp_size)):
        if i != dim:
            new_order.append(i)
            num_features *= inp_size[i]
    inv_order: List[int] = [0 for _ in new_order]
    for i in range(len(new_order)):
        inv_order[new_order[i]] = i

    inp_permute = inp.permute(new_order)
    inp_flat = inp_permute.reshape((-1, num_features))

    output_dtype = inp.dtype
    if output_dtype in (torch.float16, torch.bfloat16):
        inp_center = inp_flat.float() - mean_vector.float()
        if torch.jit.is_scripting():
            inp_transformed = inp_center.mm(transform_matrix.float())
        else:
            with _autocast_disabled(inp.device.type):
                inp_transformed = inp_center.mm(transform_matrix.float())
        inp_transformed = inp_transformed.to(output_dtype)
    else:
        inp_center = inp_flat - mean_vector
        inp_transformed = inp_center.mm(transform_matrix)

    inp_transformed = inp_transformed.reshape(inp_permute.size())

    return inp_transformed.permute(inv_order)
