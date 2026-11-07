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

from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, TypeVar, Union

import torch
from torch import nn
from torch.distributions import Distribution, Uniform

from kornia.augmentation.utils.helpers import MultiprocessWrapper, _holds_graph_tensor
from kornia.augmentation.utils.param_validation import _get_rebuild_device, _set_rebuild_device

T = TypeVar("T")

_UNSET = object()


def _rebuild_samplers_before_draw(generator: "RandomGeneratorBase", args: Tuple[Any, ...]) -> None:
    # A forward pre-hook: see ``RandomGeneratorBase._follow_learnable_ranges``.
    if generator._ranges_need_rebuild():
        registered = set(generator._parameters)
        generator._rebuild_samplers(generator.device, generator.dtype)
        # A sampler attribute assigned the range itself registers as a parameter, as it does in a build outside the
        # forward path. If the dtypes of the range and the generator only just came to match, a forward must not add
        # a key to ``state_dict``.
        for name in set(generator._parameters) - registered:
            generator.__dict__[name] = generator._parameters.pop(name)


class _PostInitInjectionMetaClass(type):
    """To inject the ``__post_init__`` function after the creation of each instance."""

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class RandomGeneratorBase(nn.Module, metaclass=_PostInitInjectionMetaClass):
    """Base class for generating random augmentation parameters."""

    device: Union[str, torch.device, None] = None
    dtype: torch.dtype

    # Whether ``make_samplers`` can run on every forward without reading the values of a range, once the validators
    # are off. A generator that decides anything from them turns this off, and a learnable range then reaches its draws
    # through ``load_state_dict`` and copies only.
    _follows_ranges_in_forward: bool = True

    def __init__(self) -> None:
        super().__init__()

    def __post_init__(self) -> None:
        self.set_rng_device_and_dtype()

    def set_rng_device_and_dtype(
        self,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Change the random generation device and dtype.

        Note:
            The generated random numbers are not reproducible across different devices and dtypes.

        """
        if device is None:
            device = torch.device("cpu")
        if self.device != device or self.dtype != dtype:
            before = dict(self.__dict__)
            if self._follows_ranges():
                self._rebuild_samplers(device, dtype)
            else:
                self.make_samplers(device, dtype)
            # What ``make_samplers`` assigns is derived from the ranges, device and dtype; ``__getstate__`` reads
            # the names to know what it may leave out of a copy and have ``__setstate__`` rebuild.
            made = {name for name, value in self.__dict__.items() if before.get(name, _UNSET) is not value}
            self._sampler_attrs = tuple(sorted(made.union(self.__dict__.get("_sampler_attrs", ()))))
            self.device = device
            self.dtype = dtype
            self._follow_learnable_ranges()

    def _rebuild_samplers(self, device: Union[str, torch.device, None], dtype: torch.dtype) -> None:
        # ``make_samplers`` from ranges that were validated when they were first built. A learnable range may have left
        # its domain since then, which raises nowhere after construction: it has to keep training, moving and copying.
        outer = _get_rebuild_device()
        _set_rebuild_device(torch.device(device))
        try:
            self.make_samplers(device, dtype)
        finally:
            _set_rebuild_device(outer)

    def _follows_ranges(self) -> bool:
        return _rebuild_samplers_before_draw in self._forward_pre_hooks.values()

    def _follow_learnable_ranges(self) -> None:
        # The samplers keep what ``make_samplers`` derived from a range. For a learnable range that is stale after an
        # in-place update and, when the derivation saved tensors, freed by the first backward, so they are built again
        # before each draw. It is a hook so that a generator without a learnable range pays nothing in ``forward``.
        if not self._follows_ranges_in_forward or self._follows_ranges():
            return
        sampler_attrs = self.__dict__.get("_sampler_attrs", ())
        if self._parameters or any(_holds_graph_tensor(self.__dict__.get(name)) for name in sampler_attrs):
            self.register_forward_pre_hook(_rebuild_samplers_before_draw)

    def _assign_derived_range(self, name: str, value: torch.Tensor) -> None:
        # ``make_samplers`` keeps a range moved to its device and dtype. That is the parameter itself when nothing had
        # to move, which registers it under ``name`` too, and ``state_dict`` has held that key ever since. A tensor
        # (the output of a parametrization, say) cannot be assigned to a registered name, so it replaces the name.
        if not isinstance(value, nn.Parameter):
            self._parameters.pop(name, None)
        setattr(self, name, value)

    def _ranges_need_rebuild(self) -> bool:
        # A generator left on the CPU with a range on an accelerator (or the reverse) would wait for the GPU to copy
        # that range before each draw. It keeps drawing where it is told to, and copies the range only after an in-place
        # update or a replacement of the parameter, which its identity and version counter reveal without reading it.
        device_type = torch.device(self.device).type
        ranges = [value for value in self._parameters.values() if value is not None]
        if all(value.device.type == device_type for value in ranges):
            return True
        versions = tuple((id(value), value._version) for value in ranges)
        if versions == self.__dict__.get("_range_versions"):
            return False
        self._range_versions = versions
        return True

    def _load_from_state_dict(self, state_dict: Dict[str, Any], prefix: str, *args: Any, **kwargs: Any) -> None:
        # A parameter counts as loaded even if it holds the saved value, because the module that shares it with this
        # generator (RandomErasing, say) loads it first.
        changed = {name for name in self._parameters if prefix + name in state_dict}
        for name, value in self._buffers.items():
            saved = state_dict.get(prefix + name)
            if (
                isinstance(value, torch.Tensor)
                and isinstance(saved, torch.Tensor)
                and name not in self._non_persistent_buffers_set
                and saved.shape == value.shape
                and not torch.equal(saved.to(device=value.device, dtype=value.dtype), value)
            ):
                changed.add(name)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if changed:
            self._ranges_loaded(changed)

    def _ranges_loaded(self, names: Set[str]) -> None:
        # A checkpoint that holds the ranges this generator already has changes nothing, so it keeps drawing as it did.
        self.__dict__.pop("_range_versions", None)
        if not self._follows_ranges():
            with torch.enable_grad():
                self.make_samplers(self.device, self.dtype)

    def __getstate__(self) -> Dict[str, Any]:
        # A range that requires grad, such as a learnable ``nn.Parameter``, makes the samplers autograd views of it.
        # ``copy.deepcopy`` refuses those, and pickle turns them into leaves that the copy's own range never reaches.
        # Leave them out: ``__setstate__`` rebuilds them from the copy's range, so gradients from the copy reach the
        # copy's parameter. Samplers without autograd history are copied as they are.
        state = super().__getstate__()
        dropped = [name for name in state.get("_sampler_attrs", ()) if _holds_graph_tensor(state.get(name))]
        for name in dropped:
            del state[name]
        if dropped:
            # ``torch.load(map_location=...)`` moves the tensors of the state but not ``self.device``. An empty tensor
            # on that device moves with them and tells ``__setstate__`` where the samplers would have gone.
            state["_sampler_device_probe"] = torch.empty(0, device=self.device)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        probe = state.pop("_sampler_device_probe", None)
        super().__setstate__(state)
        if any(name not in self.__dict__ for name in self.__dict__.get("_sampler_attrs", ())):
            if probe is not None:
                self.device = probe.device
            # Only samplers with autograd history were left out, so they are rebuilt with it, whatever the grad mode
            # of the copy: under ``torch.no_grad()`` the copy's range would otherwise never receive a gradient.
            with torch.enable_grad():
                self._rebuild_samplers(self.device, self.dtype)
        # A generator pickled before samplers followed their ranges has no hook yet.
        self._follow_learnable_ranges()

    def to(self, *args: Any, **kwargs: Any) -> "RandomGeneratorBase":
        """Update sampler device and dtype using ``torch.nn.Module.to`` semantics.

        Args:
            *args: Positional arguments accepted by ``Module.to``.
            **kwargs: Keyword arguments accepted by ``Module.to``.

        Returns:
            This generator instance.
        """
        return super().to(*args, **kwargs)

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], *args: Any, **kwargs: Any) -> "RandomGeneratorBase":
        # Follow registered tensors for direct, convenience, and parent-module moves.
        # Plain tensor constructor attributes keep their existing placement precedence.
        out = super()._apply(fn, *args, **kwargs)
        probe = fn(torch.zeros((), device=self.device, dtype=self.dtype))
        dtype = probe.dtype if probe.is_floating_point() else self.dtype
        self.set_rng_device_and_dtype(probe.device, dtype)
        return out

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        """Create distribution samplers for the given device and dtype.

        Assign the samplers to attributes of the generator, new or replaced, rather than writing them into an object
        it already holds: a copy of the generator rebuilds the attributes this call assigns, and only those.

        A generator that holds a learnable range goes through this again before each draw, with the checks in
        :mod:`kornia.augmentation.utils.param_validation` off. If it reads the values of a range here anyway, set
        ``_follows_ranges_in_forward = False``: reading them would wait for the GPU and break a ``torch.compile`` graph.

        Args:
            device: Target device.
            dtype: Target floating-point dtype.

        Raises:
            NotImplementedError: Subclass did not implement sampler creation.
        """
        raise NotImplementedError

    def forward(self, batch_shape: Tuple[int, ...], same_on_batch: bool = False) -> Dict[str, torch.Tensor]:
        """Sample random augmentation parameters.

        Args:
            batch_shape: Target batch shape.
            same_on_batch: If ``True``, use one sample and broadcast it across the
                batch dimension.

        Returns:
            Dictionary of tensors consumed by augmentation modules.

        Raises:
            NotImplementedError: Subclass did not implement parameter sampling.
        """
        raise NotImplementedError


class DistributionWithMapper(Distribution):
    """Wraps a distribution with a value mapper function.

    This is used to restrict the output values of a given distribution by a value mapper function.
    The value mapper function can be functions like sigmoid, tanh, etc.

    Args:
        dist: the target distribution.
        map_fn: the callable function to adjust the output from distributions.

    Example:
        >>> from torch.distributions import Normal
        >>> import torch.nn as nn
        >>> # without mapper
        >>> dist = DistributionWithMapper(Normal(0., 1.,), map_fn=None)
        >>> _ = torch.manual_seed(0)
        >>> dist.rsample((8,))
        tensor([ 1.5410, -0.2934, -2.1788,  0.5684, -1.0845, -1.3986,  0.4033,  0.8380])
        >>> # with sigmoid mapper
        >>> dist = DistributionWithMapper(Normal(0., 1.,), map_fn=nn.Sigmoid())
        >>> _ = torch.manual_seed(0)
        >>> dist.rsample((8,))
        tensor([0.8236, 0.4272, 0.1017, 0.6384, 0.2527, 0.1980, 0.5995, 0.6980])

    """

    def __init__(self, dist: Distribution, map_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> None:
        self.dist = dist
        self.map_fn = map_fn

    def rsample(self, sample_shape: Tuple[int, ...]) -> torch.Tensor:
        """Draw a reparameterized sample and apply ``map_fn`` when provided.

        Args:
            sample_shape: Desired sample shape.

        Returns:
            Sample tensor after optional mapping.
        """
        out = self.dist.rsample(torch.Size(sample_shape))
        if self.map_fn is not None:
            out = self.map_fn(out)
        return out

    def sample(self, sample_shape: Tuple[int, ...]) -> torch.Tensor:
        """Draw a sample and apply ``map_fn`` when provided.

        Args:
            sample_shape: Desired sample shape.

        Returns:
            Sample tensor after optional mapping.
        """
        out = self.dist.sample(torch.Size(sample_shape))
        if self.map_fn is not None:
            out = self.map_fn(out)
        return out

    def sample_n(self, n: int) -> torch.Tensor:
        """Draw ``n`` samples and apply ``map_fn`` when provided.

        Args:
            n: Number of samples.

        Returns:
            Sample tensor after optional mapping.
        """
        out = self.dist.sample_n(n)
        if self.map_fn is not None:
            out = self.map_fn(out)
        return out

    def __getattr__(self, attr: str) -> Any:
        # Special methods stay on the wrapper: delegating a probe such as ``__deepcopy__`` would copy the wrapped
        # distribution and drop ``map_fn``.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return getattr(object.__getattribute__(self, "dist"), attr)


class UniformDistribution(MultiprocessWrapper, Uniform):
    """Wrapper around torch Uniform distribution which makes it work with the 'spawn' multiprocessing context."""
