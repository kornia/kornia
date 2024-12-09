# type: ignore
# pytorch tensor wrapper class
# insipired by:
# https://github.com/pytorch/pytorch/blob/591dfffa38848de54b7f5f4e49260847024c9281/test/test_overrides.py#L748
import collections

import torch
from torch import Tensor


# TODO: promote to KORNIA_WRAP
def wrap(v, cls):
    # wrap inputs if necessary
    if type(v) in {tuple, list}:
        return type(v)(wrap(vi, cls) for vi in v)

    return cls(v) if isinstance(v, Tensor) else v


# TODO: promote to KORNIA_UNWRAP
def unwrap(v):
    if type(v) in {tuple, list}:
        return type(v)(unwrap(vi) for vi in v)

    return v._data if isinstance(v, TensorWrapper) else v


class TensorWrapper:
    def __init__(self, data: Tensor) -> None:
        self.__dict__["_data"] = data
        self.__dict__["used_attrs"] = set()
        self.__dict__["used_calls"] = set()

    def unwrap(self):
        return unwrap(self)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"{self._data}"

    def __getattr__(self, name):
        if name == "data":
            return self._data
        elif name in self.__dict__:
            return self.__dict__[name]
        self.used_attrs.add(name)

        val = getattr(self._data, name)

        # NOTE: not clear is really needed
        # If it's a method
        # if callable(val):
        #    c = getattr(type(self._data), name)
        #    # Don't append self to args if classmethod/staticmethod
        #    if c is val:
        #        return lambda *a, **kw: wrap(self.__torch_function__(c, (type(self),), args=a, kwargs=kw), type(self))
        #    # Otherwise append self to args
        #    return lambda *a, **kw: wrap(
        #        #self.__torch_function__(c, (type(self),), args=(self,) + a, kwargs=kw), type(self)
        #    )

        return wrap(val, type(self))

    def __setattr__(self, name, value) -> None:
        if name in self.__dict__:
            self.__dict__[name] = value

        self.used_attrs.add(name)
        setattr(self._data, name, value)

    def __setitem__(self, key, value) -> None:
        self._data[key] = value

    def __getitem__(self, key):
        return wrap(self._data[key], type(self))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        # Find an instance of this class in the arguments
        args_of_this_cls = []
        for a in args:
            if isinstance(a, cls):
                args_of_this_cls.append(a)
            elif isinstance(a, collections.abc.Sequence):
                args_of_this_cls.extend(el for el in a if isinstance(el, cls))
        # assert len(args_of_this_cls) > 0
        for a in args_of_this_cls:
            a.used_calls.add(func)
        args = unwrap(tuple(args))
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}

        return wrap(func(*args, **kwargs), cls)

    # TODO: `def __add__(self, other) -> Self:` when mypy release >0.991
    def __add__(self, other):
        return self.__unary_op__(torch.add, other)

    def __radd__(self, other):
        return self.__unary_op__(torch.add, other)

    def __mul__(self, other):
        return self.__unary_op__(torch.mul, other)

    def __rmul__(self, other):
        return self.__unary_op__(torch.mul, other)

    def __sub__(self, other):
        return self.__unary_op__(torch.sub, other)

    def __rsub__(self, other):
        return self.__unary_op__(torch.sub, other)

    def __truediv__(self, other):
        return self.__unary_op__(torch.true_divide, other)

    def __floordiv__(self, other):
        return self.__unary_op__(torch.floor_divide, other)

    def __ge__(self, other):
        return self.__unary_op__(torch.ge, other)

    def __gt__(self, other):
        return self.__unary_op__(torch.gt, other)

    def __lt__(self, other):
        return self.__unary_op__(torch.lt, other)

    def __le__(self, other):
        return self.__unary_op__(torch.le, other)

    def __eq__(self, other):
        return self.__unary_op__(torch.eq, other)

    def __ne__(self, other):
        return self.__unary_op__(torch.ne, other)

    def __bool__(self) -> bool:
        return self.__unary_op__(Tensor.__bool__)

    def __int__(self) -> int:
        return self.__unary_op__(Tensor.__int__)

    def __neg__(self):
        return self.__unary_op__(Tensor.negative)

    def __unary_op__(self, func, other=None):
        args = (self, other) if other is not None else (self,)
        return self.__torch_function__(func, (type(self),), args)

    def __len__(self) -> int:
        return len(self._data)
