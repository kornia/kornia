kornia's validation errors now also derive from the matching built-in exception: `ShapeError`,
`ValueCheckError`, `DeviceError` and `ImageError` are `ValueError`s, and `TypeCheckError` is a
`TypeError`. Previously all five derived only from `BaseError` (an `Exception`), so an
`except ValueError:` or `except TypeError:` around a kornia call caught the functions that raise the
built-ins themselves but let a failed `KORNIA_CHECK_*` escape, for example `gaussian_blur2d` on a
2-D tensor. Such clauses now catch these errors as well; `except BaseError` and the specific classes
keep working. A failed plain `KORNIA_CHECK` still raises the bare `BaseError`, which is neither
built-in. Separately, a call whose tensors share a device but differ in dtype, such as
`find_homography_dlt` with a `float32` and a `float64` point set, `get_motion_kernel2d`,
`geometry.transform.Affine` or an augmentation's parameter generator given mixed dtypes, now raises
`TypeCheckError` ("expected torch.float32, got torch.float64") instead of a `DeviceError` that
listed the same device twice; `except DeviceError` no longer catches it. A device mismatch still
raises `DeviceError`.
