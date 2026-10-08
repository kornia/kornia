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

"""nn.Module containing the functionalities for computing the real roots of polynomial equation."""

import math
from typing import NamedTuple, Tuple

import torch

from kornia.core.check import KORNIA_CHECK_SHAPE


# Reference : https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/polynom_solver.cpp
def solve_quadratic(coeffs: torch.Tensor) -> torch.Tensor:
    r"""Solve given quadratic equation.

    The function takes the coefficients of quadratic equation and returns the real roots.

    .. math:: coeffs[0]x^2 + coeffs[1]x + coeffs[2] = 0

    Convention:
        - The coefficients of :func:`solve_quadratic`, :func:`solve_cubic` and :func:`solve_quartic` are batched
          ``(B, k + 1)`` for degree ``k``, highest degree first; :ref:`two-view-conventions` compares this with
          ``numpy.roots``.
        - Only real roots are returned, and a repeated root is repeated. A missing real root is reported as
          ``0.0``, which is indistinguishable from a root at 0, so count real roots from the discriminant when
          it matters.
        - For ``coeffs = [a, b, c]`` and ``D = b**2 - 4 * a * c``, ``solve_quadratic`` returns
          ``[(-b + sqrt(D)) / (2 * a), (-b - sqrt(D)) / (2 * a)]``, so the order flips with the sign of ``a``.
        - A zero leading coefficient lowers the degree: with ``a = 0`` the root ``-c / b`` of the linear equation
          is in slot 0, as :func:`solve_cubic` and :func:`solve_quartic` do for their lower-degree rows.
        - Half inputs are evaluated in float32. On MPS, subnormal float32 coefficients use CPU float64;
          captured MPS graphs use CPU float64 throughout. The output retains the input dtype and device.

    Args:
        coeffs : The coefficients of quadratic equation :`(B, 3)`

    Returns:
        A torch.Tensor of shape `(B, 2)` containing the real roots to the quadratic equation.

    Example:
        >>> coeffs = torch.tensor([[1., 4., 4.]])
        >>> roots = solve_quadratic(coeffs)

    """
    KORNIA_CHECK_SHAPE(coeffs, ["B", "3"])
    return _solve_quadratic(coeffs)


def _solve_quadratic(coeffs: torch.Tensor) -> torch.Tensor:
    """Solve quadratics as :func:`solve_quadratic` does, for a ``(B, 3)`` input that is already checked."""
    # Forming b**2 or 4*a*c in a half dtype is needlessly fragile; solve in
    # float32 just as the cubic solver does, then preserve the public dtype.
    if coeffs.dtype in (torch.float16, torch.bfloat16):
        return _solve_quadratic(coeffs.float()).to(coeffs.dtype)

    # MPS flushes float32 subnormals in products even when the input tensor
    # retains them. CPU float64 preserves the public tiny-root convention.
    if coeffs.dtype == torch.float32 and coeffs.device.type == "mps":
        if torch.compiler.is_compiling():
            roots = _solve_quadratic(coeffs.cpu().double())
            return roots.to(dtype=coeffs.dtype).to(device=coeffs.device)
        bits = coeffs.view(torch.int32).bitwise_and(0x7FFFFFFF)
        subnormal = ((bits > 0) & (bits < 2**23)).any(-1)
        if bool(subnormal.any()):
            placeholder = torch.tensor([1.0, 0.0, -1.0], device=coeffs.device, dtype=coeffs.dtype)
            roots = _solve_quadratic(torch.where(subnormal[:, None], placeholder, coeffs))
            precise = _solve_quadratic(coeffs[subnormal].cpu().double())
            roots[subnormal] = precise.to(dtype=coeffs.dtype).to(device=coeffs.device)
            return roots

    a, b, c = _scaled_quadratic_coefficients(coeffs)

    # Calculate discriminant
    # Multiply a and c first: 4*a can overflow even when a*c is finite
    # (for example a=1e38, c=-1e-38 in float32).
    delta = b * b - 4 * (a * c)

    # Create masks for negative and zero discriminant
    mask_negative = delta < 0
    mask_nonpositive = delta <= 0

    # With a == 0 the equation is linear, bx + c = 0: its root goes to slot 0 and slot 1 is padded.
    # Dividing by a placeholder 1 there keeps the unused quadratic lanes (and their gradients) finite.
    one = torch.ones_like(a)
    mask_linear = a == 0
    mask_b_zero = b == 0

    # Branch-free selection so the function traces under graph capture. The square root is only taken
    # where delta > 0: a zero discriminant yields the double root -b/(2a) with sqrt_delta = 0, and a
    # negative one yields zeros; feeding those lanes a safe placeholder keeps their gradients finite.
    zero = torch.zeros_like(delta)
    sqrt_delta = torch.where(mask_nonpositive, zero, torch.sqrt(torch.where(mask_nonpositive, 1.0, delta)))

    # (-b +- sqrt(delta)) / (2a) subtracts nearly equal numbers for the root of smaller magnitude when
    # |4ac| << b^2 (#4914). q = -(b + sign(b) sqrt(delta)) / 2 adds numbers of the same sign; the roots are
    # q / a and c / q, which are (-b - sqrt(delta)) / (2a) and (-b + sqrt(delta)) / (2a) for b >= 0 and the
    # other way round for b < 0. c / q is only taken where delta > 0 and a != 0, where |q| >= sqrt(delta) / 2 > 0;
    # at a double root both slots are q / a = -b / (2a). Elsewhere a placeholder q keeps the discarded lane's
    # gradient finite: with no real root and a tiny b, c / q^2 overflows and torch.where would turn it into nan.
    b_nonnegative = b >= 0
    sign_b = torch.where(b_nonnegative, one, -one)
    q = -0.5 * (b + sign_b * sqrt_delta)
    mask_distinct = ~(mask_nonpositive | mask_linear)
    root_q_over_a = q / torch.where(mask_linear, one, a)
    root_c_over_q = torch.where(mask_distinct, c / torch.where(mask_distinct, q, one), root_q_over_a)
    root_plus = torch.where(b_nonnegative, root_c_over_q, root_q_over_a)
    root_minus = torch.where(b_nonnegative, root_q_over_a, root_c_over_q)

    # The a * x^2 / b term is 0 in the forward pass, but it keeps the root's dependence on a in the
    # gradient (d root / da = -root^2 / b). With b == 0 as well there is no root to report. The lane
    # takes c only from the linear rows: torch.where differentiates the lane it discards too, and for an
    # ordinary row with a tiny b, (c / b)^2 overflows there and turns the row's gradient into nan.
    safe_b = torch.where(mask_b_zero, one, b)
    root_linear = -torch.where(mask_linear, c, zero) / safe_b
    root_linear = torch.where(mask_b_zero, zero, root_linear - a * root_linear * root_linear / safe_b)

    root_0 = torch.where(mask_linear, root_linear, torch.where(mask_negative, zero, root_plus))
    root_1 = torch.where(mask_linear | mask_negative, zero, root_minus)
    return torch.stack([root_0, root_1], dim=-1)


# Bit layout of the float dtypes solve_cubic scales in: the integer dtype of the same width, the exponent bias and the
# number of mantissa bits. A float with a zero mantissa is 2 ** (stored exponent - bias).
_FLOAT_LAYOUT = {torch.float32: (torch.int32, 127, 23), torch.float64: (torch.int64, 1023, 52)}

# A cubic root that is this many times larger than the other two is taken as dominant by solve_cubic, which then gets
# the other two from Vieta's relations instead of the closed form. See the comment there.
_DOMINANT_ROOT_RATIO = 2.0**4


def _exact_power_of_two(exponent: torch.Tensor) -> torch.Tensor:
    """Return ``2 ** exponent`` for an integer-valued float tensor, exact on every backend.

    ``torch.exp2`` and ``torch.pow`` are not exact for integer arguments on every backend (MPS), and a scale that is
    not a power of two changes the bits of the scaled row. The exponent is clamped so that both ``2 ** exponent`` and
    ``2 ** -exponent`` are normal floats, and is written into the exponent field of the float.
    """
    int_dtype, bias, mantissa_bits = _FLOAT_LAYOUT[exponent.dtype]
    biased = exponent.clamp(1 - bias, bias - 1).to(int_dtype) + bias
    return (biased * 2**mantissa_bits).view(exponent.dtype)


def _exact_powers_of_two(exponent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(2 ** exponent, 2 ** -exponent)`` as :func:`_exact_power_of_two` computes each of them."""
    int_dtype, bias, mantissa_bits = _FLOAT_LAYOUT[exponent.dtype]
    clamped = exponent.clamp(1 - bias, bias - 1).to(int_dtype)
    return (
        ((bias + clamped) * 2**mantissa_bits).view(exponent.dtype),
        ((bias - clamped) * 2**mantissa_bits).view(exponent.dtype),
    )


def _scaled_quadratic_coefficients(coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scale a quadratic exactly when its discriminant terms leave the dtype range."""
    raw_a, raw_b, raw_c = coeffs.unbind(dim=-1)
    finfo = torch.finfo(coeffs.dtype)
    if coeffs.device.type == "cpu" and coeffs.numel() > 0 and not torch.compiler.is_compiling():
        absolute = coeffs.detach().abs()
        # Every nonzero term is normal and every product safely representable.
        # Keep ordinary eager batches out of the conditioning machinery. Zeros
        # are neutral; a NaN propagates through aminmax and fails both tests.
        smallest, largest = torch.stack(torch.aminmax(torch.where(absolute == 0, 1.0, absolute))).tolist()
        if smallest >= math.sqrt(finfo.tiny) * 4 and largest <= math.sqrt(finfo.max) / 8:
            return raw_a, raw_b, raw_c
    magnitude = coeffs.abs().amax(dim=-1).detach()
    one = torch.ones_like(magnitude)
    finfo = torch.finfo(coeffs.dtype)
    max_value = torch.tensor(finfo.max, device=coeffs.device, dtype=coeffs.dtype)
    tiny_value = torch.tensor(finfo.tiny, device=coeffs.device, dtype=coeffs.dtype)
    b_squared = (raw_b * raw_b).detach()
    abs_ac = (raw_a * raw_c).detach().abs()
    # One discriminant term underflowing is harmless while the other is at least 16 * tiny: its absolute error
    # is below half the smallest subnormal. Rescaling such a row by its largest coefficient could instead flush a
    # small coefficient to zero, so only overflow or a discriminant that is tiny as a whole rescales.
    rescale = (
        ~torch.isfinite(b_squared)
        | ~torch.isfinite(abs_ac)
        | (b_squared > max_value / 8)
        | (abs_ac > max_value / 32)
        | ((b_squared + 4 * abs_ac < tiny_value * 16) & ((raw_b != 0) | ((raw_a != 0) & (raw_c != 0))))
    )
    exponent = torch.floor(torch.log2(torch.where(magnitude > 0, magnitude, one)))
    scale = torch.where(rescale, _exact_power_of_two(exponent), one)
    return (coeffs / scale[:, None]).unbind(dim=-1)


def _cubic_discriminant_is_uncertain(coeffs: torch.Tensor) -> torch.Tensor:
    """Return float32 cubic rows whose Cardano discriminant needs more precision."""
    a, b, c, d = coeffs.unbind(dim=-1)
    one = torch.ones_like(a)
    zero = torch.zeros_like(a)
    cubic = a != 0
    safe_a = torch.where(cubic, a, one)
    b_a = torch.where(cubic, b / safe_a, zero)
    c_a = torch.where(cubic, c / safe_a, zero)
    d_a = torch.where(cubic, d / safe_a, zero)
    q = (3 * c_a - b_a * b_a) / 9
    r = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a * b_a) / 54
    q3 = q * q * q
    discriminant = q3 + r * r
    error_bound = 32 * torch.finfo(coeffs.dtype).eps * (q3.abs() + r * r)
    return cubic & (
        (torch.isfinite(discriminant) & (discriminant.abs() <= error_bound)) | ~torch.isfinite(discriminant)
    )


def solve_cubic(coeffs: torch.Tensor) -> torch.Tensor:
    r"""Solve given cubic equation.

    The function takes the coefficients of cubic equation and returns
    the real roots.

    .. math:: coeffs[0]x^3 + coeffs[1]x^2 + coeffs[2]x + coeffs[3] = 0

    Convention:
        - Coefficient layout and zero padding as :func:`solve_quadratic`. Three real roots are returned unsorted,
          and a single real root is in slot 0.
        - A zero leading coefficient lowers the degree, and the roots of the remaining polynomial come first.
        - The closed form is evaluated on the row scaled by an exact power of two to a unit root bound, and the
          roots are scaled back, so its intermediates neither overflow nor underflow for a tiny leading coefficient
          or for roots far from unit scale (#4914).
        - Half inputs are evaluated in float32. Float32 cubics are solved in float64 on CPU and in captured graphs,
          so a row's roots do not depend on the rest of its batch. Eager accelerator paths promote only the rows
          whose float32 discriminant sign is uncertain, and MPS solves those rows on the CPU. The output retains
          the input dtype and device.

    Args:
        coeffs : The coefficients cubic equation : `(B, 4)`

    Returns:
        A torch.Tensor of shape `(B, 3)` containing the real roots to the cubic equation.

    Example:
        >>> solve_cubic(torch.tensor([[1., 0., 0., 1.]]))
        tensor([[-1.,  0.,  0.]])

    .. note::
       At the acos boundary reached by a repeated (or near-repeated) real root, backward suppresses
       the derivative of the acos argument to keep gradients finite. Repeated-root derivatives are
       undefined; this is a surrogate convention, not a mathematical Jacobian. :func:`solve_quartic`
       inherits this convention for the rows it solves as cubics.

    """
    return _solve_cubic_with_count(coeffs)[0]


def _solve_cubic_with_count(coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve a cubic as :func:`solve_cubic` does and also count its real roots."""
    KORNIA_CHECK_SHAPE(coeffs, ["B", "4"])
    return _solve_cubic(coeffs)


def _solve_cubic(coeffs: torch.Tensor, _allow_promotion: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve and count as :func:`_solve_cubic_with_count` does, for a ``(B, 4)`` input that is already checked."""
    # Cubic intermediates underflow in half precision. The dtype test is static
    # for a compiled graph, while the recursive call contains only tensor work.
    if coeffs.dtype in (torch.float16, torch.bfloat16):
        roots, num_real = _solve_cubic(coeffs.float(), _allow_promotion)
        return roots.to(coeffs.dtype), num_real

    # Float32 cannot reliably resolve the sign of Cardano's discriminant for
    # close large roots. CPU and compiled graphs solve every row in float64: on
    # CPU that costs less than locating the uncertain rows, and it keeps each
    # row independent of its batch. Eager accelerator execution promotes only
    # rows inside the floating-point error bound.
    # The seven-point kernel keeps its native precision in _solve_cubic_real.
    if coeffs.dtype == torch.float32 and _allow_promotion:
        # MPS has no float64 kernels; separate device and dtype copies avoid
        # its combined conversion losing values. Captured graphs keep fixed shape.
        if coeffs.device.type == "cpu" or torch.compiler.is_compiling():
            precise_coeffs = coeffs.cpu().double() if coeffs.device.type == "mps" else coeffs.double()
            roots, num_real = _solve_cubic(precise_coeffs, False)
            return roots.to(dtype=coeffs.dtype).to(device=coeffs.device), num_real.to(device=coeffs.device)
        with torch.no_grad():
            uncertain = _cubic_discriminant_is_uncertain(coeffs)
            if coeffs.device.type == "mps":
                bits = coeffs.view(torch.int32).bitwise_and(0x7FFFFFFF)
                uncertain = uncertain | ((bits > 0) & (bits < 2**23)).any(-1)
        if torch.any(uncertain):
            placeholder = torch.tensor([1.0, 0.0, 0.0, -1.0], device=coeffs.device, dtype=coeffs.dtype)
            native_coeffs = torch.where(uncertain[:, None], placeholder, coeffs)
            roots, num_real = _solve_cubic(native_coeffs, False)
            selected = coeffs[uncertain]
            precise_coeffs = selected.cpu().double() if coeffs.device.type == "mps" else selected.double()
            precise_roots, precise_count = _solve_cubic(precise_coeffs, False)
            roots = roots.clone()
            num_real = num_real.clone()
            roots[uncertain] = precise_roots.to(dtype=coeffs.dtype).to(device=coeffs.device)
            num_real[uncertain] = precise_count.to(device=coeffs.device)
            return roots, num_real

    a, b, c, d = coeffs.unbind(dim=-1)
    zero = torch.zeros_like(a)
    one = torch.ones_like(a)
    mask_a_zero = a == 0
    mask_b_zero = b == 0
    mask_cubic = ~mask_a_zero
    mask_second_order = mask_a_zero & ~mask_b_zero
    mask_first_order = mask_a_zero & mask_b_zero & (c != 0)

    # All candidates are evaluated at fixed batch shape. Feed unused lanes
    # benign values before nonlinear operations: torch.where selects values,
    # but autograd still visits the unselected expression's backward graph.
    safe_a = torch.where(mask_cubic, a, one)
    b_a = torch.where(mask_cubic, b / safe_a, zero)
    c_a = torch.where(mask_cubic, c / safe_a, zero)
    d_a = torch.where(mask_cubic, d / safe_a, zero)

    # Scale the independent variable by an exact power of two. Its detached
    # exponent is a piecewise-constant conditioning choice, not a derivative.
    bound = torch.maximum(torch.maximum(b_a.abs(), c_a.abs().sqrt()), d_a.abs().pow(1.0 / 3.0)).detach()
    positive_bound = bound > 0
    exponent = torch.floor(torch.log2(torch.where(positive_bound, bound, one))) + 1
    exponent = torch.where(positive_bound, exponent, zero)
    scale, inv_scale = _exact_powers_of_two(exponent)
    b_a = b_a * inv_scale
    c_a = c_a * inv_scale * inv_scale
    d_a = d_a * inv_scale * inv_scale * inv_scale
    b_a2 = b_a * b_a
    q = (3 * c_a - b_a2) / 9
    r = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a2) / 54
    q3 = q * q * q
    discriminant = q3 + r * r
    shift = b_a / 3

    q_zero = q == 0
    r_zero = r == 0
    cubic_q_zero = mask_cubic & q_zero
    cubic_q_nonzero = mask_cubic & ~q_zero
    mask_q_only = cubic_q_zero & ~r_zero
    mask_qr_zero = cubic_q_zero & r_zero
    mask_three = cubic_q_nonzero & (discriminant <= 0)
    mask_one = cubic_q_nonzero & (discriminant > 0)

    # A captured graph evaluates every branch at fixed shape. Eager execution
    # reads all branch flags with one host synchronization and skips the
    # branches no row takes; their torch.where selections would be identities.
    compiling = torch.compiler.is_compiling()
    if compiling:
        has_q_only = has_qr_zero = has_three = has_one = has_second_order = has_first_order = True
    else:
        flags = torch.stack([mask_q_only, mask_qr_zero, mask_three, mask_one, mask_second_order, mask_first_order])
        has_q_only, has_qr_zero, has_three, has_one, has_second_order, has_first_order = flags.any(-1).tolist()

    q_only_root = zero
    if has_q_only:
        q_only_q = torch.where(mask_q_only, q, zero)
        q_only_r = torch.where(mask_q_only, r, one)
        a_q_only = torch.sign(q_only_r) * torch.pow(2 * q_only_r.abs(), 1.0 / 3.0)
        q_only_root = a_q_only - q_only_q / a_q_only - shift

    # At the acos boundary use its exact value from a detached tensor and a
    # safe interior input for backward; repeated-root derivatives are undefined.
    three_roots = zero[:, None].expand(-1, 3)
    if has_three:
        three_q = torch.where(mask_three, q, -one)
        three_r = torch.where(mask_three, r, zero)
        three_q3 = three_q * three_q * three_q
        ratio = torch.clamp(three_r / torch.sqrt(-three_q3), min=-1.0, max=1.0)
        at_boundary = ratio.abs() >= 1.0
        theta = torch.where(at_boundary, ratio.detach().acos(), torch.where(at_boundary, zero, ratio).acos())
        sqrt_q = torch.sqrt(-three_q)
        three_roots = torch.stack(
            [
                2 * sqrt_q * torch.cos(theta / 3.0) - shift,
                2 * sqrt_q * torch.cos((theta + 2 * math.pi) / 3.0) - shift,
                2 * sqrt_q * torch.cos((theta + 4 * math.pi) / 3.0) - shift,
            ],
            dim=-1,
        )

    one_root = zero
    if has_one:
        one_q = torch.where(mask_one, q, one)
        one_r = torch.where(mask_one, r, one)
        one_d = torch.where(mask_one, discriminant, one)
        a_one = torch.pow(one_r.abs() + torch.sqrt(one_d), 1.0 / 3.0)
        a_one = torch.where(one_r < 0, -a_one, a_one)
        b_one = -one_q / a_one
        quotient_is_better = (one_q > 0) & (a_one * a_one < 4.79 * one_q)
        sum_ab = torch.where(quotient_is_better, 2 * one_r / (a_one * a_one + b_one * b_one + one_q), a_one + b_one)
        one_root = sum_ab - shift

    cubic_roots = zero[:, None].expand(-1, 3)
    if has_q_only:
        cubic_roots = torch.where(mask_q_only[:, None], torch.stack([q_only_root, zero, zero], dim=-1), cubic_roots)
    if has_qr_zero:
        cubic_roots = torch.where(mask_qr_zero[:, None], (-shift)[:, None].expand(-1, 3), cubic_roots)
    if has_three:
        cubic_roots = torch.where(mask_three[:, None], three_roots, cubic_roots)
    if has_one:
        cubic_roots = torch.where(mask_one[:, None], torch.stack([one_root, zero, zero], dim=-1), cubic_roots)
    cubic_roots = cubic_roots * scale[:, None]
    cubic_count = torch.where(mask_qr_zero | mask_three, 3, torch.where(mask_cubic, 1, 0))

    # A dominant root makes the closed-form discriminant ill-conditioned. The
    # detached gate permits us to form Vieta's smaller quadratic only where it
    # will be used, keeping unrelated lanes' backward values finite.
    slot0 = cubic_roots.abs().argmax(dim=-1, keepdim=True)
    dominant = cubic_roots.gather(1, slot0).squeeze(1)
    dominant_detached = dominant.detach()
    safe_dominant = torch.where(dominant_detached == 0, one, dominant_detached)
    detached_lead = a.detach() * safe_dominant
    detached_product = -d.detach() / detached_lead
    other_scale = detached_product.abs().sqrt()
    # The largest root is at least a third of the root bound, which is above half the scale. A smaller
    # closed-form root is a cancellation remnant beside a complex pair, e.g. the root 0 of x^3 + x^2 + 3x, and
    # Vieta's quotients by it would invent a real pair of size 1 / remnant.
    mask_dominant = (
        mask_cubic
        & (dominant_detached != 0)
        & (dominant_detached.abs() > _DOMINANT_ROOT_RATIO * other_scale)
        & (8 * dominant_detached.abs() >= scale)
    )

    if compiling:
        dominant_for_vieta = torch.where(mask_dominant, dominant, one)
        a_for_vieta = torch.where(mask_dominant, a, one)
        c_for_vieta = torch.where(mask_dominant, c, zero)
        d_for_vieta = torch.where(mask_dominant, d, zero)
        dominant_roots, pair_is_real = _cubic_roots_beside_dominant(
            a_for_vieta, c_for_vieta, d_for_vieta, dominant_for_vieta, dominant, slot0, cubic_roots
        )
        cubic_roots = torch.where(mask_dominant[:, None], dominant_roots, cubic_roots)
        cubic_count = torch.where(mask_dominant, torch.where(pair_is_real, 3, 1), cubic_count)
    elif bool(mask_dominant.any()):
        # Eager execution solves only the dominant rows; every operation is row-wise, so they keep their values.
        # The clone mirrors the separate placeholder tensor above, so gradients accumulate in the same order.
        rows = mask_dominant.nonzero().squeeze(1)
        dominant_rows = dominant[rows]
        dominant_roots, pair_is_real = _cubic_roots_beside_dominant(
            a[rows], c[rows], d[rows], dominant_rows.clone(), dominant_rows, slot0[rows], cubic_roots[rows]
        )
        cubic_roots = cubic_roots.index_put((rows,), dominant_roots)
        cubic_count = cubic_count.index_put((rows,), torch.where(pair_is_real, 3, 1))

    # Lower degrees use the public quadratic convention, with inputs selected
    # before evaluation so cubic rows cannot poison their discarded gradients.
    roots = torch.where(mask_cubic[:, None], cubic_roots, zero[:, None])
    num_real = torch.where(mask_cubic, cubic_count, 0)
    if has_second_order:
        quad_coeffs = torch.where(mask_second_order[:, None], coeffs[:, 1:], torch.stack([one, zero, zero], dim=-1))
        quadratic_roots = _solve_quadratic(quad_coeffs)
        quadratic_padded = torch.cat([quadratic_roots, zero[:, None]], dim=-1)
        quadratic_a, quadratic_b, quadratic_c = _scaled_quadratic_coefficients(quad_coeffs)
        quadratic_delta = quadratic_b * quadratic_b - 4 * (quadratic_a * quadratic_c)
        roots = torch.where(mask_second_order[:, None], quadratic_padded, roots)
        num_real = torch.where(mask_second_order, torch.where(quadratic_delta < 0, 0, 2), num_real)

    if has_first_order:
        first_c = torch.where(mask_first_order, c, one)
        first_d = torch.where(mask_first_order, d, zero)
        first_a = torch.where(mask_first_order, a, zero)
        first_b = torch.where(mask_first_order, b, zero)
        first_root = -first_d / first_c
        first_root = first_root - (first_a * first_root + first_b) * first_root * first_root / first_c
        roots = torch.where(mask_first_order[:, None], torch.stack([first_root, zero, zero], dim=-1), roots)
        num_real = torch.where(mask_first_order, 1, num_real)
    return roots, num_real


def _cubic_roots_beside_dominant(
    a: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    dominant_for_vieta: torch.Tensor,
    dominant: torch.Tensor,
    slot0: torch.Tensor,
    previous: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Complete a cubic's dominant root with Vieta's quadratic for the other two, row by row.

    ``dominant_for_vieta`` is ``dominant`` on the rows that use the result and a safe placeholder elsewhere. The other
    two roots take the slots left by ``slot0`` in the order closest to the ``previous`` closed-form values.
    """
    lead = a * dominant_for_vieta
    product = -d / lead
    total = (c + d / dominant_for_vieta) / lead
    others = _solve_quadratic(torch.stack([torch.ones_like(total), -total, product], dim=-1))
    pair_is_real = total * total - 4 * product >= 0
    rest = torch.sort(torch.cat([(slot0 + 1) % 3, (slot0 + 2) % 3], dim=-1), dim=-1).values
    previous_rest = previous.gather(1, rest)
    swap = (previous_rest - others.flip(-1)).abs().sum(-1) < (previous_rest - others).abs().sum(-1)
    others = torch.where(swap[:, None], others.flip(-1), others)
    with_pair = torch.zeros_like(previous).scatter(1, slot0, dominant[:, None]).scatter(1, rest, others)
    zero = torch.zeros_like(dominant)
    alone = torch.stack([dominant, zero, zero], dim=-1)
    return torch.where(pair_is_real[:, None], with_pair, alone), pair_is_real


def _solve_cubic_real(coeffs: torch.Tensor, polish: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Real roots ``(B, 3)`` of the cubics ``coeffs (B, 4)``, highest degree first, and a mask of the genuine ones.

    Cardano's formula for one real root, the trigonometric one for three, followed by a Newton step. A cubic with one
    real root repeats it in the two masked slots, so a caller builds every candidate from finite roots and masks
    afterwards. The caller chooses a well-conditioned pencil parametrization with a nonzero leading coefficient.

    A private kernel for the seven-point solvers rather than :func:`solve_cubic`, whose public contract differs where a
    hot loop cares: it pads missing roots with 0.0, indistinguishable from a genuine root at 0, and defines a
    surrogate backward at repeated roots. The private kernel assumes a nonzero leading coefficient.
    Here the closed form runs without gradient and the Newton step carries it:
    at a simple root that is the implicit-function derivative ``-(dp/dc) / p'(x)``, and the ``clamp``-guarded
    ``sqrt``, ``acos`` and cube roots of the closed form, whose derivatives are unbounded at their bounds (#4229),
    never enter the backward pass. At a repeated root ``p'(x) = 0``; the division is guarded and the gradient finite.
    ``polish=False`` returns the detached closed form for Ferrari, whose own root polishing supplies the Jacobian.
    """
    c3, c2, c1, c0 = coeffs.unbind(1)
    a, b, c = c2 / c3, c1 / c3, c0 / c3
    with torch.no_grad():
        a3 = a / 3
        p = b - a * a3
        q = (2 * a3 * a3 - b) * a3 + c
        discriminant = 0.25 * q * q + p * p * p / 27
        # Cancellation at a repeated root can round the discriminant slightly positive.
        discriminant_scale = 0.25 * q.square() + p.abs().pow(3) / 27
        three = discriminant <= 32 * torch.finfo(coeffs.dtype).eps * discriminant_scale
        root = discriminant.clamp(min=0).sqrt()
        u, w = root - 0.5 * q, -root - 0.5 * q
        single = torch.copysign(u.abs().pow(1 / 3), u) + torch.copysign(w.abs().pow(1 / 3), w)
        radius = (-p / 3).clamp(min=0).sqrt()
        safe_radius = torch.where(radius > 0, radius, torch.ones_like(radius))
        angle = torch.acos((-0.5 * q / safe_radius.pow(3)).clamp(-1, 1)) / 3
        offsets = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3], dtype=c3.dtype, device=c3.device)
        triple = 2 * radius[:, None] * torch.cos(angle[:, None] - offsets)
        x = torch.where(three[:, None], triple, single[:, None].expand(-1, 3)) - a3[:, None]
    if not polish:
        return x, torch.stack([torch.ones_like(three), three, three], 1)
    value = ((x + a[:, None]) * x + b[:, None]) * x + c[:, None]
    slope = (3 * x + 2 * a[:, None]) * x + b[:, None]
    # At repeated roots rounding can leave a tiny nonzero slope: dividing two rounding errors then moves an
    # already accurate root far away. Bound the derivative relative to its terms, including their cancellation.
    slope_scale = 3 * x.square() + 2 * a[:, None].abs() * x.abs() + b[:, None].abs()
    simple = slope.abs() > 8 * torch.finfo(coeffs.dtype).eps * slope_scale
    correction = value / torch.where(simple, slope, torch.ones_like(slope))
    x = x - torch.where(simple, correction, torch.zeros_like(correction))
    valid = torch.stack([torch.ones_like(three), three, three], 1)
    return x, valid


class _QuarticColumns(NamedTuple):
    """Detached ``(B, 1)`` columns of the monic quartic ``x^4 + a x^3 + b x^2 + c x + d``.

    They feed root classification and rounding-error bounds only, never a derivative, so the derivative
    coefficients and the magnitudes are formed once for every evaluation.
    """

    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    d: torch.Tensor
    a3: torch.Tensor
    a6: torch.Tensor
    b2: torch.Tensor
    abs_a: torch.Tensor
    abs_b: torch.Tensor
    abs_c: torch.Tensor
    abs_d: torch.Tensor
    abs_a3: torch.Tensor
    abs_b2: torch.Tensor


@torch.no_grad()
def _quartic_columns(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> _QuarticColumns:
    a, b, c, d = (v.detach()[:, None] for v in (a, b, c, d))
    abs_a, abs_b, abs_c, abs_d = a.abs(), b.abs(), c.abs(), d.abs()
    return _QuarticColumns(a, b, c, d, 3 * a, 6 * a, 2 * b, abs_a, abs_b, abs_c, abs_d, 3 * abs_a, 2 * abs_b)


def _quartic_value_scale(p: _QuarticColumns, ax: torch.Tensor) -> torch.Tensor:
    """Horner sum of the quartic's term magnitudes at ``|x|``."""
    return (((ax + p.abs_a) * ax + p.abs_b) * ax + p.abs_c) * ax + p.abs_d


def _quartic_slope_scale(p: _QuarticColumns, ax: torch.Tensor) -> torch.Tensor:
    """Horner sum of the derivative's term magnitudes at ``|x|``."""
    return ((4 * ax + p.abs_a3) * ax + p.abs_b2) * ax + p.abs_c


@torch.no_grad()
def _quartic_stationary_points(x: torch.Tensor, p: _QuarticColumns) -> tuple[torch.Tensor, torch.Tensor]:
    """Project candidates near a multiple root to a nearby stationary point."""
    ax = x.abs()
    slope = ((4 * x + p.a3) * x + p.b2) * x + p.c
    nearby = slope.abs() <= math.sqrt(torch.finfo(x.dtype).eps) * _quartic_slope_scale(p, ax)
    if not torch.compiler.is_compiling():
        result = x.clone()
        if bool(nearby.any()):
            rows, columns = nearby.nonzero(as_tuple=True)
            t = x[rows, columns]
            a3, a6, b2, c = p.a3[rows, 0], p.a6[rows, 0], p.b2[rows, 0], p.c[rows, 0]
            for _ in range(3):
                slope = ((4 * t + a3) * t + b2) * t + c
                curvature = (12 * t + a6) * t + b2
                t = t - slope / torch.where(curvature != 0, curvature, 1.0)
            result[rows, columns] = t
        return result, nearby
    stationary = x
    for _ in range(3):
        slope = ((4 * stationary + p.a3) * stationary + p.b2) * stationary + p.c
        curvature = (12 * stationary + p.a6) * stationary + p.b2
        stationary = stationary - torch.where(nearby, slope, 0.0) / torch.where(
            nearby & (curvature != 0), curvature, 1.0
        )
    return stationary, nearby


@torch.no_grad()
def _quartic_local_discriminant_is_real(
    coeffs: torch.Tensor, x: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Classify the local Taylor quadratic, returning reality, multiplicity, discriminant, slope and curvature."""
    if mask is not None and not torch.compiler.is_compiling():
        zero = torch.zeros_like(x)
        real = torch.ones_like(x, dtype=torch.bool)
        repeated = torch.zeros_like(real)
        delta, slope, curvature = zero.clone(), zero.clone(), zero.clone()
        if bool(mask.any()):
            rows, columns = mask.nonzero(as_tuple=True)
            selected = _quartic_local_discriminant_is_real(coeffs[rows], x[rows, columns, None])
            for destination, source in zip((real, repeated, delta, slope, curvature), selected):
                destination[rows, columns] = source[:, 0]
        return real, repeated, delta, slope, curvature
    # An exact power-of-two scaling preserves the input polynomial's signs without the
    # rounding introduced by monic normalization, and protects the squared quantities below.
    exponent = torch.floor(torch.log2(coeffs.abs().amax(dim=-1, keepdim=True)))
    coeffs = coeffs * _exact_power_of_two(-exponent)
    # A small p and p' do not distinguish an exact double root from a nearby complex pair.
    # The local quadratic has discriminant p'^2 - 2*p*p''. At a double root it vanishes;
    # at a nearby extremum with no real pair it is negative. Ordinary Horner rounding can
    # hide the sign, so evaluate p with compensated Horner (Graillat, 2008, Algorithm 4):
    # https://doi.org/10.1016/j.camwa.2008.02.027
    # TwoProduct splits each operand into high/low halves; TwoSum retains each addition's
    # rounding error. The correction polynomial is accumulated alongside ordinary Horner.
    splitter = 2.0**27 + 1.0 if x.dtype == torch.float64 else 2.0**12 + 1.0
    split_x = splitter * x
    x_high = split_x - (split_x - x)
    x_low = x - x_high
    value = coeffs[:, :1].expand_as(x)
    correction = torch.zeros_like(x)
    for coefficient in coeffs[:, 1:].unbind(-1):
        product = value * x
        split_value = splitter * value
        value_high = split_value - (split_value - value)
        value_low = value - value_high
        product_error = ((value_high * x_high - product) + value_high * x_low + value_low * x_high) + value_low * x_low
        total = product + coefficient[:, None]
        z = total - product
        sum_error = (product - (total - z)) + (coefficient[:, None] - z)
        correction = correction * x + (product_error + sum_error)
        value = total
    value = value + correction

    a, b, c, d, e = (v[:, None] for v in coeffs.unbind(-1))
    slope = ((4.0 * a * x + 3.0 * b) * x + 2.0 * c) * x + d
    curvature = (12.0 * a * x + 6.0 * b) * x + 2.0 * c
    ax = x.abs()
    value_scale = (((a.abs() * ax + b.abs()) * ax + c.abs()) * ax + d.abs()) * ax + e.abs()
    slope_scale = ((4.0 * a.abs() * ax + 3.0 * b.abs()) * ax + 2.0 * c.abs()) * ax + d.abs()
    curvature_scale = (12.0 * a.abs() * ax + 6.0 * b.abs()) * ax + 2.0 * c.abs()
    # u = eps/2; gamma_8 bounds eight rounded operations. Compensated degree-four
    # Horner has error u*|p| + gamma_8^2*sum(|a_i*x^i|), rather than O(eps)*scale.
    # Use eps in the last rounding allowance and propagate the derivative errors into
    # the discriminant. This budget is derived from the evaluation, not the root grid.
    eps = torch.finfo(x.dtype).eps
    gamma = 4.0 * eps / (1.0 - 4.0 * eps)
    value_error = gamma**2 * value_scale + eps * value.abs()
    slope_error = gamma * slope_scale
    curvature_error = gamma * curvature_scale
    discriminant = slope.square() - 2.0 * curvature * value
    error = (2.0 * slope.abs() + slope_error) * slope_error
    error = error + 2.0 * ((curvature.abs() + curvature_error) * value_error + value.abs() * curvature_error)
    error = error + eps * (slope.square() + 2.0 * (curvature * value).abs())
    real = torch.isfinite(error) & (discriminant >= -error)
    repeated = torch.isfinite(error) & (discriminant.abs() <= error)
    if mask is not None:
        real = ~mask | real
        repeated = mask & repeated
    return real, repeated, discriminant, slope, curvature


def solve_quartic(coeffs: torch.Tensor) -> torch.Tensor:
    r"""Solve given quartic equation.

    The function takes the coefficients of quartic equation and returns
    the real roots.

    .. math:: coeffs[0]x^4 + coeffs[1]x^3 + coeffs[2]x^2 + coeffs[3]x + coeffs[4] = 0

    Convention:
        - Coefficient layout and zero padding as :func:`solve_quadratic`. Genuine quartic roots are sorted in
          descending order, followed by padding; lower-degree rows preserve :func:`solve_cubic`'s order.
        - Quartics are evaluated after an exact power-of-two variable rescaling to a unit root bound.
          Root reality and multiplicity near a stationary point are resolved against the original coefficients
          with compensated Horner evaluation, before monic normalization can erase the distinction.
        - A row is solved as the cubic of its last four coefficients when its leading coefficient is 0, or when both
          hold: ``|a|`` is smaller than ``1e-6`` (``1e-12`` in float64) times ``min(1, max_i |coeffs_i|)``, and the
          scale-invariant root bound ``max(|b/a|, |c/a|^(1/2), |d/a|^(1/3), |e/a|^(1/4))`` exceeds ``1 / tol``
          (tested as ``|coeffs_k| > |a| / tol^k``, without dividing by ``a``). Since the bound is at most 4 times
          the largest root's magnitude, a quartic whose roots are all smaller than ``1 / (4 * tol)`` stays a
          quartic at every scale; the bound can only move a row from the cubic path to the quartic path.

    Args:
        coeffs : The coefficients quartic equation : `(B, 5)`

    Returns:
        A torch.Tensor of shape `(B, 4)` containing the real roots to the quartic equation.

    Example:
        >>> coeffs = torch.tensor([[1., -10., 35., -50., 24.]])
        >>> roots = solve_quartic(coeffs)

    .. note::
       Ferrari intermediates and root polishing are evaluated in ``float64`` and the returned roots
       preserve the input dtype and device. On MPS, which has no float64, the quartic computation
       runs on the CPU; autograd follows the copies in both directions.

    .. note::
       Variable rescaling is bounded to normal reciprocal powers of two in the compute dtype. It does not
       recover input coefficients that have underflowed, and extreme subnormal coefficient scales may remain
       below unit scale.

    .. note::
       Simple roots use the implicit-function Jacobian, coupled to preserve Vieta's sum
       when all four roots are real. A repeated-root Jacobian is undefined; repeated roots
       use a finite surrogate with the same sum convention. Coefficient rounding can turn a generating
       repeated root into a complex pair; the solver classifies the represented input polynomial.

    .. note::
       The same surrogate convention applies at this function's own two ``sqrt`` boundaries:
       when the resolvent radicand ``R^2`` is 0 (a pure biquadratic such as :math:`x^4 - 16`)
       and when the constant-term identity used for ``E`` has a zero radicand, backward suppresses
       the diverging ``sqrt`` derivative to keep gradients finite. The result is finite but is not
       the root Jacobian; the forward values are unaffected.
    """
    KORNIA_CHECK_SHAPE(coeffs, ["B", "5"])

    original_dtype = coeffs.dtype
    # Evaluate Ferrari in double precision from the input coefficients, before forming a rounded resolvent.
    # MPS has no float64 arithmetic; the copies remain differentiable.
    if coeffs.device.type == "mps":
        work = coeffs.cpu().double()
    else:
        work = coeffs.double()
    zero_tol = 1e-12 if original_dtype == torch.float64 else 1e-6
    # Selections, rounding-error bounds and conditioning choices below never carry a derivative;
    # forming them without autograd leaves every gradient unchanged.
    with torch.no_grad():
        absolute = work.abs()
        abs_a, abs_b, abs_c, abs_d, abs_e = absolute.unbind(-1)
        row_scale = absolute.amax(-1).clamp(max=1.0)
        bound = (
            (abs_b > abs_a / zero_tol)
            | (abs_c > abs_a / zero_tol**2)
            | (abs_d > abs_a / zero_tol**3)
            | (abs_e > abs_a / zero_tol**4)
        )
        lower = (abs_a == 0) | ((abs_a < zero_tol * row_scale) & bound)
    if not torch.compiler.is_compiling() and bool(lower.all()):
        lower_roots = _solve_cubic(coeffs[:, 1:])[0]
        return torch.cat([lower_roots, torch.zeros_like(coeffs[:, :1])], -1)
    fallback = torch.tensor([1.0, 0.0, 0.0, 0.0, -1.0], dtype=work.dtype, device=work.device)
    quartic_coeffs = torch.where(lower[:, None], fallback, work)
    a_q, b_q, c_q, d_q, e_q = quartic_coeffs.unbind(-1)
    A, B, C, D = b_q / a_q, c_q / a_q, d_q / a_q, e_q / a_q
    with torch.no_grad():
        root_bound = torch.maximum(
            torch.maximum(A.abs(), B.abs().sqrt()), torch.maximum(C.abs().pow(1 / 3), D.abs().sqrt().sqrt())
        )
        positive = root_bound > 0
        exponent = torch.floor(torch.log2(torch.where(positive, root_bound, torch.ones_like(root_bound))))
        exponent = torch.where(positive, exponent, torch.zeros_like(exponent))
        variable_scale, inverse_scale = _exact_powers_of_two(exponent)
    A = A * inverse_scale
    B = B * inverse_scale * inverse_scale
    C = C * inverse_scale * inverse_scale * inverse_scale
    D = D * inverse_scale * inverse_scale * inverse_scale * inverse_scale

    # Keep the established pure-biquadratic boundary surrogate for the linear term.
    pure_biquadratic = (A == 0) & (B == 0) & (C == 0)
    C = torch.where(pure_biquadratic, C.detach(), C)

    # Resolvent cubic coefficients. Its roots are taken detached, so the resolvent carries no derivative.
    with torch.no_grad():
        rc_a = torch.ones_like(A)
        rc_b = -B
        rc_c = A * C - 4.0 * D
        rc_d = -1.0 * (A * A * D - 4.0 * B * D + C * C)

        cubic_coeffs = torch.stack([rc_a, rc_b, rc_c, rc_d], dim=1)

        # The largest real resolvent root gives the best separated Ferrari factors. The private
        # fixed-shape kernel supplies an explicit validity mask instead of zero placeholders.
        y_roots, valid = _solve_cubic_real(cubic_coeffs, polish=False)
        # The trigonometric form's error is relative to the largest root. When the quartic's roots span many
        # decades, two resolvent roots sit far below the third and come back with no correct digits, sometimes
        # with the wrong sign. As in solve_cubic, take them from Vieta's relations with the dominant root. Its
        # gate also requires the dominant root to reach the root bound, of which the largest root is at least a
        # third: a smaller one is a remnant of a closed form that has lost a complex pair.
        slot0 = y_roots.abs().argmax(-1, keepdim=True)
        dominant = y_roots.gather(1, slot0).squeeze(1)
        safe_dominant = torch.where(dominant != 0, dominant, torch.ones_like(dominant))
        resolvent_bound = torch.maximum(torch.maximum(rc_b.abs(), rc_c.abs().sqrt()), rc_d.abs().pow(1 / 3))
        beside = (
            valid.all(-1)
            & (dominant.abs() > _DOMINANT_ROOT_RATIO * (rc_d / safe_dominant).abs().sqrt())
            & (8 * dominant.abs() >= resolvent_bound)
        )
        vieta_roots, pair_is_real = _cubic_roots_beside_dominant(
            rc_a, rc_c, rc_d, safe_dominant, dominant, slot0, y_roots
        )
        y_roots = torch.where((beside & pair_is_real)[:, None], vieta_roots, y_roots)
    A_sq = A * A
    candidates = 0.25 * A_sq[:, None] - B[:, None] + y_roots
    index = torch.where(valid, candidates, -torch.inf).argmax(-1, keepdim=True)
    y = y_roots.gather(1, index).squeeze(1)
    R_sq = candidates.gather(1, index).squeeze(1)

    # R^2 = A^2 / 4 - B + y can retain cancellation-level roundoff for an exact biquadratic.
    # Float64 needs a wider four-epsilon budget for cancellation between those terms, while
    # float32 (including half inputs evaluated in float32) keeps the narrower half-epsilon budget
    # so genuine small positive R^2 values are not snapped away. This happens before the guarded
    # sqrt so the exact-zero gradient convention remains unchanged.
    with torch.no_grad():
        R_sq_scale = torch.maximum(torch.ones_like(R_sq), 0.25 * A_sq + torch.abs(B) + torch.abs(y))
        R_sq_snap_multiplier = 4.0 if R_sq.dtype == torch.float64 else 0.5
        R_sq_snap_tol = R_sq_snap_multiplier * torch.finfo(R_sq.dtype).eps * R_sq_scale
        R_sq_snapped = torch.abs(R_sq) <= R_sq_snap_tol
    R_sq = torch.where(R_sq_snapped, torch.zeros_like(R_sq), R_sq)

    # `clamp(min=0).sqrt()` does not guard the gradient: d(sqrt)/dx is unbounded at 0, and on
    # torch < 2.14 clamp passes the incoming gradient straight through at the bound, as measured in PR #4406, so
    # R_sq == 0 -- a biquadratic such as x^4 - 16 -- gave inf and then nan. Substitute a safe
    # radicand instead, as solve_quadratic above already does, so sqrt is never differentiated at 0.
    # On torch >= 2.14 clamp already zeroes the boundary gradient, so the pins for this guard pass
    # on base there too; the 2.5.1 and 2.9.1 CI legs are the ones that discriminate.
    mask_R_sq_positive = R_sq > 0
    R = torch.where(
        mask_R_sq_positive,
        torch.sqrt(torch.where(mask_R_sq_positive, R_sq, torch.ones_like(R_sq))),
        torch.zeros_like(R_sq),
    )

    # Compute |E| from the constant-term identity E^2 = y^2 / 4 - D instead of dividing by R.
    # The sign follows the equivalent cross term A*y - 2*C; at an exact zero cross term, choose
    # the positive branch to preserve the previous R~=0 forward convention. Guard the sqrt the
    # same way as R above so a zero radicand keeps a finite surrogate gradient (#4339).
    E_radicand = 0.25 * y * y - D
    mask_E_radicand_positive = E_radicand > 0
    E_magnitude = torch.where(
        mask_E_radicand_positive,
        torch.sqrt(torch.where(mask_E_radicand_positive, E_radicand, torch.ones_like(E_radicand))),
        torch.zeros_like(E_radicand),
    )
    E_cross_term = A * y - 2.0 * C
    E_constant = torch.where(E_cross_term < 0, -E_magnitude, E_magnitude)

    # Away from R == 0, the division form can satisfy the linear coefficient more accurately,
    # while the constant-term form remains stable near the division singularity. Compare their
    # normalized Ferrari coefficient reconstruction errors in the actual Ferrari compute dtype.
    safe_R = torch.where(R > 0, R, torch.ones_like(R))
    E_division = E_cross_term / (4.0 * safe_R)
    with torch.no_grad():
        root_sq_scale = torch.maximum(torch.ones_like(R_sq), A_sq)
        root_sq_scale = torch.maximum(root_sq_scale, torch.abs(B))
        root_sq_scale = torch.maximum(root_sq_scale, torch.abs(y))
        root_scale = torch.sqrt(root_sq_scale)

        division_C_error = torch.abs(0.5 * A * y - 2.0 * R * E_division - C) / (root_sq_scale * root_scale)
        division_D_error = torch.abs(0.25 * y * y - E_division * E_division - D) / (root_sq_scale * root_sq_scale)
        constant_C_error = torch.abs(0.5 * A * y - 2.0 * R * E_constant - C) / (root_sq_scale * root_scale)
        constant_D_error = torch.abs(0.25 * y * y - E_constant * E_constant - D) / (root_sq_scale * root_sq_scale)
        division_error = torch.maximum(division_C_error, division_D_error)
        constant_error = torch.maximum(constant_C_error, constant_D_error)
        use_constant_E = (R == 0) | (constant_error < division_error)
    E = torch.where(use_constant_E, E_constant, E_division)

    # Solve two resulting quadratic equations
    # Quad 1: x^2 + (A/2 - R)x + (y/2 - E) = 0
    q1_b = 0.5 * A - R
    q1_c = 0.5 * y - E

    # Quad 2: x^2 + (A/2 + R)x + (y/2 + E) = 0
    q2_b = 0.5 * A + R
    q2_c = 0.5 * y + E

    # When one root pair is much larger than the other, the small factor's constant y/2 -+ E is the difference of two
    # numbers of the large pair's size, and in float32 that cancellation flips its discriminant: the small real roots
    # come back as zero placeholders. The large factor has no such cancellation. Rebuild the small factor from
    # it with Vieta's relations for the product (x^2 + b1 x + c1)(x^2 + b2 x + c2): c1 * c2 = D and
    # b1 * c2 + b2 * c1 = C, as solve_quadratic takes its small root as c / q. A separation of _DOMINANT_ROOT_RATIO
    # between the two constants decides it; the roots do not depend on the ratio anywhere from 2 to 1024.
    first_small = torch.abs(q1_c) <= torch.abs(q2_c)
    c_large = torch.where(first_small, q2_c, q1_c)
    b_large = torch.where(first_small, q2_b, q1_b)
    separated = _DOMINANT_ROOT_RATIO * torch.minimum(torch.abs(q1_c), torch.abs(q2_c)) < torch.abs(c_large)
    safe_c_large = torch.where(separated, c_large, torch.ones_like(c_large))
    c_small = D / safe_c_large
    b_small = (C - b_large * c_small) / safe_c_large
    q1_b = torch.where(separated & first_small, b_small, q1_b)
    q1_c = torch.where(separated & first_small, c_small, q1_c)
    q2_b = torch.where(separated & ~first_small, b_small, q2_b)
    q2_c = torch.where(separated & ~first_small, c_small, q2_c)

    factor_b = torch.stack([q1_b, q2_b], -1)
    factor_c = torch.stack([q1_c, q2_c], -1)
    discriminant = factor_b.square() - 4 * factor_c
    midpoint = -0.5 * factor_b
    # A factor discriminant near zero is sensitive to resolvent rounding. Resolve its sign
    # against the original polynomial using compensated Horner; never polish a padding value.
    eps = torch.finfo(work.dtype).eps
    compiling = torch.compiler.is_compiling()
    columns = _quartic_columns(A, B, C, D)
    stationary, _ = _quartic_stationary_points(midpoint, columns)
    midpoint = midpoint + (stationary - midpoint).detach()
    with torch.no_grad():
        ax = midpoint.abs()
        value = (((midpoint + columns.a) * midpoint + columns.b) * midpoint + columns.c) * midpoint + columns.d
        slope = ((4 * midpoint + columns.a3) * midpoint + columns.b2) * midpoint + columns.c
        uncertain = (value.abs() <= math.sqrt(eps) * _quartic_value_scale(columns, ax)) & (
            slope.abs() <= math.sqrt(eps) * _quartic_slope_scale(columns, ax)
        )
    # Eager execution skips the local refinement when no factor is uncertain: every
    # selection below would then keep the Ferrari factor.
    if compiling or bool(uncertain.any()):
        locally_real, locally_double, local_delta, local_slope, local_curvature = _quartic_local_discriminant_is_real(
            quartic_coeffs, midpoint * variable_scale[:, None], uncertain
        )
        # In a tight root pair the local Taylor quadratic resolves a discriminant that
        # the global Ferrari reconstruction loses. Restrict it to a locally quadratic
        # region: the cubic and quartic terms over the factor radius are bounded by
        # one eighth of the curvature term, so a wide factor around a double root
        # cannot replace the other two simple roots.
        with torch.no_grad():
            width_sq = discriminant.abs() / 4
            curvature = (12 * midpoint + columns.a6) * midpoint + columns.b2
            cubic_term = (24 * midpoint + columns.a6).abs() * width_sq.sqrt() / 6
            local_region = cubic_term + width_sq <= curvature.abs() / 8
            use_local = uncertain & local_region & (local_curvature != 0)
        safe_curvature = torch.where(use_local, local_curvature, torch.ones_like(local_curvature))
        local_center = midpoint - local_slope / safe_curvature / variable_scale[:, None]
        local_discriminant = 4 * local_delta / safe_curvature.square() / variable_scale[:, None].square()
        real_double = use_local & locally_double
        local_discriminant = torch.where(real_double, torch.zeros_like(local_discriminant), local_discriminant)
        discriminant = torch.where(use_local, local_discriminant, discriminant)
        factor_b = torch.where(use_local, -2 * local_center, factor_b)
        factor_c = torch.where(use_local, local_center.square() - local_discriminant / 4, factor_c)
        real = torch.where(use_local, locally_real, discriminant >= 0)
    else:
        use_local = real_double = torch.zeros_like(uncertain)
        real = discriminant >= 0
    positive = discriminant > 0
    radius = torch.where(positive, torch.sqrt(torch.where(positive, discriminant, 1.0)), 0.0)
    b_nonnegative = factor_b >= 0
    q = -0.5 * (factor_b + torch.where(b_nonnegative, radius, -radius))
    other = torch.where(positive, factor_c / torch.where(positive, q, 1.0), q)
    plus = torch.where(b_nonnegative, other, q)
    minus = torch.where(b_nonnegative, q, other)
    roots = torch.stack([plus, minus], -1).flatten(1)
    genuine = real.repeat_interleave(2, -1)
    shift = A / 4
    depressed_q = C - 2 * shift * B + 8 * shift.pow(3)
    biquadratic = depressed_q == 0
    if compiling or bool(biquadratic.any()):
        depressed_p = B - 6 * shift.square()
        depressed_r = D - shift * C + shift.square() * B - 3 * shift.pow(4)
        z = _solve_quadratic(torch.stack([torch.ones_like(A), depressed_p, depressed_r], -1))
        z_real = depressed_p.square() - 4 * depressed_r >= 0
        z_positive = z > 0
        z_radius = torch.where(z_positive, torch.sqrt(torch.where(z_positive, z, 1.0)), 0.0)
        bi_roots = torch.stack([z_radius - shift[:, None], -z_radius - shift[:, None]], -1).flatten(1)
        bi_real = ((z >= 0) & z_real[:, None]).repeat_interleave(2, -1)
        roots = torch.where(biquadratic[:, None], bi_roots, roots)
        genuine = torch.where(biquadratic[:, None], bi_real, genuine)

    # Trailing zero coefficients d = e = 0 make 0 a multiple root, which Ferrari only approximates: the
    # factors' constants are rounding noise, and the pair at 0 can leave as padding or take a simple root
    # with it. Deflate x^k instead and solve the rest with solve_cubic's lower-degree convention, so the
    # zeros are exact and the other roots continue as Ferrari's candidates would. A single trailing zero
    # needs no deflation: Newton reaches the simple root at 0 exactly.
    zero_pair = (d_q == 0) & (e_q == 0)
    if compiling or bool(zero_pair.any()):
        with torch.no_grad():
            c_zero = zero_pair & (c_q == 0)
            b_zero = c_zero & (b_q == 0)
            ones, zeros = torch.ones_like(a_q), torch.zeros_like(a_q)
            deflated = torch.where(
                b_zero[:, None],
                torch.stack([zeros, zeros, zeros, a_q], -1),
                torch.where(
                    c_zero[:, None],
                    torch.stack([zeros, zeros, a_q, b_q], -1),
                    torch.stack([zeros, a_q, b_q, c_q], -1),
                ),
            )
            deflated = torch.where(zero_pair[:, None], deflated, torch.stack([zeros, ones, zeros, -ones], -1))
            deflated_roots, deflated_count = _solve_cubic(deflated)
            # Slots below the deflated degree hold its roots; the remaining slots are the zeros.
            degree = 2 - c_zero.long() - b_zero.long()
            slot = torch.arange(4, device=roots.device)
            deflated_genuine = (slot < deflated_count[:, None]) | (slot >= degree[:, None])
            deflated_roots = torch.cat([deflated_roots, zeros[:, None]], -1) * inverse_scale[:, None]
        roots = torch.where(zero_pair[:, None], deflated_roots, roots)
        genuine = torch.where(zero_pair[:, None], deflated_genuine, genuine)
        use_local = use_local & ~zero_pair[:, None]
        real_double = real_double & ~zero_pair[:, None]

    with torch.no_grad():
        stationary, unresolved = _quartic_stationary_points(roots, columns)
        root_locally_real, is_double, _, _, _ = _quartic_local_discriminant_is_real(
            quartic_coeffs, stationary * variable_scale[:, None], unresolved
        )
        genuine = genuine & root_locally_real
        repeated_roots = genuine & is_double
        newton = genuine & ~repeated_roots
    roots = torch.where(repeated_roots, roots + (stationary - roots).detach(), roots)
    a1, b1, c1, d1 = A[:, None], B[:, None], C[:, None], D[:, None]
    a3, b2 = 3 * a1, 2 * b1
    # Guard the Newton division at unresolved multiple roots to preserve multiplicity.
    value = (((roots + a1) * roots + b1) * roots + c1) * roots + d1
    for _ in range(4):
        slope = ((4 * roots + a3) * roots + b2) * roots + c1
        with torch.no_grad():
            simple = newton & (slope.abs() > 8 * eps * _quartic_slope_scale(columns, roots.abs()))
        candidate = roots - torch.where(simple, value, 0.0) / torch.where(simple, slope, 1.0)
        residual = (((candidate + a1) * candidate + b1) * candidate + c1) * candidate + d1
        improved = simple & (residual.abs() <= value.abs())
        # Once no root moves, every remaining step would repeat this one exactly; eager execution stops.
        if not compiling and not bool((improved & (candidate != roots)).any()):
            break
        roots = torch.where(improved, candidate, roots)
        value = torch.where(improved, residual, value)
    # Attach the implicit-function Jacobian at simple roots without moving their values: the
    # detached difference is exactly zero, whereas (fixed - correction) + correction rounds.
    fixed = roots.detach()
    value_for_grad = (((fixed + a1) * fixed + b1) * fixed + c1) * fixed + d1
    slope_for_grad = ((4 * fixed + a3) * fixed + b2) * fixed + c1
    with torch.no_grad():
        simple = newton & (slope_for_grad.abs() > 8 * eps * _quartic_slope_scale(columns, fixed.abs()))
    correction = value_for_grad / torch.where(simple, slope_for_grad, 1.0)
    implicit = fixed + (correction.detach() - correction)
    roots = torch.where(simple, implicit, roots)
    # Multiple-root Jacobians are undefined. Keep the finite midpoint convention.
    # For close simple roots, tiny forward errors prevent cancellation of their large
    # individual Jacobians; enforce Vieta's sum analytically for every four-real-root row.
    repeated = real_double.repeat_interleave(2, -1) | repeated_roots
    coupled = torch.where(repeated.any(-1, keepdim=True), repeated, use_local.repeat_interleave(2, -1))
    count = coupled.sum(-1, keepdim=True).clamp(min=1)
    sum_error = -A[:, None] - roots.sum(-1, keepdim=True)
    adjustment = (sum_error - sum_error.detach()) / count
    roots = roots + torch.where(coupled & genuine.all(-1, keepdim=True), adjustment, 0.0)
    with torch.no_grad():
        value = (((roots + columns.a) * roots + columns.b) * roots + columns.c) * roots + columns.d
        genuine = genuine & (value.abs() <= 32 * eps * _quartic_value_scale(columns, roots.abs()))
    roots = torch.where(genuine, roots * variable_scale[:, None], 0.0)
    order = torch.where(genuine, roots, -torch.inf).argsort(dim=-1, descending=True, stable=True)
    roots = roots.gather(1, order)
    if torch.compiler.is_compiling():
        lower_coeffs = torch.where(
            lower.to(device=coeffs.device)[:, None], coeffs[:, 1:], torch.zeros_like(coeffs[:, 1:])
        )
        lower_roots = _solve_cubic(lower_coeffs)[0].to(device=work.device).to(dtype=work.dtype)
        padded = torch.cat([lower_roots, torch.zeros_like(lower_roots[:, :1])], -1)
        roots = torch.where(lower[:, None], padded, roots)
    elif bool(lower.any()):
        selected = lower.to(device=coeffs.device)
        lower_roots = _solve_cubic(coeffs[selected, 1:])[0].to(device=work.device).to(dtype=work.dtype)
        roots[lower] = torch.cat([lower_roots, torch.zeros_like(lower_roots[:, :1])], -1)

    return roots.to(dtype=original_dtype).to(device=coeffs.device)


# Reference
# https://github.com/danini/graph-cut-ransac/blob/master/src/pygcransac/include/
# estimators/solver_essential_matrix_five_point_nister.h#L108


T_deg1 = torch.zeros(16, 10)
T_deg1[0, 0] = 1  # x * x → x^2
T_deg1[1, 1] = 1  # x * y
T_deg1[4, 1] = 1  # y * x
T_deg1[2, 2] = 1  # x * z
T_deg1[8, 2] = 1  # z * x
T_deg1[3, 3] = 1  # x * 1
T_deg1[12, 3] = 1  # 1 * x
T_deg1[5, 4] = 1  # y * y
T_deg1[6, 5] = 1  # y * z
T_deg1[9, 5] = 1  # z * y
T_deg1[7, 6] = 1  # y * 1
T_deg1[13, 6] = 1  # 1 * y
T_deg1[10, 7] = 1  # z * z
T_deg1[11, 8] = 1  # z * 1
T_deg1[14, 8] = 1  # 1 * z
T_deg1[15, 9] = 1  # 1 * 1


def multiply_deg_one_poly(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Multiply two polynomials of the first order [@nister2004efficient].

    Args:
        a: a first order polynomial for variables :math:`(x,y,z,1)`.
        b: a first order polynomial for variables :math:`(x,y,z,1)`.

    Returns:
        degree 2 poly with the order :math:`(x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1)`.

    """
    global T_deg1  # noqa: PLW0603
    if T_deg1.device != a.device or T_deg1.dtype != a.dtype:
        T_deg1 = T_deg1.to(device=a.device, dtype=a.dtype)
    return (a.unsqueeze(2) * b.unsqueeze(1)).flatten(start_dim=-2) @ T_deg1


# Reference
# https://github.com/danini/graph-cut-ransac/blob/aae1f40c2e10e31fd2191bac601c53a189673f60/src/pygcransac/
# include/estimators/solver_essential_matrix_five_point_nister.h#L156

T_deg2 = torch.zeros(40, 20)
T_deg2[0, 0] = 1  # (0*4+0)
T_deg2[17, 1] = 1  # (4*4+1)
T_deg2[1, 2] = 1  # (0*4+1)
T_deg2[4, 2] = 1  # (1*4+0)
T_deg2[5, 3] = 1  # (1*4+1)
T_deg2[16, 3] = 1  # (4*4+0)
T_deg2[2, 4] = 1  # (0*4+2)
T_deg2[8, 4] = 1  # (2*4+0)
T_deg2[3, 5] = 1  # (0*4+3)
T_deg2[12, 5] = 1  # (3*4+0)
T_deg2[18, 6] = 1  # (4*4+2)
T_deg2[21, 6] = 1  # (5*4+1)
T_deg2[19, 7] = 1  # (4*4+3)
T_deg2[25, 7] = 1  # (6*4+1)
T_deg2[6, 8] = 1  # (1*4+2)
T_deg2[9, 8] = 1  # (2*4+1)
T_deg2[20, 8] = 1  # (5*4+0)
T_deg2[7, 9] = 1  # (1*4+3)
T_deg2[13, 9] = 1  # (3*4+1)
T_deg2[24, 9] = 1  # (6*4+0)
T_deg2[10, 10] = 1  # (2*4+2)
T_deg2[28, 10] = 1  # (7*4+0)
T_deg2[11, 11] = 1  # (2*4+3)
T_deg2[14, 11] = 1  # (3*4+2)
T_deg2[32, 11] = 1  # (8*4+0)
T_deg2[15, 12] = 1  # (3*4+3)
T_deg2[36, 12] = 1  # (9*4+0)
T_deg2[22, 13] = 1  # (5*4+2)
T_deg2[29, 13] = 1  # (7*4+1)
T_deg2[23, 14] = 1  # (5*4+3)
T_deg2[26, 14] = 1  # (6*4+2)
T_deg2[33, 14] = 1  # (8*4+1)
T_deg2[27, 15] = 1  # (6*4+3)
T_deg2[37, 15] = 1  # (9*4+1)
T_deg2[30, 16] = 1  # (7*4+2)
T_deg2[31, 17] = 1  # (7*4+3)
T_deg2[34, 17] = 1  # (8*4+2)
T_deg2[35, 18] = 1  # (8*4+3)
T_deg2[38, 18] = 1  # (9*4+2)
T_deg2[39, 19] = 1  # (9*4+3)


def multiply_deg_two_one_poly(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""Multiply two polynomials a and b of degrees two and one [@nister2004efficient].

    Args:
        a: a second degree poly for variables :math:`(x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1)`.
        b: a first degree poly for variables :math:`(x y z 1)`.

    Returns:
        a third degree poly for variables,
        :math:`(x^3, y^3, x^2*y, x*y^2, x^2*z, x^2, y^2*z, y^2,
        x*y*z, x*y, x*z^2, x*z, x, y*z^2, y*z, y, z^3, z^2, z, 1)`.

    """
    global T_deg2  # noqa: PLW0603
    if T_deg2.device != a.device or T_deg2.dtype != a.dtype:
        T_deg2 = T_deg2.to(device=a.device, dtype=a.dtype)
    product_basis = a.unsqueeze(2) * b.unsqueeze(1)
    product_vector = product_basis.flatten(start_dim=-2)
    return product_vector @ T_deg2


# Compute degree 10 poly representing determinant (equation 14 in the paper)
# https://github.com/danini/graph-cut-ransac/blob/aae1f40c2e10e31fd2191bac601c53a189673f60/src/pygcransac/
# include/estimators/solver_essential_matrix_five_point_nister.h#L368C5-L368C82

multiplication_indices = torch.tensor(
    [
        [12, 16, 33],
        [12, 20, 29],
        [3, 33, 25],
        [7, 29, 25],
        [3, 20, 38],
        [7, 16, 38],
        [11, 16, 33],
        [11, 20, 29],
        [12, 15, 33],
        [12, 16, 32],
        [12, 19, 29],
        [12, 20, 28],
        [2, 33, 25],
        [3, 32, 25],
        [3, 33, 24],
        [6, 29, 25],
        [7, 28, 25],
        [7, 29, 24],
        [2, 20, 38],
        [3, 19, 38],
        [3, 20, 37],
        [6, 16, 38],
        [7, 15, 38],
        [7, 16, 37],
        [10, 16, 33],
        [10, 20, 29],
        [11, 15, 33],
        [11, 16, 32],
        [11, 19, 29],
        [11, 20, 28],
        [14, 12, 33],
        [12, 15, 32],
        [12, 16, 31],
        [12, 18, 29],
        [12, 19, 28],
        [12, 20, 27],
        [1, 33, 25],
        [2, 32, 25],
        [2, 33, 24],
        [3, 31, 25],
        [3, 32, 24],
        [3, 33, 23],
        [5, 29, 25],
        [6, 28, 25],
        [6, 29, 24],
        [7, 27, 25],
        [7, 28, 24],
        [7, 29, 23],
        [1, 20, 38],
        [2, 19, 38],
        [2, 20, 37],
        [3, 18, 38],
        [3, 19, 37],
        [3, 20, 36],
        [5, 16, 38],
        [6, 15, 38],
        [6, 16, 37],
        [7, 14, 38],
        [7, 15, 37],
        [7, 16, 36],
        [3, 20, 35],
        [3, 22, 33],
        [7, 16, 35],
        [7, 22, 29],
        [9, 16, 33],
        [9, 20, 29],
        [10, 15, 33],
        [10, 16, 32],
        [10, 19, 29],
        [10, 20, 28],
        [13, 12, 33],
        [11, 14, 33],
        [11, 15, 32],
        [11, 16, 31],
        [11, 18, 29],
        [11, 19, 28],
        [11, 20, 27],
        [14, 12, 32],
        [12, 15, 31],
        [12, 16, 30],
        [12, 17, 29],
        [12, 18, 28],
        [12, 19, 27],
        [12, 20, 26],
        [0, 33, 25],
        [1, 32, 25],
        [1, 33, 24],
        [2, 31, 25],
        [2, 32, 24],
        [2, 33, 23],
        [3, 30, 25],
        [3, 31, 24],
        [3, 32, 23],
        [4, 29, 25],
        [5, 28, 25],
        [5, 29, 24],
        [6, 27, 25],
        [6, 28, 24],
        [6, 29, 23],
        [7, 26, 25],
        [7, 27, 24],
        [7, 28, 23],
        [0, 20, 38],
        [1, 19, 38],
        [1, 20, 37],
        [2, 18, 38],
        [2, 19, 37],
        [2, 20, 36],
        [3, 17, 38],
        [3, 18, 37],
        [3, 19, 36],
        [4, 16, 38],
        [5, 15, 38],
        [5, 16, 37],
        [6, 14, 38],
        [6, 15, 37],
        [6, 16, 36],
        [7, 13, 38],
        [7, 14, 37],
        [7, 15, 36],
        [2, 20, 35],
        [2, 22, 33],
        [3, 19, 35],
        [3, 20, 34],
        [3, 21, 33],
        [3, 22, 32],
        [6, 16, 35],
        [6, 22, 29],
        [7, 15, 35],
        [7, 16, 34],
        [7, 21, 29],
        [7, 22, 28],
        [8, 16, 33],
        [8, 20, 29],
        [9, 15, 33],
        [9, 16, 32],
        [9, 19, 29],
        [9, 20, 28],
        [10, 14, 33],
        [10, 15, 32],
        [10, 16, 31],
        [10, 18, 29],
        [10, 19, 28],
        [10, 20, 27],
        [13, 11, 33],
        [13, 12, 32],
        [11, 14, 32],
        [11, 15, 31],
        [11, 16, 30],
        [11, 17, 29],
        [11, 18, 28],
        [11, 19, 27],
        [11, 20, 26],
        [14, 12, 31],
        [12, 15, 30],
        [12, 17, 28],
        [12, 18, 27],
        [12, 19, 26],
        [0, 32, 25],
        [0, 33, 24],
        [1, 31, 25],
        [1, 32, 24],
        [1, 33, 23],
        [2, 30, 25],
        [2, 31, 24],
        [2, 32, 23],
        [3, 30, 24],
        [3, 31, 23],
        [4, 28, 25],
        [4, 29, 24],
        [5, 27, 25],
        [5, 28, 24],
        [5, 29, 23],
        [6, 26, 25],
        [6, 27, 24],
        [6, 28, 23],
        [7, 26, 24],
        [7, 27, 23],
        [0, 19, 38],
        [0, 20, 37],
        [1, 18, 38],
        [1, 19, 37],
        [1, 20, 36],
        [2, 17, 38],
        [2, 18, 37],
        [2, 19, 36],
        [3, 17, 37],
        [3, 18, 36],
        [4, 15, 38],
        [4, 16, 37],
        [5, 14, 38],
        [5, 15, 37],
        [5, 16, 36],
        [6, 13, 38],
        [6, 14, 37],
        [6, 15, 36],
        [7, 13, 37],
        [7, 14, 36],
        [1, 20, 35],
        [1, 22, 33],
        [2, 19, 35],
        [2, 20, 34],
        [2, 21, 33],
        [2, 22, 32],
        [3, 18, 35],
        [3, 19, 34],
        [3, 21, 32],
        [3, 22, 31],
        [5, 16, 35],
        [5, 22, 29],
        [6, 15, 35],
        [6, 16, 34],
        [6, 21, 29],
        [6, 22, 28],
        [7, 14, 35],
        [7, 15, 34],
        [7, 21, 28],
        [7, 22, 27],
        [8, 15, 33],
        [8, 16, 32],
        [8, 19, 29],
        [8, 20, 28],
        [9, 14, 33],
        [9, 15, 32],
        [9, 16, 31],
        [9, 18, 29],
        [9, 19, 28],
        [9, 20, 27],
        [10, 13, 33],
        [10, 14, 32],
        [10, 15, 31],
        [10, 16, 30],
        [10, 17, 29],
        [10, 18, 28],
        [10, 19, 27],
        [10, 20, 26],
        [13, 11, 32],
        [13, 12, 31],
        [11, 14, 31],
        [11, 15, 30],
        [11, 17, 28],
        [11, 18, 27],
        [11, 19, 26],
        [14, 12, 30],
        [12, 17, 27],
        [12, 18, 26],
        [0, 31, 25],
        [0, 32, 24],
        [0, 33, 23],
        [1, 30, 25],
        [1, 31, 24],
        [1, 32, 23],
        [2, 30, 24],
        [2, 31, 23],
        [3, 30, 23],
        [4, 27, 25],
        [4, 28, 24],
        [4, 29, 23],
        [5, 26, 25],
        [5, 27, 24],
        [5, 28, 23],
        [6, 26, 24],
        [6, 27, 23],
        [7, 26, 23],
        [0, 18, 38],
        [0, 19, 37],
        [0, 20, 36],
        [1, 17, 38],
        [1, 18, 37],
        [1, 19, 36],
        [2, 17, 37],
        [2, 18, 36],
        [3, 17, 36],
        [4, 14, 38],
        [4, 15, 37],
        [4, 16, 36],
        [5, 13, 38],
        [5, 14, 37],
        [5, 15, 36],
        [6, 13, 37],
        [6, 14, 36],
        [7, 13, 36],
        [0, 20, 35],
        [0, 22, 33],
        [1, 19, 35],
        [1, 20, 34],
        [1, 21, 33],
        [1, 22, 32],
        [2, 18, 35],
        [2, 19, 34],
        [2, 21, 32],
        [2, 22, 31],
        [3, 17, 35],
        [3, 18, 34],
        [3, 21, 31],
        [3, 22, 30],
        [4, 16, 35],
        [4, 22, 29],
        [5, 15, 35],
        [5, 16, 34],
        [5, 21, 29],
        [5, 22, 28],
        [6, 14, 35],
        [6, 15, 34],
        [6, 21, 28],
        [6, 22, 27],
        [7, 13, 35],
        [7, 14, 34],
        [7, 21, 27],
        [7, 22, 26],
        [8, 14, 33],
        [8, 15, 32],
        [8, 16, 31],
        [8, 18, 29],
        [8, 19, 28],
        [8, 20, 27],
        [9, 13, 33],
        [9, 14, 32],
        [9, 15, 31],
        [9, 16, 30],
        [9, 17, 29],
        [9, 18, 28],
        [9, 19, 27],
        [9, 20, 26],
        [10, 13, 32],
        [10, 14, 31],
        [10, 15, 30],
        [10, 17, 28],
        [10, 18, 27],
        [10, 19, 26],
        [13, 11, 31],
        [13, 12, 30],
        [11, 14, 30],
        [11, 17, 27],
        [11, 18, 26],
        [12, 17, 26],
        [0, 30, 25],
        [0, 31, 24],
        [0, 32, 23],
        [1, 30, 24],
        [1, 31, 23],
        [2, 30, 23],
        [4, 26, 25],
        [4, 27, 24],
        [4, 28, 23],
        [5, 26, 24],
        [5, 27, 23],
        [6, 26, 23],
        [0, 17, 38],
        [0, 18, 37],
        [0, 19, 36],
        [1, 17, 37],
        [1, 18, 36],
        [2, 17, 36],
        [4, 13, 38],
        [4, 14, 37],
        [4, 15, 36],
        [5, 13, 37],
        [5, 14, 36],
        [6, 13, 36],
        [0, 19, 35],
        [0, 20, 34],
        [0, 21, 33],
        [0, 22, 32],
        [1, 18, 35],
        [1, 19, 34],
        [1, 21, 32],
        [1, 22, 31],
        [2, 17, 35],
        [2, 18, 34],
        [2, 21, 31],
        [2, 22, 30],
        [3, 17, 34],
        [3, 21, 30],
        [4, 15, 35],
        [4, 16, 34],
        [4, 21, 29],
        [4, 22, 28],
        [5, 14, 35],
        [5, 15, 34],
        [5, 21, 28],
        [5, 22, 27],
        [6, 13, 35],
        [6, 14, 34],
        [6, 21, 27],
        [6, 22, 26],
        [7, 13, 34],
        [7, 21, 26],
        [8, 13, 33],
        [8, 14, 32],
        [8, 15, 31],
        [8, 16, 30],
        [8, 17, 29],
        [8, 18, 28],
        [8, 19, 27],
        [8, 20, 26],
        [9, 13, 32],
        [9, 14, 31],
        [9, 15, 30],
        [9, 17, 28],
        [9, 18, 27],
        [9, 19, 26],
        [10, 13, 31],
        [10, 14, 30],
        [10, 17, 27],
        [10, 18, 26],
        [13, 11, 30],
        [11, 17, 26],
        [0, 30, 24],
        [0, 31, 23],
        [1, 30, 23],
        [4, 26, 24],
        [4, 27, 23],
        [5, 26, 23],
        [0, 17, 37],
        [0, 18, 36],
        [1, 17, 36],
        [4, 13, 37],
        [4, 14, 36],
        [5, 13, 36],
        [0, 18, 35],
        [0, 19, 34],
        [0, 21, 32],
        [0, 22, 31],
        [1, 17, 35],
        [1, 18, 34],
        [1, 21, 31],
        [1, 22, 30],
        [2, 17, 34],
        [2, 21, 30],
        [4, 14, 35],
        [4, 15, 34],
        [4, 21, 28],
        [4, 22, 27],
        [5, 13, 35],
        [5, 14, 34],
        [5, 21, 27],
        [5, 22, 26],
        [6, 13, 34],
        [6, 21, 26],
        [8, 13, 32],
        [8, 14, 31],
        [8, 15, 30],
        [8, 17, 28],
        [8, 18, 27],
        [8, 19, 26],
        [9, 13, 31],
        [9, 14, 30],
        [9, 17, 27],
        [9, 18, 26],
        [10, 13, 30],
        [10, 17, 26],
        [0, 30, 23],
        [4, 26, 23],
        [0, 17, 36],
        [4, 13, 36],
        [0, 17, 35],
        [0, 18, 34],
        [0, 21, 31],
        [0, 22, 30],
        [1, 17, 34],
        [1, 21, 30],
        [4, 13, 35],
        [4, 14, 34],
        [4, 21, 27],
        [4, 22, 26],
        [5, 13, 34],
        [5, 21, 26],
        [8, 13, 31],
        [8, 14, 30],
        [8, 17, 27],
        [8, 18, 26],
        [9, 13, 30],
        [9, 17, 26],
        [0, 17, 34],
        [0, 21, 30],
        [4, 13, 34],
        [4, 21, 26],
        [8, 13, 30],
        [8, 17, 26],
    ],
    dtype=torch.int64,
)


signs = torch.tensor(
    [
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
        1.0,
        1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        1.0,
        1.0,
        -1.0,
    ],
    dtype=torch.float32,
)


coefficient_map = torch.tensor(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        4,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        5,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        6,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        10,
        10,
        10,
        10,
        10,
        10,
    ],
    dtype=torch.int64,
)


def determinant_to_polynomial(
    A: torch.Tensor,
) -> torch.Tensor:
    r"""Represent the determinant by the 10th polynomial, used for 5PC solver [@nister2004efficient].

    Convention:
        - Each row of ``A`` holds two cubics (columns 0 to 3 and 4 to 7) and a quartic (columns 8 to 12) in
          ``z``, highest degree first. The returned coefficients are lowest degree first (``cs[i]`` multiplies
          ``z**i``), the reverse of the :func:`solve_quadratic` layout.

    Args:
        A: torch.Tensor :math:`(B, 3, 13)`.

    Returns:
        a degree 10 poly of shape :math:`(B, 11)`, representing determinant (Eqn. 14 in the paper).

    """
    B, device, dtype = A.shape[0], A.device, A.dtype
    global multiplication_indices, signs, coefficient_map  # noqa: PLW0603

    multiplication_indices = multiplication_indices.to(device)
    signs = signs.to(device, dtype)
    coefficient_map = coefficient_map.to(device)

    A_flat = A.view(B, -1)
    gathered_values = A_flat[:, multiplication_indices]
    products = torch.prod(gathered_values, dim=-1)
    signed_products = products * signs

    cs = torch.zeros(B, 11, device=device, dtype=dtype)
    batch_coefficient_map = coefficient_map.repeat(B, 1)
    cs.scatter_add_(dim=1, index=batch_coefficient_map, src=signed_products)
    return cs
