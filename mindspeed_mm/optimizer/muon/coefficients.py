"""Coefficient schedules for Muon's Newton-Schulz iterations.

Each iteration applies ``X <- (a I + b A + c A^2) X`` with ``A = X X^T``. A
schedule is a sequence of ``(a, b, c)`` triples, one per iteration; if it is
shorter than ``ns_steps``, the last triple is repeated. The values are plain
Python floats, so choosing a triple never needs a device sync.
"""

import math


# Keller Jordan's original Muon coefficients, one triple for every iteration.
# https://kellerjordan.github.io/posts/muon/
KELLER_COEFFICIENTS = ((3.4445, -4.7750, 2.0315),)

# Jiacheng You's five-step schedule (preset name "you").
# https://x.com/YouJiacheng/status/1905861218138804534
YOU_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)

# Polar Express schedule, https://arxiv.org/pdf/2505.16932. As in the paper,
# a, b and c are divided by 1.05, 1.05^3 and 1.05^5 (the polynomial degree of
# each term), which keeps singular values slightly below 1 for robustness to
# rounding error.
_UNDAMPED_POLAR_EXPRESS_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)

POLAR_EXPRESS_SAFETY_FACTOR = 1.05

POLAR_EXPRESS_COEFFICIENTS = tuple(
    (
        a / POLAR_EXPRESS_SAFETY_FACTOR,
        b / POLAR_EXPRESS_SAFETY_FACTOR ** 3,
        c / POLAR_EXPRESS_SAFETY_FACTOR ** 5,
    )
    for a, b, c in _UNDAMPED_POLAR_EXPRESS_COEFFICIENTS
)

NS_COEFFICIENT_PRESETS = {
    "keller": KELLER_COEFFICIENTS,
    "you": YOU_COEFFICIENTS,
    "polar_express": POLAR_EXPRESS_COEFFICIENTS,
}

DEFAULT_NS_COEFFICIENTS = "polar_express"


def resolve_ns_coefficients(spec=None):
    """Normalize a preset name or explicit schedule into finite float triples."""
    if spec is None:
        spec = DEFAULT_NS_COEFFICIENTS
    if isinstance(spec, str):
        try:
            return NS_COEFFICIENT_PRESETS[spec]
        except KeyError:
            raise ValueError(
                f"unknown Newton-Schulz coefficient preset {spec!r}; "
                f"choose one of {sorted(NS_COEFFICIENT_PRESETS)} or pass an "
                "explicit sequence of (a, b, c) triples"
            ) from None

    try:
        schedule = tuple(tuple(float(value) for value in triple) for triple in spec)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Newton-Schulz coefficients must be a sequence of (a, b, c) triples"
        ) from error
    if not schedule:
        raise ValueError("Newton-Schulz coefficient schedule must not be empty")
    for triple in schedule:
        if len(triple) != 3:
            raise ValueError(
                f"each Newton-Schulz coefficient entry must be (a, b, c), got {triple}"
            )
        if not all(math.isfinite(value) for value in triple):
            raise ValueError(
                f"Newton-Schulz coefficients must be finite, got {triple}"
            )
    return schedule
