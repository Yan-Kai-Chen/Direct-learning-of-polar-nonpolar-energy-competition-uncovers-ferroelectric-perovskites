"""Shared integer-image enumeration for periodic diffusion math."""

from __future__ import annotations

import itertools
from functools import lru_cache


@lru_cache(maxsize=None)
def integer_vectors(dimension: int, radius: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        value
        for value in itertools.product(range(-radius, radius + 1), repeat=dimension)
        if any(value)
    )


@lru_cache(maxsize=None)
def integer_shell(dimension: int, radius: int) -> tuple[tuple[int, ...], ...]:
    if radius < 1:
        raise ValueError("integer-shell radius must be positive")
    return tuple(
        value
        for value in itertools.product(range(-radius, radius + 1), repeat=dimension)
        if max(abs(component) for component in value) == radius
    )


__all__ = ["integer_shell", "integer_vectors"]
