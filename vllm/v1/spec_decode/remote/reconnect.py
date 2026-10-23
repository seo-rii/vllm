# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded reconnect attempts for a session's fixed remote endpoint."""

import math
import time
from collections.abc import Callable


class ReconnectBackoff:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        *,
        initial_delay_s: float = 1.0,
        max_delay_s: float = 30.0,
    ) -> None:
        if (
            not math.isfinite(initial_delay_s)
            or not math.isfinite(max_delay_s)
            or not 0 < initial_delay_s <= max_delay_s
        ):
            raise ValueError("reconnect delays must be finite, positive and ordered")
        self._clock = clock
        self._initial_delay_s = initial_delay_s
        self._max_delay_s = max_delay_s
        self._delay_s = initial_delay_s
        self._retry_at = -math.inf

    def ready(self) -> bool:
        return self._clock() >= self._retry_at

    def failed(self) -> None:
        self._retry_at = self._clock() + self._delay_s
        self._delay_s = min(2 * self._delay_s, self._max_delay_s)

    def succeeded(self) -> None:
        self._retry_at = -math.inf
        self._delay_s = self._initial_delay_s
