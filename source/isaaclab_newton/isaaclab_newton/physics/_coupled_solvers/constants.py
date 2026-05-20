# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility constants for Newton coupled-solver backports."""

from __future__ import annotations

from newton._src.geometry import ParticleFlags
from newton._src.sim import BodyFlags

BODY_FLAG_PROXY = int(getattr(BodyFlags, "PROXY", 1 << 2))
"""Proxy-body bit used by Newton PR 2848.

Normal Newton releases before that PR do not expose ``BodyFlags.PROXY``. The
vendored coupled solvers keep using the same bit value so the local fallback can
run against those releases without mutating Newton's enum classes.
"""

PARTICLE_FLAG_ACTIVE = int(ParticleFlags.ACTIVE)
"""Active-particle bit used by Newton releases."""

PARTICLE_FLAG_PROXY = int(getattr(ParticleFlags, "PROXY", 1 << 1))
"""Proxy-particle bit used by Newton PR 2848."""
