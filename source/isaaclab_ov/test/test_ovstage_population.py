# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Population requirements from physics and cameras must compose in one native request."""

from types import SimpleNamespace

import pytest
from isaaclab_ov import stage as stage_module

pytestmark = [pytest.mark.kitless]


@pytest.mark.parametrize("physics", [False, True])
def test_camera_population_preserves_rendering_and_physics_requirements(monkeypatch, physics):
    calls = []
    camera_desc, physics_desc = object(), object()
    native = SimpleNamespace(
        population=SimpleNamespace(
            DEFAULT_TIME=0.0,
            Desc=lambda **kwargs: SimpleNamespace(**kwargs),
            open_usd_from_string_with_desc=lambda stage, usd, **kwargs: calls.append(kwargs),
        ),
    )
    monkeypatch.setattr(stage_module, "ovstage", native)
    monkeypatch.setattr(stage_module, "_physics_population_desc", lambda: physics_desc)
    stage_module.populate_stage(object(), "scene", ordinal=17, domains=1, physics=physics, population_desc=camera_desc)
    assert len(calls) == 1
    assert calls[0]["ordinal"] == 17
    descriptors = calls[0]["descs"]
    assert descriptors[0].domains == 1
    assert descriptors[-1] is camera_desc
    assert (physics_desc in descriptors) is physics
