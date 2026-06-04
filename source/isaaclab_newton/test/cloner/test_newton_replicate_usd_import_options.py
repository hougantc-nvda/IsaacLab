# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-registered USD import options during Newton replication."""

from __future__ import annotations

import newton
import torch
from isaaclab_newton.cloner.newton_replicate import newton_physics_replicate
from isaaclab_newton.physics import NewtonManager

from pxr import Gf, Usd, UsdGeom, UsdPhysics


def _replicate_stage(stage: Usd.Stage) -> tuple[newton.ModelBuilder, object]:
    """Replicate a single Isaac Lab environment stage into Newton."""
    return newton_physics_replicate(
        stage,
        ["/World/envs/env_0"],
        ["/World/envs/env_{}"],
        torch.tensor([0]),
        torch.tensor([[True]]),
        simplify_meshes=False,
    )


def _define_env_stage() -> Usd.Stage:
    """Create an in-memory stage with Isaac Lab's ``/World/envs/env_0`` layout."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")
    return stage


def test_newton_physics_replicate_honors_registered_static_visual_imports():
    """Replicated Newton prototypes honor registered env-relative USD import options."""
    from newton import ShapeFlags

    stage = _define_env_stage()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/PackingTable")
    cube = UsdGeom.Cube.Define(stage, "/World/envs/env_0/PackingTable/top")
    cube.CreateSizeAttr(1.0)

    try:
        NewtonManager.clear()
        NewtonManager.clear_usd_import_options()
        NewtonManager.register_usd_import_options("PackingTable", load_static_visual_shapes=True)

        builder, _ = _replicate_stage(stage)

        table_shapes = [
            (label, flags) for label, flags in zip(builder.shape_label, builder.shape_flags) if "PackingTable" in label
        ]
        assert [label for label, _flags in table_shapes] == ["/World/envs/env_0/PackingTable/top"]
        assert bool(table_shapes[0][1] & ShapeFlags.VISIBLE)
    finally:
        NewtonManager.clear()
        NewtonManager.clear_usd_import_options()


def test_newton_physics_replicate_honors_registered_fixed_imports():
    """Registered static prop imports keep collider bodies fixed in replicated Newton worlds."""
    from newton import JointType, ShapeFlags

    stage = _define_env_stage()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/PackingTable")
    body = UsdGeom.Xform.Define(stage, "/World/envs/env_0/PackingTable/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    cube = UsdGeom.Cube.Define(stage, "/World/envs/env_0/PackingTable/body/collider")
    cube.CreateSizeAttr(1.0)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    try:
        NewtonManager.clear()
        NewtonManager.clear_usd_import_options()
        NewtonManager.register_usd_import_options("PackingTable", floating=False)

        builder, _ = _replicate_stage(stage)

        assert builder.joint_type == [JointType.FIXED]
        assert [label for label in builder.shape_label if "PackingTable" in label] == [
            "/World/envs/env_0/PackingTable/body/collider"
        ]
        assert bool(builder.shape_flags[0] & ShapeFlags.COLLIDE_SHAPES)
    finally:
        NewtonManager.clear()
        NewtonManager.clear_usd_import_options()


def test_newton_physics_replicate_preserves_scaled_xform_collider_bounds():
    """Xform collider bounds are not double-scaled when approximated as boxes."""
    from newton import ShapeFlags

    stage = _define_env_stage()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/PackingTable")
    collider_xform = UsdGeom.Xform.Define(stage, "/World/envs/env_0/PackingTable/ScaledCollider")
    collider_xform.AddScaleOp().Set(Gf.Vec3f(0.01, 0.01, 0.01))
    UsdPhysics.CollisionAPI.Apply(collider_xform.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/envs/env_0/PackingTable/ScaledCollider/child")
    cube.CreateSizeAttr(200.0)

    try:
        NewtonManager.clear()
        NewtonManager.clear_usd_import_options()
        NewtonManager.register_usd_import_options("PackingTable", load_xform_collision_shapes=True)

        builder, _ = _replicate_stage(stage)

        assert builder.shape_label == ["/World/envs/env_0/PackingTable/ScaledCollider"]
        assert 0.9 < builder.shape_scale[0][0] < 1.1
        assert bool(builder.shape_flags[0] & ShapeFlags.COLLIDE_SHAPES)
    finally:
        NewtonManager.clear()
        NewtonManager.clear_usd_import_options()


def test_newton_manager_discovers_isaaclab_env_roots():
    """Verify live Newton imports see Isaac Lab's ``/World/envs`` layout."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/envs")
    UsdGeom.Xform.Define(stage, "/World/envs/env_1")
    UsdGeom.Xform.Define(stage, "/World/envs/env_0")

    assert NewtonManager._discover_env_paths(stage) == [
        (0, "/World/envs/env_0"),
        (1, "/World/envs/env_1"),
    ]
