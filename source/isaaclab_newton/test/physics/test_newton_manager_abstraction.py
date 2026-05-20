# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the per-solver :class:`NewtonManager` abstraction.

Covers:

* :attr:`NewtonSolverCfg.class_type` resolves to the matching manager subclass.
* :meth:`NewtonCfg.__post_init__` propagates ``solver_cfg.class_type`` onto
  :attr:`NewtonCfg.class_type` so that ``SimulationContext`` picks the right
  manager.
* Each leaf manager subclasses :class:`NewtonManager` and implements
  :meth:`_build_solver` (with the abstract base raising ``NotImplementedError``).
* The cross-config validation in :meth:`NewtonMJWarpManager._build_solver`
  rejects the ``MJWarp + use_mujoco_contacts=True + collision_cfg`` combination.
* Manager name dispatch (used by :class:`InteractiveScene` and the various
  factory dispatchers) still starts with ``"newton"``.
* End-to-end: spinning up a simulation with each solver builds the correct
  solver, sets the right ``_use_single_state`` / ``_needs_collision_pipeline``
  flags, and lands canonical state on :class:`NewtonManager` so that external
  ``NewtonManager._foo`` reads keep working.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab.managers import SceneEntityCfg
from isaaclab_newton.physics import (
    AdmmContactPairCfg,
    AdmmCouplingCfg,
    CoupledProxyCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
    FeatherstoneSolverCfg,
    KaminoSolverCfg,
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonCoupledManager,
    NewtonFeatherstoneManager,
    NewtonKaminoManager,
    NewtonManager,
    NewtonMJWarpManager,
    NewtonMPMManager,
    NewtonSolverCfg,
    NewtonXPBDManager,
    ProxyCouplingCfg,
    XPBDSolverCfg,
)
from newton.solvers import SolverFeatherstone, SolverImplicitMPM, SolverKamino, SolverMuJoCo, SolverXPBD

try:
    from newton.solvers.coupled_experimental import SolverAdmmCoupled, SolverCoupled, SolverProxyCoupled
except ImportError:
    SolverAdmmCoupled = None
    SolverCoupled = None
    SolverProxyCoupled = None

from isaaclab.sim import SimulationCfg, build_simulation_context

# ---------------------------------------------------------------------------
# Lightweight (no sim) parametrisation
# ---------------------------------------------------------------------------

# (solver_cfg_factory, expected_manager, expected_solver_cls,
#  expected_use_single_state, expected_needs_collision_pipeline)
SOLVER_MATRIX = [
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=True),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        False,
        id="mjwarp_internal_contacts",
    ),
    pytest.param(
        lambda: MJWarpSolverCfg(use_mujoco_contacts=False),
        NewtonMJWarpManager,
        SolverMuJoCo,
        True,
        True,
        id="mjwarp_newton_pipeline",
    ),
    pytest.param(
        lambda: XPBDSolverCfg(),
        NewtonXPBDManager,
        SolverXPBD,
        False,
        True,
        id="xpbd",
    ),
    pytest.param(
        lambda: FeatherstoneSolverCfg(),
        NewtonFeatherstoneManager,
        SolverFeatherstone,
        False,
        True,
        id="featherstone",
    ),
    pytest.param(
        lambda: KaminoSolverCfg(use_collision_detector=True),
        NewtonKaminoManager,
        SolverKamino,
        False,
        False,
        id="kamino_internal_contacts",
    ),
    pytest.param(
        lambda: KaminoSolverCfg(use_collision_detector=False),
        NewtonKaminoManager,
        SolverKamino,
        False,
        True,
        id="kamino_newton_pipeline",
    ),
    pytest.param(
        lambda: MPMSolverCfg(max_iterations=2, voxel_size=0.05),
        NewtonMPMManager,
        SolverImplicitMPM,
        True,
        False,
        id="implicit_mpm",
    ),
    pytest.param(
        lambda: CoupledSolverCfg(
            coupling_type="base",
            entries=[
                CoupledSolverEntryCfg(name="rigid", solver_cfg=XPBDSolverCfg(iterations=1), bodies=[0]),
                CoupledSolverEntryCfg(
                    name="particle",
                    solver_cfg=XPBDSolverCfg(iterations=1),
                    particles=[0],
                    in_place=True,
                ),
            ],
        ),
        NewtonCoupledManager,
        SolverCoupled,
        False,
        True,
        marks=pytest.mark.skipif(SolverCoupled is None, reason="Newton SolverCoupled is unavailable"),
        id="base_coupled_xpbd_body_particle",
    ),
    pytest.param(
        lambda: CoupledSolverCfg(
            entries=[
                CoupledSolverEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        use_mujoco_contacts=False,
                        njmax=100,
                        nconmax=100,
                        iterations=2,
                        ls_iterations=2,
                    ),
                    bodies=[0],
                    joints=[0],
                ),
                CoupledSolverEntryCfg(
                    name="sand",
                    solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.05),
                    particles=list(range(8)),
                ),
            ],
            proxy_coupling=ProxyCouplingCfg(
                proxies=[
                    CoupledProxyCfg(
                        source="rigid",
                        destination="sand",
                        bodies=[0],
                    )
                ],
            ),
        ),
        NewtonCoupledManager,
        SolverProxyCoupled,
        False,
        True,
        marks=pytest.mark.skipif(SolverProxyCoupled is None, reason="Newton SolverProxyCoupled is unavailable"),
        id="proxy_coupled_mjwarp_mpm",
    ),
    pytest.param(
        lambda: CoupledSolverCfg(
            coupling_type="admm",
            entries=[
                CoupledSolverEntryCfg(
                    name="rigid",
                    solver_cfg=XPBDSolverCfg(iterations=1),
                    bodies=[0],
                ),
                CoupledSolverEntryCfg(
                    name="particle",
                    solver_cfg=XPBDSolverCfg(iterations=1),
                    particles=[0],
                    in_place=True,
                ),
            ],
            admm_coupling=AdmmCouplingCfg(iterations=1, rho=1.0, gamma=0.0),
            use_collision_pipeline=False,
        ),
        NewtonCoupledManager,
        SolverAdmmCoupled,
        False,
        False,
        marks=pytest.mark.skipif(SolverAdmmCoupled is None, reason="Newton SolverAdmmCoupled is unavailable"),
        id="admm_coupled_xpbd_body_particle",
    ),
]


# ---------------------------------------------------------------------------
# class_type wiring (no SimulationContext required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_solver_cfg_class_type_resolves_to_subclass(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """Each ``*SolverCfg.class_type`` resolves to its matching manager subclass."""
    solver_cfg = solver_cfg_factory()
    # ``class_type`` is a lazy ``"module:Class"`` reference; calling its
    # ``_resolve()`` returns the actual class. ``__name__`` works without
    # forcing import (LazyType caches metadata) and is sufficient identity.
    assert solver_cfg.class_type.__name__ == expected_manager.__name__


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline",
    SOLVER_MATRIX,
)
def test_newton_cfg_post_init_propagates_class_type(
    solver_cfg_factory, expected_manager, _solver_cls, _single_state, _pipeline
):
    """``NewtonCfg.__post_init__`` lifts ``solver_cfg.class_type`` onto ``NewtonCfg.class_type``."""
    cfg = NewtonCfg(solver_cfg=solver_cfg_factory())
    assert cfg.class_type.__name__ == expected_manager.__name__


# ---------------------------------------------------------------------------
# Manager class hierarchy and factory contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "manager",
    [
        NewtonMJWarpManager,
        NewtonXPBDManager,
        NewtonFeatherstoneManager,
        NewtonKaminoManager,
        NewtonMPMManager,
        NewtonCoupledManager,
    ],
)
def test_subclass_of_newton_manager(manager):
    """All concrete managers inherit from :class:`NewtonManager`."""
    assert issubclass(manager, NewtonManager)
    # Subclasses must override the abstract factory.
    assert manager._build_solver is not NewtonManager._build_solver


def test_abstract_build_solver_raises():
    """Calling :meth:`_build_solver` on the abstract base raises."""
    with pytest.raises(NotImplementedError):
        NewtonManager._build_solver(model=None, solver_cfg=NewtonSolverCfg())


@pytest.mark.parametrize(
    "manager",
    [
        NewtonMJWarpManager,
        NewtonXPBDManager,
        NewtonFeatherstoneManager,
        NewtonKaminoManager,
        NewtonMPMManager,
        NewtonCoupledManager,
    ],
)
def test_manager_name_starts_with_newton(manager):
    """The ``"newton"`` prefix is required by :class:`InteractiveScene` and the
    various backend factories that dispatch on ``physics_manager.__name__.lower()``.
    """
    assert manager.__name__.lower().startswith("newton")


def test_coupled_entry_threads_generic_entry_options():
    """Isaac Lab entry cfg exposes Newton's generic SolverCoupled.Entry options."""

    def _configure_view(_view):
        return None

    entry = NewtonCoupledManager._build_entry(
        CoupledSolverEntryCfg(
            name="xpbd",
            solver_cfg=XPBDSolverCfg(iterations=1),
            particles=[0],
            configure_view=_configure_view,
            in_place=True,
        )
    )
    assert entry.configure_view is _configure_view
    assert callable(entry.solver)
    assert entry.in_place is True


def test_coupled_proxy_int_mode_is_normalized():
    """Integer proxy modes are normalized before constructing Newton proxy configs."""
    proxy = NewtonCoupledManager._build_proxy(
        CoupledProxyCfg(source="src", destination="dst", particles=[0], mode=1)
    )
    assert proxy.mode == "staggered"


def test_coupled_selectors_resolve_bodies_shapes_joints_particles():
    """Front-end selectors resolve to the raw ids Newton coupled solvers expect."""
    builder = NewtonManager.create_builder()
    base = builder.add_body(mass=1.0, label="/World/envs/env_0/Robot/base")
    finger = builder.add_body(mass=1.0, label="/World/envs/env_0/Robot/finger")
    joint = builder.add_joint_revolute(parent=base, child=finger, axis=(0, 0, 1))
    base_shape = builder.add_shape_box(base, hx=0.05, hy=0.05, hz=0.05)
    finger_shape = builder.add_shape_box(finger, hx=0.02, hy=0.02, hz=0.02)
    ground_shape = builder.add_ground_plane()
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.1), vel=wp.vec3(0.0), mass=0.1, radius=0.02)
    builder.add_particle(pos=wp.vec3(0.0, 0.0, 0.2), vel=wp.vec3(0.0), mass=0.1, radius=0.02)
    model = builder.finalize(device="cpu")

    scene_cfg = SimpleNamespace(robot=SimpleNamespace(prim_path="/World/envs/env_.*/Robot"))
    entry = NewtonCoupledManager._resolve_entry_cfg(
        model,
        CoupledSolverEntryCfg(
            name="rigid",
            solver_cfg=XPBDSolverCfg(iterations=1),
            body_entities=[SceneEntityCfg("robot")],
            particle_range=(0, None),
            include_static_shapes=True,
        ),
        scene_cfg,
    )
    assert entry.bodies == [base, finger]
    assert joint in entry.joints
    assert entry.shapes == [base_shape, finger_shape, ground_shape]
    assert entry.particles == [0, 1]

    proxy = NewtonCoupledManager._resolve_proxy_cfg(
        model,
        CoupledProxyCfg(
            source="rigid",
            destination="soft",
            body_entities=[SceneEntityCfg("robot", body_names=["finger"])],
            particle_range=(1, None),
        ),
        scene_cfg,
    )
    assert proxy.bodies == [finger]
    assert proxy.particles == [1]

    local_id_entry = NewtonCoupledManager._resolve_entry_cfg(
        model,
        CoupledSolverEntryCfg(
            name="finger",
            solver_cfg=XPBDSolverCfg(iterations=1),
            body_entities=[SceneEntityCfg("robot", body_ids=[1])],
        ),
        scene_cfg,
    )
    assert local_id_entry.bodies == [finger]


def test_coupled_scene_entity_selectors_require_scene_cfg():
    """SceneEntityCfg selectors fail early when the solver cfg has no scene cfg."""
    builder = NewtonManager.create_builder()
    builder.add_body(mass=1.0, label="/World/envs/env_0/Robot/base")
    model = builder.finalize(device="cpu")

    with pytest.raises(ValueError, match="scene_cfg"):
        NewtonCoupledManager._resolve_entry_cfg(
            model,
            CoupledSolverEntryCfg(name="rigid", solver_cfg=XPBDSolverCfg(), body_entities=[SceneEntityCfg("robot")]),
            None,
        )


@pytest.mark.parametrize(
    "cfg, match",
    [
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                ],
            ),
            "Duplicate",
        ),
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg(), in_place=True, substeps=2),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg()),
                ],
            ),
            "in_place requires substeps=1",
        ),
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg(), shapes=[0]),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg(), shapes=[0]),
                ],
            ),
            "shapes index 0 is owned by both",
        ),
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg()),
                ],
                proxy_coupling=ProxyCouplingCfg(
                    proxies=[CoupledProxyCfg(source="missing", destination="b", particles=[0])]
                ),
            ),
            "source 'missing'",
        ),
        (
            CoupledSolverCfg(
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg()),
                ],
                proxy_coupling=ProxyCouplingCfg(
                    proxies=[CoupledProxyCfg(source="a", destination="b", particles=[0], mode=2)]
                ),
            ),
            "Unsupported CoupledProxyCfg mode",
        ),
        (
            CoupledSolverCfg(
                coupling_type="admm",
                entries=[
                    CoupledSolverEntryCfg(name="a", solver_cfg=XPBDSolverCfg()),
                    CoupledSolverEntryCfg(name="b", solver_cfg=XPBDSolverCfg()),
                ],
                admm_coupling=AdmmCouplingCfg(contact_pairs=[AdmmContactPairCfg(source="a", destination="a")]),
            ),
            "source and destination",
        ),
    ],
)
def test_coupled_cfg_validation_rejects_invalid_configs(cfg, match):
    """Invalid coupled configs fail before Newton constructs sub-solvers."""
    with pytest.raises(ValueError, match=match):
        NewtonCoupledManager._validate_solver_cfg(cfg)


# ---------------------------------------------------------------------------
# End-to-end: build each solver via SimulationContext
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "solver_cfg_factory, expected_manager, expected_solver_cls,"
    " expected_use_single_state, expected_needs_collision_pipeline",
    SOLVER_MATRIX,
)
def test_initialize_solver_populates_canonical_state(
    solver_cfg_factory,
    expected_manager,
    expected_solver_cls,
    expected_use_single_state,
    expected_needs_collision_pipeline,
):
    """End-to-end: ``SimulationContext`` resolves the right manager subclass and
    ``initialize_solver`` lands the right solver + flags on :class:`NewtonManager`.

    External code reads :class:`NewtonManager` attributes directly (``_solver``,
    ``_use_single_state``, ``_needs_collision_pipeline``).  Even though dispatch
    runs through a leaf subclass (e.g. :class:`NewtonMJWarpManager`), shared
    state is assigned through the explicit base class so that those reads keep
    working regardless of which leaf is active.  This test is the regression
    guard for that contract.

    The builder is pre-populated directly (instead of relying on a USD stage)
    with either a minimal particle grid for MPM or a one-body / one-joint
    scene for rigid/articulation solvers:

    1. :class:`SolverImplicitMPM` requires particles and MPM custom attributes
       registered on the builder before particle creation.
    2. :class:`SolverMuJoCo` requires at least one joint to convert the model
       to MJCF; a ground-plane-only scene fails MJCF conversion.
    3. Pre-populating ``NewtonManager._builder`` causes
       :meth:`NewtonManager.start_simulation` to skip
       :meth:`instantiate_builder_from_stage`, so the test does not depend on
       USD asset packages.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(solver_cfg=solver_cfg_factory(), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        # Resolved manager class matches the expected leaf.
        resolved_manager = sim.physics_manager
        # ``physics_manager`` is a LazyType proxy — compare by ``__name__`` to
        # avoid forcing identity-by-id checks against the unresolved proxy.
        assert resolved_manager.__name__ == expected_manager.__name__
        assert resolved_manager.__name__.lower().startswith("newton")

        builder = NewtonManager.create_builder()
        if expected_solver_cls is SolverImplicitMPM:
            assert builder.has_custom_attribute("mpm:young_modulus")
            builder.add_particle_grid(
                pos=wp.vec3(-0.05, -0.05, 0.10),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=2,
                dim_y=2,
                dim_z=2,
                cell_x=0.05,
                cell_y=0.05,
                cell_z=0.05,
                mass=0.01,
                jitter=0.0,
                radius_mean=0.02,
            )
        elif SolverProxyCoupled is not None and expected_solver_cls is SolverProxyCoupled:
            assert builder.has_custom_attribute("mpm:young_modulus")
            body = builder.add_body(mass=1.0)
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
            builder.add_ground_plane()
            builder.add_particle_grid(
                pos=wp.vec3(-0.05, -0.05, 0.10),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                dim_x=2,
                dim_y=2,
                dim_z=2,
                cell_x=0.05,
                cell_y=0.05,
                cell_z=0.05,
                mass=0.01,
                jitter=0.0,
                radius_mean=0.02,
            )
        elif SolverCoupled is not None and expected_solver_cls is SolverCoupled:
            body = builder.add_body(mass=1.0)
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
            builder.add_particle(
                pos=wp.vec3(0.0, 0.0, 0.1),
                vel=wp.vec3(0.0),
                mass=0.1,
                radius=0.02,
            )
        elif SolverAdmmCoupled is not None and expected_solver_cls is SolverAdmmCoupled:
            assert builder.has_custom_attribute("coupling:body_particle_attachment_body")
            body = builder.add_body(mass=1.0)
            particle = builder.add_particle(
                pos=wp.vec3(0.0, 0.0, 0.0),
                vel=wp.vec3(0.0),
                mass=0.1,
                radius=0.02,
            )
            SolverAdmmCoupled.add_body_particle_attachment(builder, body, particle, stiffness=10.0)
        else:
            # Pre-populate the builder with a minimal scene so MJCF conversion has
            # something to work with.
            body = builder.add_body(mass=1.0)
            builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)

        # Force resolution and bring up the solver.
        sim.reset()

        # Canonical state lives on the base class.
        assert NewtonManager._solver is not None
        assert isinstance(NewtonManager._solver, expected_solver_cls)
        if SolverCoupled is not None and expected_solver_cls is SolverCoupled:
            assert NewtonCoupledManager.get_entry_solver("rigid") is not None
            assert NewtonCoupledManager.get_entry_solver("particle") is not None
        if SolverProxyCoupled is not None and expected_solver_cls is SolverProxyCoupled:
            assert NewtonCoupledManager.get_entry_solver("rigid") is not None
            assert NewtonCoupledManager.get_entry_solver("sand") is not None
        if SolverAdmmCoupled is not None and expected_solver_cls is SolverAdmmCoupled:
            assert NewtonCoupledManager.get_entry_solver("rigid") is not None
            assert NewtonCoupledManager.get_entry_solver("particle") is not None
        assert NewtonManager._use_single_state is expected_use_single_state
        assert NewtonManager._needs_collision_pipeline is expected_needs_collision_pipeline

        # ``_contacts`` is allocated whichever way contacts are handled
        # (MuJoCo internal buffer or Newton pipeline output).
        # Kamino with internal contacts does not currently set NewtonManager._contacts.
        if expected_needs_collision_pipeline and expected_solver_cls not in (SolverKamino, SolverImplicitMPM):
            assert NewtonManager._contacts is not None

        # One step should not raise — proves the dispatch wiring lines up
        # end-to-end.  (We do not assert physics; that's covered by the
        # asset/sensor test suites.)
        sim.step(render=False)


def test_mjwarp_internal_contacts_with_collision_cfg_raises():
    """Combining ``use_mujoco_contacts=True`` with a ``collision_cfg`` is rejected.

    The check lives in :meth:`NewtonMJWarpManager._build_solver` because it
    needs both the solver cfg subtype and the parent :class:`NewtonCfg`, so it
    fires during :meth:`NewtonManager.initialize_solver` (i.e. on
    ``sim.reset()``) rather than at cfg construction time.
    """
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True),
            collision_cfg=NewtonCollisionPipelineCfg(),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        builder = NewtonManager.create_builder()
        body = builder.add_body(mass=1.0)
        builder.add_joint_revolute(parent=-1, child=body, axis=(0, 0, 1))
        NewtonManager.set_builder(builder)

        with pytest.raises(ValueError, match="collision_cfg cannot be set"):
            sim.reset()
