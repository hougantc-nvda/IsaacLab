# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVXR renderer: lease, owner transfer, scene publication and camera panels.

The native OVRTX and OVStage endpoints are doubled; the renderer, its lease and the clone
preparation are real.
"""

import contextlib
import importlib.util
import inspect
import types
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg, SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
    pytest.mark.ov,
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers import OVRTXBackendCfg  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import (  # noqa: E402
        OVRTXBackend,
        OVRTXCameraRenderData,
        OVRTXRenderer,
        _AsyncWriteBuffers,
    )
    from isaaclab_teleop import ovxr_gpu_publication  # noqa: E402
    from isaaclab_teleop import ovxr_renderer as ovxr_renderer_module  # noqa: E402
    from isaaclab_teleop.ovxr_gpu_publication import _GeometryPublicationBuffers  # noqa: E402
    from isaaclab_teleop.ovxr_renderer import OvxrRenderer, OvxrRendererCfg, OvxrScenePublisher  # noqa: E402

_SPAWN = PinholeCameraCfg(
    focal_length=24.0,
    focus_distance=400.0,
    horizontal_aperture=20.955,
    clipping_range=(0.1, 1.0e5),
)


def _make_camera_cfg(data_types: list[str]) -> CameraCfg:
    return CameraCfg(
        height=8,
        width=16,
        prim_path="/World/envs/env_0/Camera",
        spawn=_SPAWN,
        data_types=data_types,
    )


def _make_ovrtx_camera_render_data() -> OVRTXCameraRenderData:
    spec = CameraRenderSpec(_make_camera_cfg(["rgb"]), "cpu", 2, ("/World/envs/env_0/Camera",), 2)
    return OVRTXCameraRenderData(spec, "RenderCamera_0", [f"/World/envs/env_{i}/Camera" for i in range(2)])


@pytest.fixture(autouse=True)
def _simulation_registry(monkeypatch):
    from pxr import Usd

    from isaaclab.renderers import RenderContext

    for name in ("ISAAC_LAB_OVRTX_USE_OVSTAGE", "ISAAC_LAB_SHARE_OVSTAGE"):
        monkeypatch.delenv(name, raising=False)
    registry = []
    sim = types.SimpleNamespace(
        device="cpu",
        has_gui=False,
        has_offscreen_render=False,
        visualizers=[],
        get_setting=lambda _key: False,
        vis_marker_registry=VisMarkerRegistry(),
        _backend_registry=registry,
        _render_context=RenderContext(registry),
        backend_manager=types.SimpleNamespace(clone_context_type=None, backend=None, _clone_recipes=[]),
        clone_contexts={},
        stage=Usd.Stage.CreateInMemory(),
        plan=None,
    )
    sim.render_context = sim._render_context
    sim.get_clone_plan = lambda: sim.plan
    sim.get_scene_data_provider = lambda: types.SimpleNamespace(
        backend=types.SimpleNamespace(transform_paths=[]), get_geometry_points=lambda: {}
    )
    sim.get_backends = SimulationContext.get_backends.__get__(sim)
    sim.get_or_create_backend = SimulationContext.get_or_create_backend.__get__(sim)
    sim.close_backend = SimulationContext.close_backend.__get__(sim)
    monkeypatch.setattr(SimulationContext, "_instance", sim)


@pytest.fixture
def leased_renderer(monkeypatch):
    """Exercise real deferred resource acquisition with native SDK endpoints doubled."""
    import isaaclab_ov.stage as stage_module

    monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    monkeypatch.setattr(OVRTXBackend, "_initial_gpu_selection", None)
    native, stage, paths = MagicMock(), MagicMock(), MagicMock()
    floor = stage.get_attribute_write_floor.return_value.__enter__.return_value
    floor.fetch.return_value = 0

    def advance(*, ordinal, **kwargs):
        return types.SimpleNamespace(wait=lambda: setattr(floor.fetch, "return_value", ordinal))

    stage.advance_write_floor.side_effect = advance
    monkeypatch.setattr(ovrtx_renderer_module, "Renderer", lambda _cfg: native)
    monkeypatch.setattr(ovrtx_renderer_module, "redirect_shader_cache", lambda _cfg: None)
    monkeypatch.setattr(stage_module, "create_ovstage", lambda _name: contextlib.nullcontext(stage))
    monkeypatch.setattr(stage_module.ovstage, "PathDictionary", lambda _: contextlib.nullcontext(paths))
    monkeypatch.setattr(stage_module.ovstage.population, "open_usd_from_string", lambda *args, **kwargs: None)
    renderer = SimulationContext.instance().get_or_create_backend(OvxrRendererCfg())
    assert renderer.backend is renderer.scene is None
    return renderer


def _prepare_leased_renderer(renderer, *, bind=True):
    """Prepare clones and, with ``bind``, the Stage on a render owner that claimed the transfer."""
    from isaaclab_ov.cloner import OvrtxReplicateContext

    OvrtxReplicateContext.prepare(renderer._sim, {})
    if bind:
        lease = renderer._render_lease.active
        if lease is not None:
            lease.claim_owner_transfer(lease.offer_owner_transfer())
        renderer.prepare_stage(renderer._sim.stage, 1)


def test_render_lease_borrows_registry_pair_and_blocks_shared_users(leased_renderer, monkeypatch):
    renderer = leased_renderer
    shared = SimulationContext.instance().get_or_create_backend(renderer.cfg)
    assert shared is renderer
    lease = renderer.acquire_render_lease()
    with pytest.raises(RuntimeError, match="already active"):
        shared.acquire_render_lease()
    _prepare_leased_renderer(renderer)
    with pytest.raises(RuntimeError, match="Lease"):
        shared.render_batch((object(),))
    shared.render_batch(())
    with monkeypatch.context() as patch:
        patch.setattr(ovrtx_renderer_module, "OVRTXMesh", lambda *_: pytest.fail("constructed a mesh while leased"))
        with pytest.raises(RuntimeError, match="log runtime meshes"):
            shared.log_mesh("blocked")
    assert "blocked" not in shared.meshes
    native = lease.ovrtx_renderer
    assert native is shared.backend.renderer
    assert lease.ovstage is shared.scene.stage
    # A failed transfer drain must retain native-owner guards through registry cleanup.
    publication = _GeometryPublicationBuffers.__new__(_GeometryPublicationBuffers)
    source, stream = object(), object()
    publication._sources = [source]
    publication._source_streams = [stream]
    publication._stream = None
    publication._buffers = {}
    renderer._geometry_publication = publication
    with monkeypatch.context() as patch:
        patch.setattr(wp, "synchronize_stream", MagicMock(side_effect=RuntimeError("producer drain failed")))
        with pytest.raises(RuntimeError, match="producer drain failed"):
            lease.release()
        assert publication._sources == [source]
        assert renderer._sim.vis_marker_registry.has_consumers
        with pytest.raises(RuntimeError, match="another owner borrows it"):
            SimulationContext.instance().close_backend(renderer.backend)
        with pytest.raises(RuntimeError, match="another owner borrows it"):
            SimulationContext.instance().close_backend(renderer.scene)
        native.destroy.assert_not_called()
    monkeypatch.setattr(wp, "synchronize_stream", lambda _stream: None)
    lease.release()
    lease.release()
    assert not renderer._sim.vis_marker_registry.has_consumers
    native.destroy.assert_not_called()
    SimulationContext.instance().close_backend(renderer.backend)
    native.destroy.assert_called_once_with()


# Every public OVRTXRenderer operation has a declared behavior while the session holds the lease, so a
# new public operation fails the coverage test until OvxrRenderer guards it or it is listed here.
_LEASE_REJECTED_OPERATIONS = {
    "create_render_data": lambda renderer: renderer.create_render_data(
        CameraRenderSpec(_make_camera_cfg(["rgb"]), "cpu", 2, ("/World/envs/env_0/Camera",), 2)
    ),
    "update_camera": lambda renderer: renderer.update_camera(object(), None, None, None),
    "update_camera_intrinsics": lambda renderer: renderer.update_camera_intrinsics(object(), None, None),
    "log_mesh": lambda renderer: renderer.log_mesh("blocked"),
    "render_batch": lambda renderer: renderer.render_batch((object(),)),
    # Releasing a product removes it from the borrowed Stage.
    "cleanup": lambda renderer: renderer.cleanup(object()),
    "close": lambda renderer: renderer.close(),
}
# The session's scene publisher owns these writes; ordinary calls return without writing.
_LEASE_DEFERRED_OPERATIONS = ("update_transforms", "update_geometries")
# OvxrRenderer adapts these to the lease instead of rejecting them.
_LEASE_ADAPTED_OPERATIONS = ("prepare_stage", "visual_material_writer")
# These read renderer metadata, act only on render data that the rejected ``create_render_data``
# produces, delegate to a rejected operation (``render``), or are the base no-op ``initialize``.
_LEASE_NEUTRAL_OPERATIONS = (
    "supported_output_types",
    "set_outputs",
    "prepare_capture",
    "read_output",
    "drain_pending_renders",
    "reset",
    "render",
    "initialize",
)


def test_every_public_ovrtx_operation_has_a_lease_policy(leased_renderer, monkeypatch):
    from isaaclab_teleop.ovxr_renderer import OvxrRenderer

    public = {name for name, _ in inspect.getmembers(OVRTXRenderer) if not name.startswith("_")}
    declared = [
        *_LEASE_REJECTED_OPERATIONS,
        *_LEASE_DEFERRED_OPERATIONS,
        *_LEASE_ADAPTED_OPERATIONS,
        *_LEASE_NEUTRAL_OPERATIONS,
    ]
    assert len(declared) == len(set(declared))
    assert public == set(declared)
    for name in (*_LEASE_REJECTED_OPERATIONS, *_LEASE_DEFERRED_OPERATIONS, *_LEASE_ADAPTED_OPERATIONS):
        assert name in vars(OvxrRenderer), f"OvxrRenderer does not override {name}"

    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    _prepare_leased_renderer(renderer)
    for name, operation in _LEASE_REJECTED_OPERATIONS.items():
        with pytest.raises(RuntimeError, match="while an OvxrRenderLease is active"):
            operation(renderer)
    for name in _LEASE_DEFERRED_OPERATIONS:
        monkeypatch.setattr(OVRTXRenderer, name, lambda _self, name=name: pytest.fail(f"leased {name} wrote"))
        getattr(renderer, name)()
    lease.release()


def test_render_lease_rejects_acquisition_after_clone_preparation(leased_renderer):
    renderer = leased_renderer
    _prepare_leased_renderer(renderer, bind=False)
    with pytest.raises(RuntimeError, match="before clone preparation"):
        renderer.acquire_render_lease()
    assert renderer._render_lease.active is None


def test_leased_stage_preparation_requires_an_unused_private_stage(leased_renderer):
    """The claimed render owner prepares only the fresh private OVStage that clone preparation created."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    _prepare_leased_renderer(renderer, bind=False)
    lease.claim_owner_transfer(lease.offer_owner_transfer())
    floor = renderer.scene.stage.get_attribute_write_floor.return_value.__enter__.return_value
    floor.fetch.return_value = 1
    with pytest.raises(RuntimeError, match="unused private OVStage"):
        renderer.prepare_stage(renderer._sim.stage, 1)
    assert not renderer._initialized_scene
    floor.fetch.return_value = 0
    renderer.prepare_stage(renderer._sim.stage, 1)
    lease.validate_live()
    lease.release()


def test_render_lease_rejects_calibration_before_scene_access(leased_renderer, monkeypatch):
    """An ordinary calibration write cannot bypass the external owner's publication lane."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    camera = _make_ovrtx_camera_render_data()
    intrinsics = wp.zeros(2, dtype=wp.mat33f, device="cpu")
    parameters = wp.zeros((5, 2), dtype=wp.float32, device="cpu")
    drain = MagicMock(return_value=[])
    monkeypatch.setattr(renderer, "drain_pending_renders", drain)
    monkeypatch.setattr(wp, "get_stream", lambda device: types.SimpleNamespace(cuda_stream=99))
    try:
        with pytest.raises(RuntimeError, match="Lease"):
            renderer.update_camera_intrinsics(camera, intrinsics, parameters)
        drain.assert_not_called()
        assert renderer.backend is renderer.scene is None
    finally:
        lease.release()


def test_render_lease_owner_transfer_is_single_use(leased_renderer):
    import threading

    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    with pytest.raises(RuntimeError, match="capability is invalid"):
        lease.claim_owner_transfer(None)
    token = lease.offer_owner_transfer()
    errors = []

    def owner():
        try:
            lease.claim_owner_transfer(token)
            with pytest.raises(RuntimeError, match="capability is invalid"):
                lease.claim_owner_transfer(token)
            lease.release()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=owner)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert renderer._render_lease._owner_thread is thread
    assert renderer.backend is renderer.scene is None


class _Ledger:
    """Stage ordinal ledger double that reserves one fixed ordinal and records seals and abandons."""

    def __init__(self, ordinal: int):
        self.ordinal = ordinal
        self.reserved, self.published, self.abandoned = [], [], []

    def reserve(self) -> int:
        self.reserved.append(self.ordinal)
        return self.ordinal

    def publish(self, ordinal: int) -> None:
        self.published.append(ordinal)

    def abandon(self, ordinal: int) -> None:
        self.abandoned.append(ordinal)


@pytest.fixture
def complete_scene_publication(leased_renderer, monkeypatch):
    """Exercise the application publisher with native component endpoints doubled."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    publisher = lease.capture_scene_publisher()
    _prepare_leased_renderer(renderer)
    stage = renderer.scene.stage
    stage.get_attribute_write_floor.reset_mock()
    stage.advance_write_floor.reset_mock()
    state = types.SimpleNamespace(
        renderer=renderer,
        lease=lease,
        publisher=publisher,
        ordinal=23,
        events=[],
        failure=None,
        cause=ValueError("component endpoint failed"),
        initial_id=renderer._render_lease._last_step_id,
    )

    def component(name):
        state.events.append((name, renderer.scene.ordinal))
        if state.failure == name:
            raise state.cause

    writer = types.SimpleNamespace(
        has_pending_updates=False,
        publish=lambda: component("material"),
        drain=lambda: component("material-drain"),
    )
    monkeypatch.setattr(renderer, "_visual_material_writer_ref", lambda: writer)
    monkeypatch.setattr(renderer, "_publish_transforms", lambda: component("transform"))
    monkeypatch.setattr(renderer, "_publish_geometries", lambda: component("geometry"))
    monkeypatch.setattr(renderer._panels, "publish_poses", lambda: component("auxiliary"))
    state.ledger = _Ledger(state.ordinal)
    state.publish = lambda step_id="step", force=False: publisher.publish(step_id, state.ledger, force=force)
    try:
        yield state
    finally:
        lease.release()


def test_render_lease_complete_scene_components_share_the_supplied_ordinal(complete_scene_publication):
    state = complete_scene_publication
    renderer = state.renderer
    assert state.publisher.publication_required("step")
    assert state.publish() == state.ordinal
    assert state.ledger.published == [state.ordinal]
    names = ["material", "transform", "geometry", "auxiliary", "material-drain"]
    assert state.events == [(name, state.ordinal) for name in names]
    assert renderer._render_lease._last_step_id == "step"
    assert renderer.scene.ordinal == state.ordinal
    assert not state.publisher.publication_required("step")
    assert state.publish() is None
    assert state.ledger.reserved == [state.ordinal]
    assert state.events == [(name, state.ordinal) for name in names]
    renderer.scene.stage.get_attribute_write_floor.assert_not_called()
    renderer.scene.stage.advance_write_floor.assert_not_called()
    assert not renderer._render_lease._poisoned

    # An independently changed XR anchor still needs this same scene identity.
    state.events.clear()
    assert state.publisher.publish("step", _Ledger(state.ordinal + 1), force=True) == state.ordinal + 1
    assert state.events == [(name, state.ordinal + 1) for name in names]


def test_publication_prepares_only_a_scene_that_needs_publication(complete_scene_publication):
    state = complete_scene_publication
    prepared = []
    assert state.publisher.publish("step", state.ledger, prepare=lambda: prepared.append(1)) == state.ordinal
    assert state.publisher.publish("step", state.ledger, prepare=lambda: prepared.append(2)) is None
    assert prepared == [1]


def test_failed_seal_poisons_the_pair_and_abandons_its_ordinal(complete_scene_publication):
    """A failed seal is a failed publication, after every component was written."""
    state = complete_scene_publication

    def seal(_ordinal):
        raise RuntimeError("seal failed")

    with pytest.raises(RuntimeError, match="seal failed"):
        state.publisher.publish("step", state.ledger, seal=seal)
    assert [name for name, _ in state.events][-1] == "material-drain"
    assert state.ledger.abandoned == [state.ordinal] and not state.ledger.published
    assert state.renderer._render_lease._poisoned
    assert state.publisher.closed


def test_markers_require_publication_only_when_their_registry_revision_changes(complete_scene_publication):
    """Unchanged markers do not repeat a scene write; added, updated, hidden or removed markers do."""
    state = complete_scene_publication
    renderer = state.renderer
    registry = renderer._sim.vis_marker_registry
    registry.revision = 4
    renderer.marker_groups = (object(),)
    assert state.publisher.publication_required("step")
    state.publish()
    assert not state.publisher.publication_required("step")
    registry.revision += 1
    assert state.publisher.publication_required("step")
    state.publish()
    assert not state.publisher.publication_required("step")
    # Removing the last group still publishes once, so that the renderer retires its markers.
    renderer.marker_groups = ()
    renderer.markers = {"retired": object()}
    registry.revision += 1
    assert state.publisher.publication_required("step")


def test_missing_camera_sample_is_rejected_before_any_component_and_can_recover(complete_scene_publication):
    """A missing second camera cannot leave the first camera or geometry partially committed."""
    state = complete_scene_publication
    ready = types.SimpleNamespace(state=("sample",), render_data=types.SimpleNamespace(camera_xform_binding=object()))
    missing = types.SimpleNamespace(state=None, render_data=types.SimpleNamespace(camera_xform_binding=object()))
    state.renderer._panels.cameras = [ready, missing]
    original_ordinal = state.renderer.scene.ordinal
    with pytest.raises(RuntimeError, match="no live world-pose sample"):
        state.publish()
    assert not state.events
    assert not state.ledger.reserved
    assert state.renderer.scene.ordinal == original_ordinal
    assert not state.renderer._render_lease._poisoned
    assert not state.publisher.closed
    assert state.renderer._render_lease._last_step_id == state.initial_id
    missing.state = ("fresh sample",)
    assert state.publish() == state.ordinal
    assert state.renderer._render_lease._last_step_id == "step"
    assert state.events


@pytest.mark.parametrize("failure", ["material", "transform", "geometry", "auxiliary", "material-drain"])
def test_render_lease_complete_scene_failure_preserves_commit_boundary(complete_scene_publication, failure):
    state = complete_scene_publication
    renderer, lease = state.renderer, state.lease
    state.failure = failure
    with pytest.raises(RuntimeError, match="Producer scene publication failed") as caught:
        state.publish()
    assert caught.value.__cause__ is state.cause
    order = ["material", "transform", "geometry", "auxiliary", "material-drain"]
    expected = order[: order.index(failure) + 1]
    if failure != "material-drain":
        expected.append("material-drain")
    assert state.events == [(name, state.ordinal) for name in expected]
    assert state.ledger.abandoned == [state.ordinal] and not state.ledger.published
    assert renderer._render_lease._last_step_id == state.initial_id
    assert renderer.scene.ordinal == state.ordinal
    assert renderer._render_lease._poisoned
    assert state.publisher.closed
    with pytest.raises(RuntimeError, match="poisoned"):
        _ = lease.ovstage
    previous_events = list(state.events)
    with pytest.raises(RuntimeError, match="unavailable|poisoned|closed"):
        state.publish("next-step")
    assert state.events == previous_events
    renderer.scene.stage.advance_write_floor.assert_not_called()
    renderer.scene.stage.get_attribute_write_floor.assert_not_called()


def test_render_lease_complete_scene_public_authority_precedes_components(complete_scene_publication):
    import threading

    state = complete_scene_publication
    errors = []

    def foreign_thread():
        try:
            state.publish()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=foreign_thread)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], RuntimeError)
    assert "thread" in str(errors[0])
    with pytest.raises(TypeError, match="physics_step_id must be hashable"):
        state.publish([])
    state.publisher.close()
    with pytest.raises(RuntimeError, match="publisher is closed"):
        state.publish()
    assert not state.events
    state.renderer.scene.stage.get_attribute_write_floor.assert_not_called()
    state.renderer.scene.stage.advance_write_floor.assert_not_called()
    assert not state.renderer._render_lease._poisoned


def test_render_lease_closed_publisher_reports_no_pending_publication(leased_renderer, monkeypatch):
    """A closed publisher must not report a write that :meth:`~OvxrScenePublisher.publish` would then reject."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    publisher = lease.capture_scene_publisher()
    assert isinstance(publisher, OvxrScenePublisher)
    _prepare_leased_renderer(renderer)
    monkeypatch.setattr(renderer, "_publish_transforms", lambda: None)
    monkeypatch.setattr(renderer, "_publish_geometries", lambda: None)
    assert publisher.publication_required("step-1")
    publisher.close()
    publisher.close()
    with pytest.raises(RuntimeError, match="publisher is closed"):
        publisher.publication_required("step-1")
    with pytest.raises(RuntimeError, match="publisher is closed"):
        publisher.publish("step-1", _Ledger(18))
    lease.release()


@pytest.mark.parametrize("failure_owner", ["producer", "borrower"])
def test_render_lease_poison_blocks_publication_but_not_retirement(leased_renderer, monkeypatch, failure_owner):
    """A failed producer write must not prevent the borrower from retiring the pair (Capture's close order)."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    publisher = lease.capture_scene_publisher()
    _prepare_leased_renderer(renderer)
    monkeypatch.setattr(renderer, "_publish_transforms", MagicMock(side_effect=ValueError("write failed")))
    monkeypatch.setattr(renderer, "_publish_geometries", lambda: None)
    if failure_owner == "producer":
        ledger = _Ledger(19)
        with pytest.raises(RuntimeError, match="scene publication failed"):
            publisher.publish("step-1", ledger)
        assert ledger.abandoned == [19] and not ledger.published
    else:
        publisher.invalidate()
    with pytest.raises(RuntimeError, match="unavailable|closed"):
        publisher.publication_required("step-2")
    with pytest.raises(RuntimeError, match="poisoned"):
        lease.validate_live()
    # OvxrFramePump.close: close publication, destroy the OVXR context inside the owner lane, release the lease.
    publisher.close()
    with lease.ovstage_owner_lane():
        pass
    ordinal = renderer.scene.ordinal
    lease.release()
    assert renderer.scene.ordinal == ordinal  # The abandoned ordinal is not rebased onto a sealed floor.
    with pytest.raises(RuntimeError, match="poisoned this OVRTX/OVStage pair; recreate it"):
        renderer.render_batch((object(),))
    with pytest.raises(RuntimeError, match="recreate it"):
        renderer.update_transforms()
    renderer.close()


def test_render_lease_release_rebases_ordinal_and_restores_ordinary_use(leased_renderer):
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    _prepare_leased_renderer(renderer)
    floor = renderer.scene.stage.get_attribute_write_floor.return_value.__enter__.return_value
    floor.fetch.return_value = 41
    lease.release()
    assert renderer.scene.ordinal == 42
    renderer.update_transforms()
    renderer.update_geometries()


@pytest.mark.parametrize(
    ("variable", "value"), [("ISAAC_LAB_OVRTX_USE_OVSTAGE", "0"), ("ISAAC_LAB_SHARE_OVSTAGE", "1")]
)
def test_render_lease_rejects_explicit_stage_opt_out(leased_renderer, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(RuntimeError, match="OVSTAGE=1|private rendering OVStage"):
        leased_renderer.acquire_render_lease()


def test_render_lease_defers_application_thread_stage_until_owner_transfer(leased_renderer):
    """Camera initialization prepares every renderer on the application thread before XR transfers ownership."""
    import threading

    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    _prepare_leased_renderer(renderer, bind=False)
    context = renderer._sim.render_context
    context.ensure_prepare_stage(renderer._sim.stage, 1)
    assert not renderer._initialized_scene
    assert renderer._render_lease.stage_deferred
    # The deferred call must leave the pair eligible for the XR worker handoff.
    publisher = lease.capture_scene_publisher()
    token = lease.offer_owner_transfer()
    errors = []

    def owner():
        try:
            lease.claim_owner_transfer(token)
            assert lease.materialize_deferred_stage(renderer._sim.stage, 1)
            assert not lease.materialize_deferred_stage(renderer._sim.stage, 1)
            lease.validate_live()
            publisher.close()
            lease.release()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=owner)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert renderer._initialized_scene


def test_render_lease_eager_scene_updates_defer_to_the_publisher(leased_renderer, monkeypatch):
    """``lazy_sensor_update=False`` refreshes every renderer; the leased renderer's publisher owns those writes."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    _prepare_leased_renderer(renderer)
    transforms, geometries = MagicMock(), MagicMock()
    # Ordinary eager updates reach the base renderer's writes once the lease is released.
    monkeypatch.setattr(OVRTXRenderer, "update_transforms", transforms)
    monkeypatch.setattr(OVRTXRenderer, "update_geometries", geometries)
    renderer._sim.render_context.update_scene_state(1)
    transforms.assert_not_called()
    geometries.assert_not_called()
    lease.release()
    renderer._sim.render_context.update_scene_state(2)
    transforms.assert_called_once_with()
    geometries.assert_called_once_with()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA material buffers")
@pytest.mark.parametrize("deferred", [False, True])
def test_leased_material_updates_reach_the_reserved_scene(leased_renderer, monkeypatch, deferred):
    """Real material composition and the lease publisher must reach native writes, including before physics changes."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    publisher = lease.capture_scene_publisher()
    context = renderer._sim.render_context
    if deferred:
        cfg = _make_camera_cfg(["rgb"])
        cfg.prim_path = "/World/Camera"
        lease.configure_auxiliary_camera(cfg)
    # Real lifecycle: clone preparation, then environment reset finalizes material consumers,
    # then the XR borrower binds the Stage.
    _prepare_leased_renderer(renderer, bind=False)
    if deferred:
        # Camera initialization requests the Stage on the application thread; the lease defers it.
        renderer.prepare_stage(renderer._sim.stage, 1)
    material = types.SimpleNamespace(
        channels=("color",),
        is_per_env=False,
        num_instances=1,
        _material_paths=("/Looks/Feedback",),
        _shader_paths=("/Looks/Feedback/Shader",),
        _input_names={"color": "diffuseColor"},
        _offsets={},
        _values={"color": torch.tensor([[1.0, 0.0, 0.0]], device="cuda")},
    )
    context.register_visual_material(material)
    context.finalize_consumers([])
    assert renderer._visual_material_writer_ref is not None
    if deferred:
        spec = CameraRenderSpec(cfg, "cpu", 1, (cfg.prim_path,), 1)
        data = renderer.create_render_data(spec)
        monkeypatch.setattr(ovrtx_renderer_module.ovstage.population, "add_usd_reference_from_string", lambda *a: 1)
        monkeypatch.setattr(ovrtx_renderer_module.ovstage.population, "apply_usd_changes", lambda *a, **kw: None)
        token = lease.offer_owner_transfer()
        lease.claim_owner_transfer(token)
        assert lease.materialize_deferred_stage(renderer._sim.stage, 1)
        renderer.update_camera(
            data,
            wp.zeros(1, dtype=wp.vec3f, device="cpu"),
            wp.array([[0.0, 0.0, 0.0, 1.0]], dtype=wp.quatf, device="cpu"),
            None,
        )
    else:
        # The render owner claims the pair first, so the Stage is prepared without deferral.
        lease.claim_owner_transfer(lease.offer_owner_transfer())
        renderer.prepare_stage(renderer._sim.stage, 1)
    stage = renderer.scene.stage
    try:
        for ordinal, color in ((23, [0.0, 1.0, 0.0]), (24, [0.0, 0.0, 1.0])):
            context.write_visual_materials([material], {"color": torch.tensor([color], device="cuda")})
            assert publisher.publication_required("unchanged-physics")
            stage.write_attribute.reset_mock()
            completion = MagicMock()
            stage.write_attribute.return_value = completion
            assert publisher.publish("unchanged-physics", _Ledger(ordinal)) == ordinal
            writes = [c for c in stage.write_attribute.call_args_list if c.args[1] == "inputs:diffuseColor"]
            assert len(writes) == 1
            assert writes[0].kwargs["ordinal"] == ordinal
            torch.testing.assert_close(writes[0].kwargs["tensors"].cpu(), torch.tensor([color]))
            assert completion.wait.called
            assert not publisher.publication_required("unchanged-physics")
    finally:
        lease.release()
        for writer in context._visual_material_writers:
            writer.close()


def test_public_markers_reach_leased_renderer_without_an_ordinary_visualizer(leased_renderer, monkeypatch):
    """Public markers appear, move without a physics step, hide and retire through the leased publisher."""
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sim import SphereCfg

    renderer = leased_renderer
    registry = renderer._sim.vis_marker_registry
    markers = VisualizationMarkers(
        VisualizationMarkersCfg(prim_path="/Visuals/HandJoints", markers={"joint": SphereCfg(radius=0.005)})
    )
    markers.visualize(translations=[[0.0, 0.0, 1.0]])
    assert not registry.get_groups()  # No renderer or visualizer requested marker state yet.
    observed = []

    class NativeMarkers:
        def __init__(self, _renderer, _cfg):
            pass

        def update(self, translations, orientations, scales, indices, *, visible):
            observed.append((translations.tolist(), visible))

        def close(self):
            observed.append("retired")

    monkeypatch.setattr(ovrtx_renderer_module, "OVRTXMarkers", NativeMarkers)
    lease = renderer.acquire_render_lease()
    try:
        publisher = lease.capture_scene_publisher()
        _prepare_leased_renderer(renderer)
        ledger = _Ledger(23)
        markers.visualize(translations=[[0.0, 0.0, 1.0]])
        publisher.publish("unchanged-physics", ledger)
        assert observed[-1] == ([[0.0, 0.0, 1.0]], True)
        assert not publisher.publication_required("unchanged-physics")
        markers.visualize(translations=[[0.5, 0.0, 1.0]])
        assert publisher.publication_required("unchanged-physics")
        publisher.publish("unchanged-physics", ledger)
        assert observed[-1] == ([[0.5, 0.0, 1.0]], True)
        markers.set_visibility(False)
        publisher.publish("unchanged-physics", ledger)
        assert observed[-1] == ([[0.5, 0.0, 1.0]], False)
        del markers
        publisher.publish("unchanged-physics", ledger)
        assert observed[-1] == "retired"
    finally:
        lease.release()
    assert not registry.has_consumers
    unused = VisualizationMarkers(
        VisualizationMarkersCfg(prim_path="/Visuals/NoConsumer", markers={"joint": SphereCfg(radius=0.005)})
    )
    unused.visualize(translations=[[0.0, 0.0, 1.0]])
    assert not registry.get_groups()


def test_publication_runs_debug_marker_callbacks_without_an_ordinary_visualizer(leased_renderer, monkeypatch):
    """Debug-vis callbacks move their markers before each required write, as a visualizer tick would."""
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    from isaaclab.sim import SphereCfg

    renderer = leased_renderer
    registry = renderer._sim.vis_marker_registry
    assert not renderer._sim.visualizers
    observed = []

    class NativeMarkers:
        def __init__(self, _renderer, _cfg):
            pass

        def update(self, translations, orientations, scales, indices, *, visible):
            observed.append(translations.tolist())

        def close(self):
            pass

    class Goal:
        """A command term's debug visualization: markers follow state the environment step changes."""

        def __init__(self):
            self.position = [0.0, 0.0, 1.0]
            self.calls = 0
            self.markers = VisualizationMarkers(
                VisualizationMarkersCfg(prim_path="/Visuals/Goal", markers={"goal": SphereCfg(radius=0.01)})
            )

        def _debug_vis_callback(self, _event):
            self.calls += 1
            self.markers.visualize(translations=[self.position])

    monkeypatch.setattr(ovrtx_renderer_module, "OVRTXMarkers", NativeMarkers)
    goal = Goal()
    registry.add_debug_vis_callback(goal)
    lease = renderer.acquire_render_lease()
    try:
        publisher = lease.capture_scene_publisher()
        _prepare_leased_renderer(renderer)
        ledger = _Ledger(31)
        assert publisher.publish("step-1", ledger) is not None
        assert observed[-1] == [[0.0, 0.0, 1.0]]
        goal.position = [0.25, 0.0, 1.0]
        assert publisher.publish("step-2", ledger) is not None
        assert observed[-1] == [[0.25, 0.0, 1.0]]
        calls = goal.calls
        # A state that needs no write runs no callbacks, so the callbacks cannot force publications.
        assert publisher.publish("step-2", ledger) is None
        assert goal.calls == calls
    finally:
        lease.release()


def test_leased_camera_cleanup_waits_for_lease_release(leased_renderer, monkeypatch):
    """Releasing a leased camera would remove its product from the borrowed Stage; it waits for release."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    cfg = _make_camera_cfg(["rgb"])
    cfg.prim_path = "/World/Camera"
    lease.configure_auxiliary_camera(cfg)
    _prepare_leased_renderer(renderer, bind=False)
    renderer.prepare_stage(renderer._sim.stage, 1)
    data = renderer.create_render_data(CameraRenderSpec(cfg, "cpu", 1, (cfg.prim_path,), 1))
    population = ovrtx_renderer_module.ovstage.population
    removed = []
    monkeypatch.setattr(population, "add_usd_reference_from_string", lambda *args: 7)
    monkeypatch.setattr(population, "apply_usd_changes", lambda *args, **kwargs: None)
    monkeypatch.setattr(population, "remove_usd", lambda _stage, reference: removed.append(reference))
    lease.claim_owner_transfer(lease.offer_owner_transfer())
    assert lease.materialize_deferred_stage(renderer._sim.stage, 1)

    with pytest.raises(RuntimeError, match="release camera render data while an OvxrRenderLease is active"):
        renderer.cleanup(data)
    assert removed == []
    assert data in renderer._camera_render_data
    assert data.camera_xform_binding is not None

    lease.release()
    renderer.close()
    assert removed == [7]
    assert data not in renderer._camera_render_data


def test_external_camera_matches_the_actual_custom_clone_root(leased_renderer):
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    renderer.scene = types.SimpleNamespace(clone_env_paths=["/Lab/Cell0"])
    cfg = types.SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot/Head", width=640, height=480, data_types=("rgb",))
    lease.configure_auxiliary_camera(cfg)
    spec = types.SimpleNamespace(
        cfg=cfg, num_instances=1, view_count=1, camera_prim_paths=("/Lab/Cell0/Robot/Head/camera",)
    )
    assert renderer._panels.accept(spec)
    assert renderer._panels.cameras[0].spec is spec
    lease.release()


def test_only_a_matched_multi_view_camera_is_rejected_as_external(leased_renderer):
    """An unrelated tiled camera takes the ordinary path; a selected camera must have one view."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    renderer.scene = types.SimpleNamespace(clone_env_paths=["/World/envs/env_0"])
    cfg = types.SimpleNamespace(prim_path="{ENV_REGEX_NS}/Robot/Head", width=640, height=480, data_types=("rgb",))
    lease.configure_auxiliary_camera(cfg)

    def tiled(path):
        return types.SimpleNamespace(cfg=cfg, num_instances=2, view_count=2, camera_prim_paths=(path, path))

    try:
        assert renderer._panels.accept(tiled("/World/envs/env_0/Other")) is None
        with pytest.raises(RuntimeError, match="exactly one camera view"):
            renderer._panels.accept(tiled("/World/envs/env_0/Robot/Head"))
    finally:
        lease.release()


def test_selected_external_camera_order_is_independent_of_sensor_preparation(leased_renderer):
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    specs = []
    for name, hz in zip(("Head", "Left", "Right", "Overview"), (20, 10, 10, 5), strict=True):
        cfg = types.SimpleNamespace(prim_path=f"/World/{name}", width=640, height=480, data_types=("rgb",))
        lease.configure_auxiliary_camera(cfg, max_fps=hz)
        specs.append(types.SimpleNamespace(cfg=cfg, num_instances=1, view_count=1, camera_prim_paths=(cfg.prim_path,)))
    for spec in reversed(specs):
        assert renderer._panels.accept(spec)
    assert [camera.spec for camera in renderer._panels.cameras] == specs
    assert [camera.max_fps for camera in renderer._panels.cameras] == [20, 10, 10, 5]
    lease.release()


@pytest.mark.parametrize("max_fps", [None, 20.0])
def test_external_camera_schedules_preserve_export_composition(leased_renderer, tmp_path, max_fps):
    """Rates are authored as layer opinions; preserved references are not flattened into the export."""
    from pxr import Sdf, Usd

    asset = tmp_path / "robot.usda"
    asset.write_text('#usda 1.0\n(defaultPrim = "Robot")\ndef Xform "Robot" {\n def Camera "Head" {\n }\n}\n')
    usd = f'#usda 1.0\ndef Xform "World" {{\n def "Robot" (prepend references = @{asset}@) {{\n }}\n}}\n'
    camera = types.SimpleNamespace(max_fps=max_fps, spec=types.SimpleNamespace(camera_prim_paths=["/World/Robot/Head"]))
    leased_renderer._panels.cameras = [camera]
    authored = leased_renderer._panels._author_schedules(usd)
    if max_fps is None:
        assert authored is usd
        return
    layer = Sdf.Layer.CreateAnonymous()
    layer.ImportFromString(authored)
    assert layer.GetPrimAtPath("/World/Robot").referenceList.prependedItems[0].assetPath == str(asset)
    stage = Usd.Stage.Open(layer)
    head = stage.GetPrimAtPath("/World/Robot/Head")
    assert head.GetAttribute("omni:sensor:frameRate").Get() == (20.0, 1.0)
    assert head.GetAttribute("omni:sensor:Core:partialOutputs").Get() is False
    camera.spec.camera_prim_paths = ["/World/Robot/Missing"]
    with pytest.raises(RuntimeError, match="absent from the exported Stage"):
        leased_renderer._panels._author_schedules(usd)


def test_external_camera_schedules_reach_native_stage():
    """Leased population authors the selected cameras' native rates and retains only those properties."""
    import ovrtx
    import ovstage
    from isaaclab_ov.stage import OvstageBackend, OvstageBackendCfg

    ovrtx.register_schema_paths()
    camera_paths = [f"/World/envs/env_0/{name}" for name in ("Head", "Wrist", "Unselected")]
    # The application export has no camera schedules; the unselected camera's own rate must not populate.
    source = """#usda 1.0
def Xform "World" {
 def Xform "envs" {
  def Xform "env_0" {
   def Camera "Head" {
   }
   def Camera "Wrist" {
   }
   def Camera "Unselected" {
    double2 omni:sensor:frameRate = (999, 1)
   }
  }
 }
}
"""
    renderer = OvxrRenderer(OvxrRendererCfg(), sim=SimulationContext.instance())
    lease = renderer.acquire_render_lease()
    with contextlib.ExitStack() as resources:
        resources.callback(lease.release)
        for path, max_fps in zip(camera_paths[:2], (20.0, 10.0), strict=True):
            cfg = _make_camera_cfg(["rgb"])
            cfg.prim_path = path
            lease.configure_auxiliary_camera(cfg, max_fps=max_fps)
        for camera in renderer._panels.cameras:
            camera.spec = types.SimpleNamespace(camera_prim_paths=[camera.cfg.prim_path])
        renderer.scene = OvstageBackend(OvstageBackendCfg())
        resources.callback(renderer.scene.close)
        renderer._panels.populate(source)

        stage, paths = renderer.scene.stage, renderer.scene.paths
        names = ("omni:sensor:frameRate", "omni:sensor:Core:partialOutputs")
        resolved = {}
        plist = paths.create_path_list_from_strings(camera_paths)
        try:
            with stage.query_from_path_list(plist) as query:
                query.wait()
                with stage.read_attributes(
                    query,
                    [paths.intern_token(name) for name in names],
                    ovstage.OrdinalRange.latest(renderer.scene.ordinal - 1),
                ) as read:
                    read.wait()
                    for group in read.groups():
                        try:
                            values = np.asarray(group.array(0).copy()).reshape(group.data_count, -1)
                            group_paths = paths.get_path_strings(group.prim_list)
                            attribute = paths.token_to_string(group.attribute)
                            for index in range(min(group.prim_count, group.data_count)):
                                path = group_paths[group.prim_index(index)]
                                resolved[path, attribute] = values[group.data_row_index(index)].tolist()
                        finally:
                            stage.release_group(group)
        finally:
            paths.destroy_path_list(plist)
    assert resolved.get((camera_paths[0], names[0])) == [20.0, 1.0]
    assert resolved.get((camera_paths[1], names[0])) == [10.0, 1.0]
    assert resolved.get((camera_paths[0], names[1])) == [False]
    assert resolved.get((camera_paths[1], names[1])) == [False]
    assert (camera_paths[2], names[0]) not in resolved


# Split-GPU rendering


def _make_ovxr_renderer_without_backend(cfg: OvxrRendererCfg) -> OvxrRenderer:
    """An OVXR renderer with a registered placeholder backend and no lease, as before clone preparation."""
    renderer = OvxrRenderer(cfg, sim=SimulationContext.instance())
    renderer.backend = OVRTXBackend.__new__(OVRTXBackend)
    renderer.backend._retained_by = None
    renderer.scene = renderer.backend
    from isaaclab_ov.stage import OvstageBackend

    renderer.scene.commit = OvstageBackend.commit.__get__(renderer.scene)
    renderer.scene._commit = OvstageBackend._commit.__get__(renderer.scene)
    renderer.scene.query = OvstageBackend.query.__get__(renderer.scene)
    renderer.scene.next_camera_id = 0
    renderer.scene.clone_copies = []
    renderer.scene.clone_env_paths = ["/World/envs/env_0", "/World/envs/env_1"]
    renderer.backend.attached = False
    renderer.backend.render_products = set()
    backend_cfg = OVRTXBackendCfg(scene_key=renderer.cfg, use_ovstage=False, read_gpu_transforms=True, device="cpu")
    renderer.scene.stage = renderer.scene.paths = None
    renderer.backend.cfg = backend_cfg
    SimulationContext.instance()._backend_registry.append((backend_cfg, renderer.backend))
    return renderer


@pytest.mark.parametrize(
    ("active_cuda_gpus", "excluded"), [(None, False), ("0", False), ("0,1", False), ("1", True), ("1,2", True)]
)
def test_rendering_gpus_excluding_the_simulation_gpu_require_the_session_lease(active_cuda_gpus, excluded):
    """Only the leased pair's private OVStage stages simulation data for other GPUs; reject it unleased."""
    renderer = _make_ovxr_renderer_without_backend(OvxrRendererCfg(active_cuda_gpus=active_cuda_gpus))
    renderer._device = "cuda:0"
    assert renderer._native_cuda_gpus() == active_cuda_gpus
    if excluded:
        with pytest.raises(ValueError, match="require the OVXR session's lease"):
            renderer._requires_private_ovstage()
    else:
        assert renderer._requires_private_ovstage() is False


def test_split_gpu_renderer_renders_only_camera_panels(monkeypatch):
    """An ordinary camera's outputs would stay on a rendering GPU that is never copied back."""
    renderer = _make_ovxr_renderer_without_backend(OvxrRendererCfg())
    monkeypatch.setattr(renderer, "_excludes_simulation_gpu", lambda: True)
    cfg = _make_camera_cfg(["rgb"])
    with pytest.raises(RuntimeError, match="renders only camera panels"):
        renderer.create_render_data(CameraRenderSpec(cfg, renderer._device, 1, (cfg.prim_path,), 1))


@pytest.mark.parametrize(
    ("simulation", "active_cuda_gpus", "requested"),
    [("cpu", None, None), ("cuda:2", None, 2), ("cuda:2", "0,2", 2), ("cuda:2", "0", None)],
)
def test_panel_products_request_the_simulation_gpu_only_when_it_renders(simulation, active_cuda_gpus, requested):
    """OVRTX skips a product request outside the rendering set, so such products use automatic assignment."""
    renderer = _make_ovxr_renderer_without_backend(OvxrRendererCfg(active_cuda_gpus=active_cuda_gpus))
    renderer._device = simulation
    renderer._warp_device = types.SimpleNamespace(is_cuda=simulation != "cpu", ordinal=2 if simulation != "cpu" else -1)
    assert renderer._render_product_device_id() == requested


def test_geometry_publication_does_not_wait_on_the_host_before_stage_consumes_it(monkeypatch):
    """A batch without foreign sources does not block on the transfer stream of an earlier copy."""
    devices = test_devices(DeviceScope.CUDA)
    if not devices:
        pytest.skip("Geometry publication requires a CUDA device")
    device = devices[0]
    publisher = _GeometryPublicationBuffers(device)
    # The transfer stream exists once a foreign batch has been copied.
    publisher._stream = wp.Stream(device)
    syncs = []
    monkeypatch.setattr(wp, "synchronize_stream", syncs.append)
    writes = []
    stage = types.SimpleNamespace(
        write_attribute=lambda *_args, **kwargs: writes.append(kwargs) or types.SimpleNamespace(wait=lambda: None)
    )
    publisher.write(stage, object(), ["/World/Mesh"], [wp.zeros(3, dtype=wp.vec3, device=device)], ordinal=1)
    assert syncs == []
    # Stage still orders its copy-in after same-device producer work.
    assert writes[0]["cuda_stream"] == publisher._stream.cuda_stream


def test_geometry_publication_cuda_devices_resize_and_failed_consumption(monkeypatch):
    """Only foreign CUDA points move; replacement sources and failed consumption cannot corrupt a snapshot."""
    with pytest.raises(ValueError, match="explicit CUDA device"):
        _GeometryPublicationBuffers("cuda")
    devices = test_devices(DeviceScope.CUDA)
    if len(devices) < 2:
        pytest.skip("Cross-device geometry publication requires two selected CUDA devices")
    source_device, target_device = devices[0], devices[-1]
    publisher = _GeometryPublicationBuffers(target_device)
    inputs = [
        wp.array([[4, 5, 6]], dtype=wp.vec3, device=target_device),
        wp.array([[7, 8, 9], [10, 11, 12]], dtype=wp.vec3, device=source_device),
    ]
    observed = []
    wrap_tensor = ovxr_gpu_publication.points_tensor_from_warp
    empty_like = wp.empty_like
    allocation_streams = []

    def allocate(source, *, device):
        # Stream-ordered allocations must precede their first copy on the consuming stream.
        allocation_streams.append(wp.get_stream(device).cuda_stream)
        assert allocation_streams[-1] == publisher._stream.cuda_stream
        return empty_like(source, device=device)

    monkeypatch.setattr(wp, "empty_like", allocate)

    def observe(array):
        observed.append(array)
        return wrap_tensor(array)

    monkeypatch.setattr(ovxr_gpu_publication, "points_tensor_from_warp", observe)
    stage = MagicMock()
    paths = ["/Local", "/Foreign"]
    syncs = []
    synchronize = wp.synchronize_stream
    monkeypatch.setattr(wp, "synchronize_stream", lambda stream: syncs.append(stream) or synchronize(stream))
    try:
        publisher.write(stage, object(), paths, inputs, 1)
        # Stage reads after the copy on the transfer stream; no host wait on that stream precedes it.
        assert publisher._stream not in syncs
        assert observed[0] is inputs[0]
        assert str(observed[1].device) == target_device
        np.testing.assert_array_equal(observed[1].numpy(), [[7, 8, 9], [10, 11, 12]])
        observed.clear()
        # The provider may replace and resize its point storage while preserving the prim path.
        with wp.ScopedStream(wp.Stream(source_device)):
            inputs[1] = wp.array([[13, 14, 15]], dtype=wp.vec3, device=source_device)
            publisher.write(stage, object(), paths, inputs, 2)
            inputs[1].fill_(wp.vec3(-100))
        snapshot = observed[1]
        assert len(allocation_streams) == 2  # First use and replacement after resize.
        np.testing.assert_array_equal(snapshot.numpy(), [[13, 14, 15]])
        observed.clear()
        operation = stage.write_attribute.return_value
        operation.wait.side_effect = RuntimeError("Stage rejected the write")
        with pytest.raises(RuntimeError, match="Stage rejected"):
            publisher.write(stage, object(), paths, inputs, 3)
        waits = operation.wait.call_count
        with pytest.raises(RuntimeError, match="close and recreate"):
            publisher.write(stage, object(), paths, inputs, 4)
        publisher.close()
        assert operation.wait.call_count == waits  # Native wait already consumed the failed operation.
    finally:
        publisher.close()


@pytest.mark.parametrize(
    ("cuda_ordinal", "active_cuda_gpus", "use_ovstage", "expected"),
    [
        (0, None, True, None),
        (0, "0,1", True, None),
        (None, "0", True, None),
        (1, "0", False, None),
        (1, "0", True, "cuda:0"),
        (2, "1,0", True, "cuda:1"),
    ],
)
def test_geometry_is_staged_on_the_first_rendering_gpu_when_rendering_excludes_the_simulation_gpu(
    monkeypatch, cuda_ordinal, active_cuda_gpus, use_ovstage, expected
):
    """Only OVStage rendering away from the CUDA simulation GPU stages geometry; others publish in place."""

    class Reached(Exception):
        pass

    renderer = _make_ovxr_renderer_without_backend(OvxrRendererCfg(active_cuda_gpus=active_cuda_gpus))
    renderer._use_ovstage = use_ovstage
    renderer._device = "cpu" if cuda_ordinal is None else f"cuda:{cuda_ordinal}"

    class Device:
        is_cuda = cuda_ordinal is not None
        ordinal = -1 if cuda_ordinal is None else cuda_ordinal  # Warp reports ordinal -1 for the CPU.

    renderer._warp_device = Device()
    assert renderer._geometry_publication_device() == expected
    if expected is None:
        return

    def reach(device):
        assert device == expected
        raise Reached

    # Stage preparation creates the buffers first; native allocation is outside this check.
    monkeypatch.setattr(ovxr_renderer_module, "_GeometryPublicationBuffers", reach)
    with pytest.raises(Reached):
        renderer.prepare_stage(renderer._sim.stage, 1)
    assert renderer._geometry_publication is None


def test_rigid_publication_uses_target_device_and_retains_failed_producer(monkeypatch):
    """Rigid poses follow the selected GPU; failed drains keep the exact producer and scene alive."""
    devices = test_devices(DeviceScope.CUDA)
    if len(devices) < 2:
        pytest.skip("Cross-device rigid publication requires two selected CUDA devices")
    source_device, target_device = devices[0], devices[-1]
    renderer = _make_ovxr_renderer_without_backend(
        OvxrRendererCfg(active_cuda_gpus=target_device.removeprefix("cuda:"))
    )
    renderer._use_ovstage = True
    renderer._warp_device = wp.get_device(source_device)
    renderer._device = source_device
    renderer._object_xform_binding = MagicMock()
    publisher = renderer._geometry_publication = _GeometryPublicationBuffers(target_device)
    expected = np.array([[[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [2, 3, 4, 1]]], dtype=np.float64)
    producer = wp.Stream(source_device)
    with wp.ScopedStream(producer):
        matrices = wp.array(expected, dtype=wp.mat44d, device=source_device)
    renderer._transform_writes = _AsyncWriteBuffers([types.SimpleNamespace(matrices=matrices)])
    renderer._sdp = types.SimpleNamespace(
        backend=types.SimpleNamespace(transforms_timestamp=1), get_transforms=lambda *_a, **_kw: True
    )
    renderer.scene.stage = stage = MagicMock()
    renderer.scene.ordinal = 11
    observed = []
    convert = ovxr_gpu_publication.xform_tensor_from_warp

    def observe(array):
        observed.append(array)
        return convert(array)

    monkeypatch.setattr(ovxr_gpu_publication, "xform_tensor_from_warp", observe)
    try:
        with wp.ScopedStream(producer):
            renderer.update_transforms()
        assert str(observed[-1].device) == target_device
        np.testing.assert_array_equal(observed[-1].numpy(), expected)
        assert stage.write_attribute.call_args.args[1] == "omni:xform"
        assert stage.write_attribute.call_args.kwargs["is_array"] is False
        # Reset/changed state must reuse the target allocation while publishing fresh values.
        snapshot = observed[-1]
        renderer._sdp.backend.transforms_timestamp = 2
        with wp.ScopedStream(producer):
            matrices.fill_(wp.mat44d(0.0))
            renderer.update_transforms()
        assert observed[-1] is snapshot
        np.testing.assert_array_equal(snapshot.numpy(), np.zeros_like(expected))
        renderer._sdp.backend.transforms_timestamp = 3
        stage.write_attribute.return_value.wait.side_effect = RuntimeError("Rejected consumption")
        with wp.ScopedStream(producer), pytest.raises(RuntimeError, match="Rejected consumption"):
            renderer.update_transforms()
        # The device's current stream has changed. Drain the captured producer, not its replacement.
        original_sync = wp.synchronize_stream
        drained = []

        def failed_drain(stream):
            drained.append(stream)
            raise RuntimeError("Drain failed")

        with monkeypatch.context() as patch:
            patch.setattr(wp, "synchronize_stream", failed_drain)
            with pytest.raises(RuntimeError, match="Drain failed"):
                renderer.close()
        assert drained == [producer]
        assert renderer._geometry_publication is publisher
        assert publisher._sources == [matrices]
        assert renderer._object_xform_binding is not None
        original_sync(producer)
    finally:
        renderer.close()


@pytest.mark.parametrize("value", ["cuda:0", "0,,1"])
def test_renderer_cfg_rejects_malformed_gpu_selection(value):
    with pytest.raises(ValueError, match="must list CUDA indices"):
        OvxrRendererCfg(active_cuda_gpus=value)
