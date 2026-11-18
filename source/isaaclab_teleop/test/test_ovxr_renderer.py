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

import pytest
import torch
import warp as wp

from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import PinholeCameraCfg, SimulationContext

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
    from isaaclab_ov.renderers import ovrtx_renderer as ovrtx_renderer_module  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_renderer import (  # noqa: E402
        OVRTXCameraRenderData,
        OVRTXRenderer,
    )
    from isaaclab_teleop.ovxr_renderer import OvxrRendererCfg, OvxrScenePublisher  # noqa: E402

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
    with pytest.raises(RuntimeError, match="another owner borrows it"):
        SimulationContext.instance().close_backend(renderer.backend)
    with pytest.raises(RuntimeError, match="another owner borrows it"):
        renderer.scene.reset()
    native = lease.ovrtx_renderer
    assert native is shared.backend.renderer
    assert lease.ovstage is shared.scene.stage
    lease.release()
    lease.release()
    assert not renderer._sim.vis_marker_registry.has_consumers
    assert renderer.backend._retained_by is None
    assert renderer.scene._retained_by is None
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
    "cleanup",
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
    assert renderer.backend._retained_by is None


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
    names = ["material", "transform", "geometry", "material-drain"]
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


@pytest.mark.parametrize("failure", ["material", "transform", "geometry", "material-drain"])
def test_render_lease_complete_scene_failure_preserves_commit_boundary(complete_scene_publication, failure):
    state = complete_scene_publication
    renderer, lease = state.renderer, state.lease
    state.failure = failure
    with pytest.raises(RuntimeError, match="Producer scene publication failed") as caught:
        state.publish()
    assert caught.value.__cause__ is state.cause
    order = ["material", "transform", "geometry", "material-drain"]
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
    assert renderer.backend._retained_by is None
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
    with pytest.raises(RuntimeError, match="another owner borrows it"):
        renderer.scene.commit()
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
def test_leased_material_updates_reach_the_reserved_scene(leased_renderer, monkeypatch):
    """Real material composition and the lease publisher must reach native writes, including before physics changes."""
    renderer = leased_renderer
    lease = renderer.acquire_render_lease()
    publisher = lease.capture_scene_publisher()
    context = renderer._sim.render_context
    # Real lifecycle: clone preparation, then environment reset finalizes material consumers,
    # then the XR borrower binds the Stage.
    _prepare_leased_renderer(renderer, bind=False)
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
