# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVRTX renderer whose native OVRTX/OVStage pair is borrowed by the OVXR session.

``OvxrRenderer`` is the simulation's OVRTX renderer for an OVXR session: the same engine and
OVStage render the simulation's scene, but the session's render worker owns them. Its lease handles
the render-thread handoff, deferred Stage preparation, scene publication ordering and the selected
camera panels. Isaac Lab's ordinary OVRTX renderer stays available to other cameras.
"""

from __future__ import annotations

import contextlib
import math
import re
import threading
import weakref
from collections.abc import Callable, Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

import ovstage
import warp as wp
from isaaclab_ov.renderers import OVRTXBackendCfg, OVRTXRenderer, OVRTXRendererCfg
from isaaclab_ov.renderers.visual_materials import OVRTXVisualMaterialWriter
from isaaclab_ov.stage import OvstageBackendCfg

from pxr import Gf, Sdf, Usd

from isaaclab.renderers.camera_render_spec import CameraRenderSpec
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab_ov.renderers.ovrtx_renderer import OVRTXCameraRenderData
    from ovrtx import Renderer
    from ovstage import Stage as OvStage
_NO_PHYSICS_STEP_ID = object()


class _OvxrLeaseState(Enum):
    """Internal lifecycle state for a renderer/scene pair lease."""

    UNBOUND = auto()
    BOUND = auto()
    RELEASED = auto()


@dataclass
class _AuxiliaryCamera:
    cfg: Any
    max_fps: float | None
    spec: CameraRenderSpec | None = None
    render_data: OVRTXCameraRenderData | None = None
    state: tuple[Any, ...] | None = None
    product_path: str | None = None


class OvxrRenderLease:
    """Exclusive non-owning lease of the OVXR session over one attached OVRTX/OVStage pair.

    The session acquires the lease before environment construction. The renderer then defers
    Stage preparation until Isaac Capture's render worker claims the single-use owner transfer
    and materializes the Stage, which binds this object to the exact live Python ``Renderer``
    and ``Stage`` instances. The session releases the lease before the renderer is closed and
    never detaches or destroys either borrowed object.
    """

    _controller: _OvxrLeaseController
    """Renderer-owned controller that created this lease."""

    def __init__(self) -> None:
        raise RuntimeError("OvxrRenderLease must be acquired from OvxrRenderer")

    def __enter__(self) -> OvxrRenderLease:
        self._validate_active()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()

    @property
    def auxiliary_render_product_paths(self) -> tuple[str, ...]:
        """RenderProduct paths of the selected auxiliary cameras, in configuration order.

        Raises:
            RuntimeError: If the lease is not bound and live, or a selected camera did not
                materialize exactly one product.
        """
        self.validate_live()
        return self._controller.product_paths(self)

    @property
    def ovrtx_renderer(self) -> Renderer:
        """Return the exact retained application renderer."""
        self.validate_live()
        return cast("Renderer", self._renderer)

    @property
    def ovstage(self) -> OvStage:
        """Return the exact retained renderer-attached OVStage."""
        self.validate_live()
        return cast("OvStage", self._stage)

    def create_native_pair(self) -> None:
        """Create the session's own OVStage and the OVRTX engine that attaches to it.

        Call this after process-wide renderer settings are applied and before the environment is
        built. Clone preparation then finds the renderer initialized and only routes scene copies
        to the Stage. Calling it again on the same renderer is a no-op.

        Raises:
            RuntimeError: If called off the owner thread, or after clone preparation initialized
                the renderer.
        """
        self._controller.create_native_pair(self)

    def configure_auxiliary_camera(self, camera_cfg: Any, *, max_fps: float | None = None) -> None:
        """Authorize an exact camera before Stage materialization.

        The leased renderer then accepts the one camera sensor whose specification matches
        ``camera_cfg`` exactly: a single camera view with the same width, height and ordered
        ``data_types``, at ``prim_path`` (``{ENV_REGEX_NS}`` matches any environment) or its
        ``camera`` child prim. Every other camera on the leased renderer remains rejected.

        Args:
            camera_cfg: Configuration of the camera sensor to authorize, such as a
                :class:`~isaaclab.sensors.camera.CameraCfg`. Its ``prim_path`` must be unique among
                the lease's cameras.
            max_fps: Maximum native rate of the camera's frames [Hz], authored on its prim in the
                exported Stage. ``None`` authors no rate, so the camera renders with every frame of
                its render product.

        Raises:
            RuntimeError: If the lease is released or already bound to a Stage, or ``prim_path`` is
                already configured.
            ValueError: If ``camera_cfg`` is ``None``, or ``max_fps`` is not finite and positive.
        """
        self._validate_active()
        if self._state is not _OvxrLeaseState.UNBOUND:
            raise RuntimeError("Auxiliary camera configuration must precede Stage materialization")
        self._controller._renderer._panels.configure(camera_cfg, max_fps=max_fps)

    def offer_owner_transfer(self) -> object:
        """Freeze one unbound lease and return its one-shot owner capability."""
        owner = self._controller
        return owner.offer_owner_transfer(self)

    def claim_owner_transfer(self, capability: object) -> None:
        """Make the calling thread the render owner using the offered capability.

        Args:
            capability: The object returned by :meth:`offer_owner_transfer`.

        Raises:
            RuntimeError: If the capability is stale, already claimed, or the lease is no longer unbound.
        """
        self._controller.claim_owner_transfer(self, capability)

    def cancel_owner_transfer(self, capability: object) -> None:
        """Withdraw an unclaimed offer on the original owner thread.

        Args:
            capability: The object returned by :meth:`offer_owner_transfer`.

        Raises:
            RuntimeError: If called off the original owner thread or with a stale capability.
        """
        self._controller.cancel_owner_transfer(self, capability)

    def materialize_deferred_stage(self, stage: Any, num_envs: int) -> bool:
        """Complete a deferred Stage preparation on the claimed render owner.

        Args:
            stage: The environment USD stage.
            num_envs: Number of environments; the lease supports exactly one.

        Returns:
            Whether a deferred preparation was completed. ``False`` means nothing was deferred.

        Raises:
            RuntimeError: If called off the render-owner thread, or before the owner transfer is claimed.
        """
        return self._controller.materialize_deferred_stage(self, stage, num_envs)

    def capture_scene_publisher(self) -> OvxrScenePublisher:
        """Capture the application-thread scene publisher before render ownership transfers.

        Call this on the owner thread while the lease is unbound and before
        :meth:`offer_owner_transfer`. The publisher stays bound to the calling thread; see
        :class:`OvxrScenePublisher` for the publication
        protocol.

        Returns:
            The lease's only scene publisher.

        Raises:
            RuntimeError: If called off the owner thread or while a transfer is offered, after the
                Stage is bound, or when the lease already has a publisher.
        """
        return self._controller.capture_scene_publisher(self)

    @contextlib.contextmanager
    def ovstage_owner_lane(self) -> Iterator[None]:
        """Guard one non-reentrant operation on the render-owner thread.

        Isaac Capture's render worker runs its native frame operations here. The application's
        publication guard separately excludes scene publication while they use the borrowed stage.
        A failed publication poisons publication only: the lane stays available so the session can
        destroy its OVXR context and retire the pair.
        """
        self._validate_live(allow_poisoned=True)
        owner = self._controller
        with owner.owner_lane("render-worker operation"):
            yield

    def release(self) -> None:
        """Release the borrowing intent without destroying either pair member."""
        if self._state is _OvxrLeaseState.RELEASED:
            return
        owner = self._controller
        owner.release(self)
        self._renderer = None
        self._stage = None
        self._state = _OvxrLeaseState.RELEASED

    def validate_live(self) -> None:
        """Verify that this lease still refers to its live, unpoisoned renderer and stage."""
        self._validate_live(allow_poisoned=False)

    def _validate_live(self, *, allow_poisoned: bool) -> None:
        self._validate_active()
        if self._state is not _OvxrLeaseState.BOUND:
            raise RuntimeError("OvxrRenderLease is not bound to a prepared Stage")
        owner = self._controller
        owner.validate_live(
            self,
            cast("Renderer", self._renderer),
            cast("OvStage", self._stage),
            self._generation,
            allow_poisoned=allow_poisoned,
        )

    @classmethod
    def _create_unbound(cls, owner: _OvxrLeaseController) -> OvxrRenderLease:
        lease = object.__new__(cls)
        lease._controller = owner
        lease._state = _OvxrLeaseState.UNBOUND
        lease._renderer = None
        lease._stage = None
        lease._generation = 0
        return lease

    def _bind(self, renderer: Renderer, stage: OvStage, generation: int) -> None:
        if self._state is not _OvxrLeaseState.UNBOUND:
            raise RuntimeError("OvxrRenderLease can only bind once")
        self._renderer = renderer
        self._stage = stage
        self._generation = generation
        self._state = _OvxrLeaseState.BOUND

    def _validate_active(self) -> None:
        if self._state is _OvxrLeaseState.RELEASED:
            raise RuntimeError("OvxrRenderLease is released")
        owner = self._controller
        owner.validate_active(self)


class OvxrScenePublisher:
    """Application-thread publisher of complete provider-derived OVStage scenes.

    Obtain it with :meth:`OvxrRenderLease.capture_scene_publisher`. While the lease is active, the
    renderer skips its own eager transform, geometry and material writes, and the OVXR session publishes
    each complete physics state through :meth:`publish`, one transaction per scene:

    1. Skip a state that needs no publication (see :meth:`publication_required`).
    2. Run the simulation's visualization-marker callbacks, then the session's preparation, such as
       refreshing camera panel poses.
    3. Reject incomplete camera panel samples before reserving an ordinal.
    4. Reserve an ordinal, write every scene component at it, and seal it.
    5. If the write or seal fails, poison the pair, abandon the ordinal and re-raise.

    :meth:`~isaaclab.sim.SimulationContext.update_visualizers` runs the marker callbacks only for
    marker-capable visualizers, so without one, step 2 is what moves debug markers before they are drawn.

    Call :meth:`publish` under the application's publication guard, which the render owner also holds
    around every operation in :meth:`OvxrRenderLease.ovstage_owner_lane`, so that scene writes and
    native frame operations never interleave. A poisoned pair rejects later publication and ordinary
    renderer use, but the session can still retire its OVXR context and release the lease; the
    renderer must then be recreated. :meth:`close` stops publication; releasing the lease also closes
    the publisher. ``isaaclab_teleop.OvLibrariesXrSession`` publishes through this method.
    """

    def __init__(self, owner: _OvxrLeaseController, lease: OvxrRenderLease) -> None:
        self._controller = owner
        self._lease = lease
        self._application_thread = threading.current_thread()
        self._lock = threading.Lock()
        self._closed = False

    @property
    def closed(self) -> bool:
        """Whether publication stopped, by :meth:`close`, a failed publication or :meth:`invalidate`."""
        return self._closed

    def publication_required(self, physics_step_id: Hashable) -> bool:
        """Return whether a new physics step, changed markers or dirty materials require a scene write.

        Args:
            physics_step_id: Hashable identity of the complete physics state to publish. Only the
                last written identity is remembered, so the identity must change whenever the
                state does, including after a reset that does not advance the physics step count.

        Returns:
            Whether ``physics_step_id`` differs from the last written state, the drawn markers
            changed, or a visual material has pending updates.

        Raises:
            RuntimeError: If called off the captured application thread, the publisher is closed or
                inactive, the lease is not bound, or a failed write poisoned the pair.
            TypeError: If ``physics_step_id`` is not hashable.
        """
        self._require_application_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("OVStage scene publisher is closed")
            return self._controller.producer_publication_required(self._lease, physics_step_id)

    def publish(
        self,
        physics_step_id: Hashable,
        ordinals: Any,
        *,
        prepare: Callable[[], None] | None = None,
        seal: Callable[[int], None] | None = None,
        force: bool = False,
    ) -> int | None:
        """Publish one complete scene as a single transaction.

        Args:
            physics_step_id: Hashable identity of the complete physics state, as for
                :meth:`publication_required`.
            ordinals: Stage ordinal ledger with ``reserve()``, ``publish(ordinal)`` and
                ``abandon(ordinal)``, such as Isaac Capture's ``StageOrdinalLedger``.
            prepare: Called once a publication is required, after the marker callbacks and before camera
                panel samples are validated, for example to refresh their poses.
            seal: Seals the written ordinal. Defaults to ``ordinals.publish``.
            force: Publish even when the state needs no publication, for example for an
                independently changed XR anchor.

        Returns:
            The sealed ordinal, or ``None`` when the state needs no publication.

        Raises:
            RuntimeError: If called off the captured application thread, the publisher is closed or
                inactive, a camera panel lacks a live pose sample, or the write or seal failed. A
                failed write or seal poisons the pair, abandons the ordinal and closes the publisher;
                a rejected sample changes nothing.
        """
        self._require_application_thread()
        with self._lock:
            if self._closed:
                raise RuntimeError("OVStage scene publisher is closed")
            controller = self._controller
            if not force and not controller.producer_publication_required(self._lease, physics_step_id):
                return None
            # Callbacks only update marker state, so a failure here leaves the pair publishable.
            controller.dispatch_marker_callbacks()
            if prepare is not None:
                prepare()
            controller.require_publishable(self._lease)
            controller._renderer._panels.validate_samples()
            ordinal = ordinals.reserve()
            try:
                controller.write_scene_components(self._lease, ordinal, physics_step_id)
                (ordinals.publish if seal is None else seal)(ordinal)
            except BaseException as error:
                self._invalidate_locked()
                try:
                    ordinals.abandon(ordinal)
                except BaseException as abandon_error:
                    # Cleanup follows the failure; retain its original cause.
                    error.add_note(f"Abandon after failed scene publication also failed: {abandon_error!r}")
                raise
            return ordinal

    def invalidate(self) -> None:
        """Reject further use of a pair after the session's own write to it failed."""
        with self._lock:
            self._invalidate_locked()

    def close(self) -> None:
        """Stop scene publication after any in-flight scene finishes. Safe to call repeatedly."""
        with self._lock:
            self._closed = True

    def _invalidate_locked(self) -> None:
        self._closed = True
        if self._controller.active is self._lease:
            self._controller._poisoned = True

    def _require_application_thread(self) -> None:
        if threading.current_thread() is not self._application_thread:
            raise RuntimeError("OVStage scene publisher must run on its captured application thread")


class _OvxrLeaseController:
    """Coordinate the OVXR session's lease without owning the renderer's native resources.

    Lease state, thread transfer and publication ordering live here. The renderer
    retains scene/camera storage and supplies the component-write operations.
    """

    def __init__(self, renderer: OvxrRenderer) -> None:
        self._renderer = renderer
        self._owner_thread = threading.current_thread()
        self._transfer_lock = threading.Lock()
        self._transfer_capability: object | None = None
        self._transfer_claimed = False
        self.active: OvxrRenderLease | None = None
        self._publisher: OvxrScenePublisher | None = None
        self._generation = 0
        self._lane_operation: str | None = None
        self._last_step_id: Hashable = _NO_PHYSICS_STEP_ID
        self._published_marker_revision: int | None = None
        self._poisoned = False
        self.stage_deferred = False
        self._marker_consumer: object | None = None
        self._previous_marker_groups: Iterable = ()
        self._created_pair = False

    def acquire(self) -> OvxrRenderLease:
        """Reserve the renderer before stage preparation; :meth:`create_native_pair` creates its pair."""
        self.require_owner_thread("acquire a render lease")
        if self._renderer._closed:
            raise RuntimeError("Cannot acquire OvxrRenderLease from a closed renderer")
        if self.active is not None:
            raise RuntimeError("An OvxrRenderLease is already active")
        prepared = self._renderer.backend is not None and not self._created_pair
        if prepared or self._renderer._initialized_scene or self._generation != 0:
            raise RuntimeError("OvxrRenderLease must be acquired before clone preparation")
        lease = OvxrRenderLease._create_unbound(self)
        registry = self._renderer._sim.vis_marker_registry
        self._marker_consumer = registry.register_consumer()
        self._previous_marker_groups = self._renderer.marker_groups
        self._renderer.marker_groups = registry.get_groups().values()
        self._renderer._panels.clear()
        self.active = lease
        return lease

    def create_native_pair(self, lease: OvxrRenderLease) -> None:
        """Create the Stage and engine this lease lends, once, before clone preparation."""
        self.validate_active(lease)
        self.require_owner_thread("create the native pair")
        if self._created_pair:
            return
        if self._renderer.backend is not None or self._renderer._initialized_scene:
            raise RuntimeError("Clone preparation already initialized this renderer")
        # The session manages its own rendering Stage: create the Stage and the engine that attaches to
        # it. Clone preparation then finds the renderer initialized and only routes scene copies.
        renderer = self._renderer
        backend_cfg = OVRTXBackendCfg(
            scene_key=renderer.cfg,
            log_file_path=renderer.cfg.log_file_path,
            log_level=renderer.cfg.log_level,
            use_ovstage=True,
            read_gpu_transforms=True,
            device=renderer._sim.device,
        )
        # Register the Stage before the engine that borrows it.
        scene = renderer._sim.get_or_create_backend(OvstageBackendCfg(scene_key=backend_cfg))
        renderer.backend = renderer._sim.get_or_create_backend(backend_cfg)
        renderer.scene = scene
        renderer._use_ovstage = True
        self._created_pair = True

    def require_owner_thread(self, operation: str) -> None:
        """Reject lease operations outside the renderer construction thread."""
        if self._transfer_capability is not None:
            raise RuntimeError("OVRTXRenderer owner transfer is pending")
        if threading.current_thread() is not self._owner_thread:
            raise RuntimeError(f"OVRTXRenderer owner thread must {operation}")

    def offer_owner_transfer(self, lease: OvxrRenderLease) -> object:
        self.require_owner_thread("offer an owner transfer")
        with self._transfer_lock:
            if (
                self.active is not lease
                or lease._state is not _OvxrLeaseState.UNBOUND
                or self._renderer._initialized_scene
                or (self._renderer.backend is not None and self._renderer.backend.attached)
                or self._generation != 0
                or self._lane_operation is not None
                or self._transfer_capability is not None
                or self._transfer_claimed
            ):
                raise RuntimeError("OVRTXRenderer is not eligible for a pre-use owner transfer")
            capability = object()
            self._transfer_capability = capability
            return capability

    def claim_owner_transfer(self, lease: OvxrRenderLease, capability: object) -> None:
        with self._transfer_lock:
            if (
                capability is None
                or capability is not self._transfer_capability
                or self._transfer_claimed
                or self.active is not lease
                or lease._state is not _OvxrLeaseState.UNBOUND
                or self._renderer._initialized_scene
                or (self._renderer.backend is not None and self._renderer.backend.attached)
                or self._generation != 0
                or self._lane_operation is not None
            ):
                raise RuntimeError("OVRTXRenderer owner transfer capability is invalid")
            self._owner_thread = threading.current_thread()
            self._transfer_claimed = True
            self._transfer_capability = None

    def cancel_owner_transfer(self, lease: OvxrRenderLease, capability: object) -> None:
        with self._transfer_lock:
            # Cancellation is the one original-owner operation allowed while an offer is pending.
            if threading.current_thread() is not self._owner_thread:
                raise RuntimeError("OVRTXRenderer owner thread must cancel an owner transfer")
            if capability is None or capability is not self._transfer_capability or self.active is not lease:
                raise RuntimeError("OVRTXRenderer owner transfer capability is invalid")
            self._transfer_capability = None

    def capture_scene_publisher(self, lease: OvxrRenderLease) -> OvxrScenePublisher:
        """Create the scene publisher before render ownership transfers."""
        self.require_owner_thread("capture an OVStage scene publisher")
        if self.active is not lease or lease._state is not _OvxrLeaseState.UNBOUND or self._publisher is not None:
            raise RuntimeError("OVRTXRenderer is not eligible to capture an OVStage scene publisher")
        publisher = OvxrScenePublisher(self, lease)
        self._publisher = publisher
        return publisher

    @contextlib.contextmanager
    def owner_lane(self, operation: str) -> Iterator[None]:
        """Reject nested operations and access outside the render-owner thread.

        This lane is not a cross-thread lock. The OVXR session holds its application publication
        guard around the lane and around scene publication.
        """
        self.require_owner_thread(f"enter the OVStage owner lane for {operation}")
        if self._lane_operation is not None:
            raise RuntimeError(
                f"OVStage owner lane is already active for {self._lane_operation}; nested writers are forbidden"
            )
        self._lane_operation = operation
        try:
            yield
        finally:
            self._lane_operation = None

    def validate_active(self, lease: OvxrRenderLease) -> None:
        self.require_owner_thread("access a render lease")
        if self.active is not lease:
            raise RuntimeError("OvxrRenderLease is not active for this renderer")
        if self._renderer._closed:
            raise RuntimeError("OvxrRenderLease owner is closed or no longer uses OVStage")

    def validate_live(
        self,
        lease: OvxrRenderLease,
        ovrtx_renderer: Renderer,
        ovstage_instance: OvStage,
        generation: int,
        *,
        allow_poisoned: bool = False,
    ) -> None:
        """Validate retained Python-object identity for the live attached pair.

        ``allow_poisoned`` admits retirement work after a failed publication.
        """
        self.validate_active(lease)
        if self._poisoned and not allow_poisoned:
            raise RuntimeError("OvxrRenderLease owner pair is poisoned")
        if generation != self._generation:
            raise RuntimeError("OvxrRenderLease has a stale Stage generation")
        if ovrtx_renderer is not self._renderer.backend.renderer or ovstage_instance is not self._renderer.scene.stage:
            raise RuntimeError("OvxrRenderLease no longer matches its exact renderer-owned pair")
        if not self._renderer._initialized_scene or not self._renderer.backend.attached:
            raise RuntimeError("OvxrRenderLease pair is no longer initialized and attached")

    def release(self, lease: OvxrRenderLease) -> None:
        """Retire the lease after the OVXR context is destroyed and rebase the dormant ordinal."""
        self.validate_active(lease)
        if self._lane_operation is not None:
            raise RuntimeError("Cannot release OvxrRenderLease while the owner lane is active")
        if self._publisher is not None:
            self._publisher.close()
        if lease._state is _OvxrLeaseState.BOUND and (not self._poisoned):
            with self.owner_lane("lease-release ordinal rebase"):
                self._renderer.scene.ordinal = self._next_ordinal()
        self._last_step_id = _NO_PHYSICS_STEP_ID
        self._published_marker_revision = None
        self._publisher = None
        self._transfer_claimed = False
        self.stage_deferred = False
        self._renderer._sim.vis_marker_registry.unregister_consumer(self._marker_consumer)
        self._marker_consumer = None
        self._renderer.marker_groups = self._previous_marker_groups
        self._previous_marker_groups = ()
        self.active = None
        self._renderer._panels.clear()

    def reject_ordinary_operation(self, operation: str, *, retiring: bool = False) -> None:
        """Reject ordinary OVRTX operations while an external lease is active.

        A pair poisoned by a failed leased write keeps rejecting ordinary use after
        release, because its ordinal is not rebased; only ``retiring`` operations
        (closing the renderer) are admitted so the pair can be recreated.
        """
        if self.active is not None:
            raise RuntimeError(f"Cannot {operation} while an OvxrRenderLease is active")
        if self._poisoned and not retiring:
            raise RuntimeError(
                f"Cannot {operation}: a failed leased publication poisoned this OVRTX/OVStage pair; recreate it"
            )

    def require_publishable(self, lease: OvxrRenderLease) -> None:
        """Raise unless ``lease`` holds the active, bound and unpoisoned publisher pair."""
        if self.active is not lease or self._publisher is None:
            raise RuntimeError("OVStage scene publisher is not active for this lease")
        if self._poisoned or lease._state is not _OvxrLeaseState.BOUND:
            raise RuntimeError("OVStage scene publisher pair is unavailable")

    def producer_publication_required(self, lease: OvxrRenderLease, physics_step_id: Hashable) -> bool:
        """Check publication on the application thread without entering the render thread."""
        self.require_publishable(lease)
        try:
            hash(physics_step_id)
        except TypeError as exc:
            raise TypeError("physics_step_id must be hashable") from exc
        writer_ref = self._renderer._visual_material_writer_ref
        writer = writer_ref() if writer_ref is not None else None
        return (
            self._last_step_id != physics_step_id
            or self._marker_revision() != self._published_marker_revision
            or (writer is not None and writer.has_pending_updates)
        )

    def dispatch_marker_callbacks(self) -> None:
        """Run the simulation's visualization-marker callbacks so the next scene write draws current markers."""
        self._renderer._sim.vis_marker_registry.dispatch_callbacks()

    def _marker_revision(self) -> int | None:
        """Revision of the simulation's marker groups that this renderer draws, or ``None`` without markers.

        The registry revision changes when a group is added, updated, hidden or removed.
        """
        if not self._renderer.marker_groups and not self._renderer.markers:
            return None
        return self._renderer._sim.vis_marker_registry.revision

    def write_scene_components(self, lease: OvxrRenderLease, ordinal: int, physics_step_id: Hashable) -> None:
        """Write every scene component at a reserved ordinal; the publisher validates and seals."""
        if ordinal <= 0:
            raise ValueError("OVStage publication ordinal must be nonzero")
        self.require_publishable(lease)
        try:
            self._write_components(ordinal)
        except Exception as exc:
            self._poisoned = True
            raise RuntimeError("Producer scene publication failed; recreate the exact pair") from exc
        self._last_step_id = physics_step_id
        self._published_marker_revision = self._marker_revision()

    def product_paths(self, lease: OvxrRenderLease) -> tuple[str, ...]:
        """Return products in application selection order, independently of sensor initialization order."""
        self.validate_live(lease, lease.ovrtx_renderer, lease.ovstage, lease._generation)
        return self._renderer._panels.product_paths()

    def prepare_stage(self, num_envs: int) -> bool:
        """Validate or defer leased preparation; ordinary rendering is immediately allowed.

        Camera initialization prepares every registered renderer on the application thread. A leased
        pair only records that request until Capture's render worker claims the owner transfer.

        Returns:
            Whether the renderer should materialize the Stage now.
        """
        lease = self.active
        if lease is not None:
            if not self._transfer_claimed:
                if lease._state is not _OvxrLeaseState.UNBOUND:
                    raise RuntimeError("Cannot replace the Stage of a bound renderer/scene pair")
                self.stage_deferred = True
                return False
            self.require_owner_thread("prepare the leased Stage")
            if lease._state is not _OvxrLeaseState.UNBOUND:
                raise RuntimeError("Cannot replace the Stage of a bound renderer/scene pair")
            if num_envs != 1:
                raise RuntimeError("The render lease requires exactly one environment")
            if not self._created_pair:
                # Clone preparation built an ordinary pair because the session never created its own.
                raise RuntimeError("The render lease requires the pair that create_native_pair() creates")
            with self._renderer.scene.stage.get_attribute_write_floor() as floor:
                floor.wait()
                # A new OVStage is sealed through the minimum ordinal; any published write raises the floor.
                if int(floor.fetch()) != ovstage.ORDINAL_MIN:
                    raise RuntimeError("The render lease requires an unused OVStage")
        return True

    def materialize_deferred_stage(self, lease: OvxrRenderLease, stage: Any, num_envs: int) -> bool:
        """Materialize a deferred Stage, with any selected cameras, after render-thread ownership transfer."""
        self.require_owner_thread("materialize a deferred Stage")
        # Unclaimed, a fallback prepare_stage would only defer again; reject even if nothing is deferred yet.
        if not self._transfer_claimed:
            raise RuntimeError("Deferred Stage materialization requires a claimed owner transfer")
        if not self.stage_deferred:
            return False
        if self.active is not lease or lease._state is not _OvxrLeaseState.UNBOUND:
            raise RuntimeError("Deferred Stage is not eligible for materialization")
        if not self._renderer._panels.prepared():
            raise RuntimeError("Deferred auxiliary camera Stage is not eligible for materialization")
        self._renderer.prepare_stage(stage, num_envs)
        return True

    def bind_stage(self) -> None:
        """Bind the prepared native pair after renderer initialization succeeds."""
        lease = self.active
        if lease is not None:
            self._renderer._panels.materialize()
            self.stage_deferred = False
            self._generation += 1
            lease._bind(self._renderer.backend.renderer, self._renderer.scene.stage, self._generation)

    def _next_ordinal(self) -> int:
        """Return ``floor + 1`` from the exact renderer-owned Stage."""
        if self._renderer.scene.stage is None:
            raise RuntimeError("Cannot query an OVStage floor before scene materialization")
        with self._renderer.scene.stage.get_attribute_write_floor() as floor_query:
            floor_query.wait()
            floor = int(floor_query.fetch())
        if floor >= ovstage.ORDINAL_MAX:
            raise RuntimeError("OVStage ordinal range is exhausted")
        return floor + 1

    def _write_components(self, ordinal: int) -> None:
        """Write selected scene components; the caller owns validation and commit."""
        self._renderer.scene.ordinal = ordinal
        with self._renderer._publish_visual_materials():
            self._renderer._publish_transforms()
            self._renderer._publish_geometries()
            self._renderer._panels.publish_poses()


class _OvxrPanelCameras:
    """Camera panels selected for the OVXR session, owned by ``OvxrRenderer``.

    They match the selected camera sensors, create their render products once the session's render
    worker prepares the Stage, author native frame rates on the export, and hold each camera's latest
    pose sample until scene publication writes it. Product creation and pose writes use the borrowed
    native pair, so they run under the lease, which itself handles only ownership, retirement and
    publication ordering.
    """

    def __init__(self, renderer: OvxrRenderer):
        self._renderer = renderer
        self.cameras: list[_AuxiliaryCamera] = []

    def clear(self) -> None:
        self.cameras = []

    def configure(self, camera_cfg: Any, *, max_fps: float | None = None) -> None:
        """Select an exact camera for the current lease; see :meth:`OvxrRenderLease.configure_auxiliary_camera`."""
        if camera_cfg is None:
            raise ValueError("Auxiliary camera configuration cannot be None")
        if any(camera.cfg.prim_path == camera_cfg.prim_path for camera in self.cameras):
            raise RuntimeError("This auxiliary camera is already configured")
        if max_fps is not None and (not math.isfinite(max_fps) or max_fps <= 0.0):
            raise ValueError("Auxiliary camera max_fps must be finite and positive [Hz]")
        self.cameras.append(_AuxiliaryCamera(camera_cfg, max_fps))

    def for_spec(self, spec: CameraRenderSpec) -> _AuxiliaryCamera | None:
        """Return the selected camera that ``spec`` describes, or ``None`` for an ordinary camera."""
        if not self.cameras or not spec.camera_prim_paths:
            return None
        for camera in self.cameras:
            expected = camera.cfg
            pattern = str(expected.prim_path)
            if "{ENV_REGEX_NS}" in pattern:
                roots = self._renderer.scene.clone_env_paths
                if len(roots) != 1:
                    raise RuntimeError("An external camera requires exactly one cloned environment root")
                pattern = pattern.replace("{ENV_REGEX_NS}", re.escape(roots[0]))
            if (
                int(spec.cfg.width) == int(expected.width)
                and int(spec.cfg.height) == int(expected.height)
                and tuple(spec.cfg.data_types) == tuple(expected.data_types)
                and re.fullmatch(f"(?:{pattern})(?:/camera)?", str(spec.camera_prim_paths[0])) is not None
            ):
                # Only a matched camera is external; other specifications take the ordinary-camera path.
                if spec.num_instances != 1 or spec.view_count != 1 or len(spec.camera_prim_paths) != 1:
                    raise RuntimeError("An external camera requires exactly one camera view")
                return camera
        return None

    def accept(self, spec: CameraRenderSpec) -> _AuxiliaryCamera | None:
        """Return the selected camera for ``spec`` after recording its specification."""
        camera = self.for_spec(spec)
        if camera is None:
            return None
        if camera.spec is None:
            camera.spec = spec
        elif camera.spec != spec:
            raise RuntimeError("Auxiliary camera specification changed after authorization")
        return camera

    def prepared(self) -> bool:
        """Whether every selected camera has its specification and render data."""
        return all(camera.spec is not None and camera.render_data is not None for camera in self.cameras)

    def product_paths(self) -> tuple[str, ...]:
        """Return products in application selection order, independently of sensor initialization order."""
        paths = tuple(camera.product_path for camera in self.cameras)
        if any(not isinstance(path, str) or not path.startswith("/") for path in paths) or len(set(paths)) != len(
            paths
        ):
            raise RuntimeError("Auxiliary camera did not materialize exactly one RenderProduct per selection")
        return paths

    def populate(self, usd: str) -> None:
        """Populate the renderer's scene with the selected cameras' native frame rates."""
        usd = self._author_schedules(usd)
        scheduled_paths = [camera.spec.camera_prim_paths[0] for camera in self.cameras if camera.max_fps is not None]
        if not scheduled_paths:
            self._renderer.scene.populate(usd)
            return
        population = ovstage.population
        description = population.Desc(
            domains=self._renderer.scene.cfg.population_domains,
            selectors=[
                population.Selector(
                    population.PrimPredicate.has_path(*scheduled_paths),
                    population.PropertyPredicate.has_name("omni:sensor:frameRate", "omni:sensor:Core:partialOutputs"),
                )
            ],
        )
        self._renderer.scene.populate(usd, population_desc=description)

    def materialize(self) -> None:
        """Register the selected cameras' products in the prepared Stage."""
        for camera in self.cameras:
            if camera.spec is None or camera.render_data is None:
                raise RuntimeError("Selected auxiliary camera was not prepared")
            self._renderer._register_camera(camera.spec, camera.render_data)
            camera.product_path = camera.render_data.render_product_path
        if self.cameras:
            self._renderer.scene.commit()

    def queue_update(self, render_data, positions, orientations, intrinsics) -> bool:
        """Retain the latest pose sample of a selected camera until publication."""
        for camera in self.cameras:
            if camera.render_data is render_data:
                camera.state = (render_data, positions, orientations, intrinsics)
                return True
        return False

    def validate_samples(self) -> None:
        """Validate the whole camera batch before any scene component starts writing."""
        for camera in self.cameras:
            if camera.state is None or camera.render_data is None or camera.render_data.camera_xform_binding is None:
                raise RuntimeError("Auxiliary camera has no live world-pose sample")

    def publish_poses(self) -> None:
        """Write each selected camera's latest pose at the publication ordinal."""
        for camera in self.cameras:
            if camera.state is None or camera.render_data.camera_xform_binding is None:
                raise RuntimeError("Auxiliary camera has no live world-pose sample")
            self._renderer._publish_camera_pose(*camera.state)

    def _author_schedules(self, usd: str) -> str:
        """Author selected native camera rates on the private export, preserving the application stage.

        The rates are authored on the export's layer, so composition arcs that the export
        preserved are not flattened. An export without rate-limited cameras is returned unchanged.
        """
        scheduled = [camera for camera in self.cameras if camera.max_fps is not None]
        if not scheduled:
            return usd
        layer = Sdf.Layer.CreateAnonymous("borrowed-renderer.usda")
        if not layer.ImportFromString(usd):
            raise RuntimeError("Cannot parse the exported leased Stage")
        stage = None
        for camera in scheduled:
            if camera.spec is None:
                raise RuntimeError("Native-rate auxiliary camera was not prepared")
            path = Sdf.Path(camera.spec.camera_prim_paths[0])
            if layer.GetPrimAtPath(path) is None:
                # A camera inside a preserved reference has no spec of its own; check the composed export.
                stage = stage or Usd.Stage.Open(layer)
                if not stage.GetPrimAtPath(path).IsValid():
                    raise RuntimeError("Native-rate auxiliary camera is absent from the exported Stage")
            prim = Sdf.CreatePrimInLayer(layer, path)
            for name, value_type, value in (
                ("omni:sensor:frameRate", Sdf.ValueTypeNames.Double2, Gf.Vec2d(camera.max_fps, 1.0)),
                ("omni:sensor:Core:partialOutputs", Sdf.ValueTypeNames.Bool, False),
            ):
                attribute = prim.attributes.get(name) or Sdf.AttributeSpec(prim, name, value_type)
                attribute.default = value
        return layer.ExportToString()


class _OvxrVisualMaterialWriter(OVRTXVisualMaterialWriter):
    """Visual-material writer of the OVXR renderer, which may exist before its deferred Stage.

    Environment reset finalizes material consumers before the session's render worker prepares the
    Stage, so addresses compile on the first publication after population.
    """

    def __init__(self, renderer: OvxrRenderer, batches: tuple[Any, ...]):
        self._pending_groups: list[tuple] = []
        super().__init__(renderer, batches)

    @property
    def has_pending_updates(self) -> bool:
        """Whether material changes require a scene publication even without a physics step."""
        return bool(self._dirty_channels)

    def publish(self) -> None:
        if self._dirty_channels and self._pending_groups:
            renderer = self._renderer_ref()
            if not renderer._initialized_scene:
                raise RuntimeError("Visual materials require a populated renderer scene before publication.")
            super()._compile_addresses(renderer, self._pending_groups)
            self._pending_groups = []
        super().publish()

    def close(self) -> None:
        self._pending_groups = []
        super().close()

    def _compile_addresses(self, renderer: OVRTXRenderer, groups: list[tuple]) -> None:
        if not renderer._initialized_scene:
            self._pending_groups = list(groups)
            return
        super()._compile_addresses(renderer, groups)


@configclass
class OvxrRendererCfg(OVRTXRendererCfg):
    """OVRTX renderer configuration for an OVXR session and the cameras it shows as panels.

    Give the session and every camera it shows as a panel the same configuration, so they share one
    ``OvxrRenderer``. Cameras with an ordinary :class:`~isaaclab_ov.renderers.OVRTXRendererCfg`
    keep Isaac Lab's ordinary renderer.
    """

    class_type: type[OvxrRenderer] | str = "{DIR}.ovxr_renderer:OvxrRenderer"


class OvxrRenderer(OVRTXRenderer):
    """OVRTX renderer whose native renderer and OVStage the OVXR session borrows.

    The session acquires the lease before stage preparation. While it is active, the session's
    render worker owns the native pair, and ordinary camera rendering, scene writes and renderer
    destruction on this renderer are rejected.
    """

    def __init__(self, cfg: OvxrRendererCfg, *, sim: Any):
        super().__init__(cfg, sim=sim)
        self._closed = False
        self._panels = _OvxrPanelCameras(self)
        self._render_lease = _OvxrLeaseController(self)

    def acquire_render_lease(self) -> OvxrRenderLease:
        """Reserve the native renderer/scene pair before stage preparation."""
        return self._render_lease.acquire()

    def prepare_stage(self, stage: Usd.Stage, num_envs: int) -> None:
        """Prepare the stage now, or defer it until the session's render worker claims the pair."""
        if not self._render_lease.prepare_stage(num_envs):
            return
        super().prepare_stage(stage, num_envs)
        self._render_lease.bind_stage()

    @property
    def visual_material_writer(self):
        """Return a writer factory once the scene exists, or from acquisition while leased.

        A leased pair registers its writer before the render worker prepares the deferred Stage,
        because environment reset finalizes material consumers first; addresses compile on the
        first publication after population.
        """
        if self._render_lease.active is not None:
            return self._create_visual_material_writer
        return super().visual_material_writer

    def _create_visual_material_writer(self, batches: tuple[Any, ...]) -> _OvxrVisualMaterialWriter:
        writer = _OvxrVisualMaterialWriter(self, batches)
        self._visual_material_writer_ref = weakref.ref(writer)
        return writer

    # Extension points of OVRTXRenderer.

    def _populate_stage(self, usd: str) -> None:
        self._panels.populate(usd)

    def _defer_camera_registration(self, spec: CameraRenderSpec) -> bool:
        return self._render_lease.stage_deferred and self._panels.for_spec(spec) is not None

    # Scene writes for the session's publisher, without this renderer's lease guards.

    def _publish_transforms(self) -> None:
        super().update_transforms()

    def _publish_geometries(self) -> None:
        super().update_geometries()

    def _publish_camera_pose(self, render_data, positions, orientations, intrinsics) -> None:
        super().update_camera(render_data, positions, orientations, intrinsics)

    # Ordinary operations, rejected while the session holds the lease.

    def create_render_data(self, spec: CameraRenderSpec) -> OVRTXCameraRenderData:
        """Register an ordinary camera, or reserve a selected panel camera until the handoff."""
        if str(wp.get_device(spec.device)) != self._device:
            raise ValueError("Cameras sharing an OVRTX renderer must use the simulation device.")
        camera = self._panels.accept(spec)
        if camera is None:
            self._render_lease.reject_ordinary_operation("create ordinary camera render data")
        elif camera.render_data is not None:
            raise RuntimeError("Auxiliary camera render data already exists")
        data = super().create_render_data(spec)
        if camera is not None:
            camera.render_data = data
        return data

    def update_transforms(self) -> None:
        # While leased, the session's scene publisher owns these writes.
        if self._render_lease.active is not None:
            return
        self._render_lease.reject_ordinary_operation("update ordinary transforms")
        super().update_transforms()

    def update_geometries(self) -> None:
        if self._render_lease.active is not None:
            return
        self._render_lease.reject_ordinary_operation("update ordinary geometries")
        super().update_geometries()

    def update_camera(self, render_data, positions, orientations, intrinsics) -> None:
        if self._panels.queue_update(render_data, positions, orientations, intrinsics):
            return
        self._render_lease.reject_ordinary_operation("update ordinary cameras")
        super().update_camera(render_data, positions, orientations, intrinsics)

    def update_camera_intrinsics(self, render_data, intrinsics, parameters):
        self._render_lease.reject_ordinary_operation("update ordinary camera intrinsics")
        return super().update_camera_intrinsics(render_data, intrinsics, parameters)

    def log_mesh(self, name: str, *args, **kwargs) -> None:
        self._render_lease.reject_ordinary_operation("log runtime meshes")
        super().log_mesh(name, *args, **kwargs)

    def render_batch(self, render_data: Sequence[OVRTXCameraRenderData]) -> None:
        if render_data:
            self._render_lease.reject_ordinary_operation("render an ordinary camera batch")
        super().render_batch(render_data)

    def cleanup(self, render_data: OVRTXCameraRenderData | None) -> None:
        """Release camera render data once the lease is released.

        Releasing a product removes it from the borrowed Stage, so a camera keeps its resources
        until the lease is released; :meth:`close` releases them afterwards.
        """
        if render_data is not None:
            self._render_lease.reject_ordinary_operation("release camera render data", retiring=True)
        super().cleanup(render_data)

    def close(self) -> None:
        """Release bindings once the lease is released; the registry owns the native engine."""
        self._render_lease.reject_ordinary_operation("close the renderer", retiring=True)
        try:
            super().close()
        finally:
            self._closed = True
