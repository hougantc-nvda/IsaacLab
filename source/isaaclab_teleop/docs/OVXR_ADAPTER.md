# OVXR application adapter

`isaaclab_teleop.OvLibrariesXrSession` connects an Isaac Lab environment to
Isaac Capture's OVXR session owner. The simulation renders with
`isaaclab_teleop.ovxr_renderer.OvxrRenderer` (selected by `OvxrRendererCfg`), an OVRTX renderer whose
native renderer and OVStage the session borrows through its lease.

## Ownership

- The simulation owns the renderer and stage and their lifetime. `OvxrRenderer`'s lease
  validates access, enforces render-thread ownership and orders scene publication. The adapter's
  publication guard serializes application writes with Capture's frame operations.
  It does not create an XR session or select teleoperation behavior.
- `isaaclab_teleop.ovxr_session.OvxrSessionBootstrap` creates the simulation
  context and reserves the lease before environment construction. It binds the
  completed environment and reads current target-frame transforms.
- `isaaclab_teleop.ov_libraries_xr.OvLibrariesXrSession` supplies the application
  adapter to Isaac Capture. It declares the two eye cameras, publishes scene and
  anchor updates, transfers render-thread ownership and coordinates shutdown.
  OVXR authors its own eye render products, render vars and render settings.
- Isaac Capture owns the OVXR session and render worker. Its borrowed OpenXR
  handles are available to the teleoperation input session.

## Lifecycle

Prepare the adapter before constructing the environment so that both use the
same simulation context and native renderer/scene pair. Start the render worker
after the environment exists. Publish simulation updates through the session;
the worker owns rendering. Stop input consumers before closing the XR session,
then close the environment on the host thread.

`prepare()` registers the session's retirement as the simulation's first
`PhysicsEvent.STOP` listener. Closing the environment clears the kitless
simulation context, which dispatches STOP before Isaac Lab releases cameras,
renderers, the native OVRTX engine and the OVStage, so the worker drains and
the lease is released first. The listener closes a session that is still open
only if it handed out no OpenXR handles. A STOP while the simulation runs (a
hard reset or a physics re-warm) raises and leaves the session open.

If startup fails before the worker takes ownership, the caller closes the
environment after releasing the lease. Closing the session waits for a worker
that is still attaching the scene, because attachment cannot be cancelled and
reads the environment. If input retirement fails, request
shutdown and leave the session open. Whenever the STOP listener cannot prove
retirement during a shutdown, it detaches the leased renderer and OVStage from
the simulation's backends, keeps them alive until the process exits, logs an
error and raises, so teardown destroys neither under a live render owner.

The lease currently requires one environment, and the session renders on its own OVStage.
Ownership transfer is single-use. Failed retirement must preserve resources;
a failed drain is not successful shutdown. Importing `OvLibrariesXrSession` requires a Capture release with OVXR
session support; see [Installing kitless XR](kitless_xr_source.md). The
`OvLibrariesXrCfg` dataclass lives in `isaaclab_teleop.ovxr_cfg` and imports
without those packages.

A failed scene write or seal stops further publication. Recreate the session
after retiring the failed pair; retrying the same generation cannot return an
older scene as if the failed update succeeded. The failure poisons producer
publication only: the render-owner lane stays available so Capture can destroy
its OVXR context and the adapter can release the lease before the host closes the
environment. After a poisoned lease is released, the renderer keeps rejecting
ordinary use (except closing it) because its write ordinal was not rebased.
During shutdown, publication storage drains before lease guards are removed.
The adapter retains the renderer and Stage if lease retirement cannot establish completion.

## Environment requirements

- `OvLibrariesXrSession.prepare()` creates the `SimulationContext` from the
  `sim_cfg` it receives, and the environment reuses that context. Pass the
  environment's own `env_cfg.sim`; before the worker claims the lease,
  `start_render_worker()` rejects an environment whose simulation device or
  time step differs, or whose `num_envs` is not 1.
- Only environments that accept a pre-created simulation context are
  supported. Manager-based environments do; `DirectRLEnv` always creates its own
  context and fails with "Simulation context already exists".
- Camera sensors on other renderers are supported: their initialization
  prepares every registered renderer on the application thread, and the XR
  renderer defers its Stage until the XR worker claims ownership. A camera that
  resolves to the XR renderer itself (a camera configured with the session's
  `OvxrRendererCfg`) is rejected while the lease is active.
- Eager scene updates (`lazy_sensor_update=False`) skip the leased renderer; its
  scene publisher writes the same physics state at the reserved ordinal.
- CPU simulation devices publish scene transforms, but visual-material
  publication requires CUDA material buffers, as it does for ordinary OVRTX
  cameras. With `--device cpu`, an environment that has `VisualMaterial` assets
  fails at `sim.reset()`.
- Dynamic anchoring takes any object implementing
  `OvLibrariesXrSession.ExternalAnchorState`; `start_render_worker()` rejects
  other objects before the worker claims the lease.

## Backend implementation

The ordinary renderer remains in `isaaclab_ov.renderers.ovrtx_renderer` and has no XR
knowledge. `isaaclab_teleop.ovxr_renderer.OvxrRenderer` subclasses it: it overrides Stage
preparation and the ordinary camera and scene operations.

- Its `_OvxrLeaseController` owns the lease state, the thread-transfer capability, the
  per-owner operation guards and the publication order.
- `OvxrScenePublisher.publish()` publishes each complete scene as one transaction: it skips an
  unchanged state, reserves an ordinal, writes every component, seals
  it, and on failure poisons the pair and abandons the ordinal. Capture's publication guard
  provides the cross-thread exclusion around it.

The lease registers the renderer as a consumer of the simulation's visualization-marker registry.
This enables the existing marker backend even without an ordinary visualizer and lets lazily created
tracking markers participate in scene publication. The registration and registry subscription retire
with the lease.

The lease classes are internal to `isaaclab_teleop`.

Acquiring a lease reserves the renderer. After the session applies its renderer
settings, and before the environment is built, it creates its own rendering OVStage
and the OVRTX engine that attaches to it, through the simulation's backend registry.
Clone preparation then finds the renderer initialized and routes the scene copies to
that Stage.
Without an active lease, the controller permits
ordinary renderer operations and does not start an XR session. While a consumer
holds the lease, ordinary camera rendering, scene writes and renderer destruction
are rejected to prevent concurrent access to the borrowed resources. Cameras with an
ordinary `OVRTXRendererCfg` keep Isaac Lab's ordinary renderer object, but in a process
with an OVXR session they follow it onto an OVStage of their own, where OVRTX renders
synchronously.

## Native validation

The adapter and lease unit tests double the native OVRTX and OVStage endpoints.
An opt-in harness runs the render-owner transfer and complete scene publication
against the real libraries, on the manager-based `Isaac-Cartpole` task with its
`ovphysx` physics preset and one environment on `cuda:0`. The transfer case
also runs with CPU physics, which publishes host scene tensors to the GPU
renderer. It needs a CUDA GPU, OVRTX, OVStage, OvPhysX and an Isaac Capture
release with OVXR session support; it starts no XR session or CloudXR runtime.
Each case runs in a child process:

```bash
ISAACLAB_RUN_NATIVE_OVRTX_TRANSFER=1 uv run python -m pytest \
    source/isaaclab_teleop/test/test_ovrtx_transfer_native_harness.py
```

Without `ISAACLAB_RUN_NATIVE_OVRTX_TRANSFER=1`, every case is skipped.
