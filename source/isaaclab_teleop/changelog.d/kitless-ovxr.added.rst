* Added :class:`~isaaclab_teleop.OvLibrariesXrSession`, an OVXR application adapter and session
  bootstrap that borrow the simulation-owned OVRTX renderer and OVStage, publish scene updates,
  and coordinate render-thread ownership and shutdown with Isaac Capture.
  A failed scene publication stops further publication, and a session whose renderer cannot be
  retired keeps its borrowed resources.
* Added :class:`~isaaclab_teleop.OvxrRendererCfg`, the OVXR session's OVRTX renderer. It lends its
  native renderer and its own OVStage to Isaac Capture's render worker, with a single-use
  render-thread ownership transfer, deferred Stage preparation, ordered scene publication
  (including dirty visual materials) and ordered cleanup. Cameras with an ordinary OVRTX
  configuration keep Isaac Lab's renderer.
* Added :class:`~isaaclab_teleop.OvLibrariesXrCfg` in :mod:`isaaclab_teleop.ovxr_cfg`, which
  imports without the native OVXR packages.
