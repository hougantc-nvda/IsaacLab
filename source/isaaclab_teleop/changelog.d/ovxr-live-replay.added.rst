* Added kitless OVXR teleoperation and MCAP replay with independently paced rendering,
  backend-derived anchors, and ordered shutdown.
* Added ``--start_on_replay`` to the MCAP replay agent. It starts playback at startup and after
  each recorded RESET, for captures made without recorded START messages.
* Added external OVXR inputs to :class:`~isaaclab_teleop.IsaacTeleopDevice`, together with the
  ``target_frame_prim_path``, ``xr_render_anchor_path``, ``external_anchor_state``,
  ``required_openxr_extensions`` and ``session_cleanup_failed`` properties.
* Added :meth:`~isaaclab_teleop.IsaacTeleopDevice.notify_reset`, which reports an environment reset
  to the input pipeline: retargeters reset on their next execution and actions computed before the
  reset are withheld, while operator control events are preserved.
* Added OVXR launch validation that rejects dynamic ``CUSTOM`` XR anchors, whose callback needs a
  previous head pose that OVXR does not supply. Kit keeps supporting them.
