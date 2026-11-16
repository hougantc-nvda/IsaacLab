* Refreshed an existing XR anchor's configured pose for each new Kit or OVXR session instead of
  retaining the previous session's pose. Update ``XrCfg`` to select the next session's anchor.
* Made ``run_teleoperation``'s mode selection keyword-only. Callers should pass
  ``use_isaac_teleop=...``; a former positional cleanup stack is no longer accepted as a mode.
