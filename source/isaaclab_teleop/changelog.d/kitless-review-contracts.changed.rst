* Refreshed an existing XR anchor's configured pose for each new Kit or OVXR session instead of
  retaining the previous session's pose. Update ``XrCfg`` to select the next session's anchor.
* Changed ``run_teleoperation`` to own its cleanup: it no longer takes a ``cleanup`` stack, and
  requires the keyword-only ``use_isaac_teleop`` argument.
