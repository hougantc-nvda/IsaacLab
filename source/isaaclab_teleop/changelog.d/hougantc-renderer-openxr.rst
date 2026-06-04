Added
^^^^^

* Added shared kitless OpenXR teleoperation helpers so renderer-owned OpenXR
  visualizers can provide IsaacTeleop handles, anchors, and CloudXR launch
  profiles without depending on Kit XR.

Fixed
^^^^^

* Fixed :class:`~isaaclab_teleop.xr_anchor_manager.XrAnchorManager` raising
  ``RuntimeError`` when carb settings are unavailable in renderer-owned OpenXR
  paths.
