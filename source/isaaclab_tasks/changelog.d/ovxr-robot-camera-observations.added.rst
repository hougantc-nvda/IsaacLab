* Added fixed-base G1 wrist cameras, and hid the robot's head mesh from the OVXR render.
* Added an ``rgb`` preset to ``IsaacContrib-PickPlace-FixedBaseUpperBodyIK-G1-Abs`` that adds the
  wrist cameras, their XR camera panels and the head and wrist RGB images to its policy
  observations (``presets=rgb``). The default renders only the head camera and keeps the policy
  observations state-only.
