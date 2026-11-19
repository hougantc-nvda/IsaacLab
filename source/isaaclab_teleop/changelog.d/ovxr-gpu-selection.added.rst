* Added :attr:`~isaaclab_teleop.OvxrRendererCfg.active_cuda_gpus` and the ``--ovrtx_active_cuda_gpus`` OVXR
  option of live teleoperation and recording to render the session on GPUs other than the physics GPU.
  Rigid transforms and deformable points from the simulation GPU are published through completed copies
  on the first rendering GPU. Malformed values are rejected when the configuration is created. Ordinary
  OVRTX cameras still render on the simulation device, so they cannot share a process with a session
  that renders on other GPUs.
* Added ``python -m isaaclab_teleop.tools.check_cuda_peer``, a preflight that checks the geometry publication
  copy between two GPUs before splitting physics and rendering across them.
