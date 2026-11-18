* Added :attr:`~isaaclab_teleop.OvxrRendererCfg.active_cuda_gpus` and the ``--ovrtx_active_cuda_gpus`` OVXR
  launch option to render the session on GPUs other than the physics GPU. Rigid transforms and
  deformable points from the simulation GPU are published through completed copies on the first
  rendering GPU. Malformed values are rejected when the configuration is created.
* Added ``python -m isaaclab_teleop.check_cuda_peer``, a preflight that checks the geometry publication
  copy between two GPUs before splitting physics and rendering across them.
