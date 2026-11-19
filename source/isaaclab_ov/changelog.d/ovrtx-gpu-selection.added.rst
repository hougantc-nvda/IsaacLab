* Added protected extension points to :class:`~isaaclab_ov.renderers.OVRTXRenderer` for the native rendering
  GPU selection, the render-product device and the OVStage writes of rigid poses and points by a subclass
  that lends its native renderer and private OVStage to another owner.
* Added :attr:`~isaaclab_ov.renderers.OVRTXBackendCfg.active_cuda_gpus`, which clone preparation copies from
  the renderer's GPU-selection extension point. :class:`~isaaclab_ov.renderers.OVRTXRendererCfg` gains no GPU
  selection, so ordinary OVRTX cameras still render on the simulation device.
