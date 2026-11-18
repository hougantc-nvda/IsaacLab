* Added protected extension points to :class:`~isaaclab_ov.renderers.OVRTXRenderer` for a subclass that
  lends its native renderer and private OVStage to another owner: scene population, deferred camera
  registration, private-OVStage selection during clone preparation, the native rendering GPU
  selection, the render-product device and the OVStage writes of rigid poses and points.
* Added the ``population_desc`` argument to :meth:`~isaaclab_ov.stage.OvstageBackend.populate`, which
  also populates native properties a rendering consumer selects, such as a camera's frame rate.
* Added :attr:`~isaaclab_ov.renderers.OVRTXBackendCfg.active_cuda_gpus`, which clone preparation copies from
  the renderer's GPU-selection extension point. :class:`~isaaclab_ov.renderers.OVRTXRendererCfg` gains no GPU
  selection, so ordinary OVRTX cameras still render on the simulation device.
