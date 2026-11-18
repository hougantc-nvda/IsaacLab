* Added protected extension points to :class:`~isaaclab_ov.renderers.OVRTXRenderer` for a subclass that
  lends its native renderer and private OVStage to another owner: scene population, deferred camera
  registration, private-OVStage reservation during clone preparation, the native rendering GPU
  selection, the render-product device and the OVStage writes of rigid poses and points.
  The engine and Stage refuse teardown and ordinary commits while such an owner holds them.
* Added :attr:`~isaaclab_ov.renderers.OVRTXBackendCfg.active_cuda_gpus` to select the native rendering GPUs.
