* Added protected extension points to :class:`~isaaclab_ov.renderers.OVRTXRenderer` for scene population
  and deferred camera registration by a subclass that lends its native renderer and private OVStage to
  another owner.
* Added the ``population_desc`` argument to :meth:`~isaaclab_ov.stage.OvstageBackend.populate`, which
  also populates native properties a rendering consumer selects, such as a camera's frame rate.
