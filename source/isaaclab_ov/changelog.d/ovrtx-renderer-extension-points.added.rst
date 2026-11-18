* Added protected extension points to :class:`~isaaclab_ov.renderers.OVRTXRenderer` for a subclass that
  lends its native renderer and private OVStage to another owner: scene population, deferred camera
  registration and private-OVStage reservation during clone preparation.
  The engine and Stage refuse teardown and ordinary commits while such an owner holds them.
