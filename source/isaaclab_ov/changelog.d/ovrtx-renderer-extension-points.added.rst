* Added protected methods to :class:`~isaaclab_ov.renderers.OVRTXRenderer` and its visual material writer
  that a renderer subclass reuses to publish dirty materials around a scene update and to compile material
  addresses after scene population. Ordinary rendering is unchanged.
