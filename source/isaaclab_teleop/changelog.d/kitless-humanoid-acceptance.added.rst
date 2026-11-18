* Added task hooks to :class:`~isaaclab_teleop.IsaacTeleopCfg`: ``require_initial_bimanual_hand_tracking``
  and the experimental ``action_conditioner`` and ``action_validator``. The live and replay agents
  apply them before each step.
* Added the :attr:`~isaaclab_teleop.IsaacTeleopDevice.hand_tracking_presence` property, which
  reports left and right hand presence in the latest input frame.
* Added an HDF5 recording check that rejects kitless physics for tasks with an action policy, which
  only the live and replay agents apply.
