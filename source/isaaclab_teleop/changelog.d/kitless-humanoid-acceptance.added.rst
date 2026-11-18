* Added task hooks to :class:`~isaaclab_teleop.IsaacTeleopCfg`: ``action_conditioner``,
  ``action_validator`` and ``require_initial_bimanual_hand_tracking``. The live and replay agents
  apply them before each step.
* Added the :attr:`~isaaclab_teleop.IsaacTeleopDevice.hand_tracking_presence` property, which
  reports left and right hand presence in the latest input frame.
* Added an HDF5 recording check that rejects kitless physics for tasks with an action policy, which
  only the live and replay agents apply.
