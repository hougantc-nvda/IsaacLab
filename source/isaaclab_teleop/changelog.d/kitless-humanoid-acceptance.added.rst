* Added task hooks to :class:`~isaaclab_teleop.IsaacTeleopCfg`: ``require_initial_bimanual_hand_tracking``
  and the experimental ``action_conditioner`` and ``action_validator``. The live, replay and recording
  agents apply them before each step.
* Added the :attr:`~isaaclab_teleop.IsaacTeleopDevice.hand_tracking_presence` property, which
  reports left and right hand presence in the latest input frame.
* Added ``--require_hand_replay_data`` to the MCAP replay agent. It ends a run at the end of the recorded
  two-hand data, without stepping the held last action, and reports it as complete when the recording
  delivered changing actions.
