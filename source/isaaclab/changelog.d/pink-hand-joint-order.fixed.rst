* Fixed Pink hand targets being applied to the wrong joints when the articulation's joint order differs
  from ``hand_joint_names`` (for example, Newton's G1 order).
  Hand-action columns now follow ``hand_joint_names`` configuration order. External recordings
  whose columns followed articulation order must reorder those columns into ``hand_joint_names``
  order before replay; arm-action columns are unchanged.
