* Changed the SO101 IK-Abs teleoperation pipeline to use a full-pose clutch home. Its
  teleoperation preparation hook selected nominal joint resets for live teleoperation and
  HDF5 recording; ordinary task construction retained the joint-noise standard deviation of ``0.02``.
