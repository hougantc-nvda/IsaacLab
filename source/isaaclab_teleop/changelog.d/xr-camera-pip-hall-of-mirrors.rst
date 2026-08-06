Fixed
^^^^^

* Fixed recursive XR camera PiP rendering by assigning application SceneUI and the XR
  presentation camera to one RTX scene partition, keeping the panel visible in the
  headset while excluding it from robot-camera render products.
