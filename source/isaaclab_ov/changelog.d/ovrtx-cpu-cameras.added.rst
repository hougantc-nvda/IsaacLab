* Added CPU simulation-device support to OVRTX scene publication of transforms, geometry, runtime
  meshes and markers, so kitless OVXR can present CPU physics.
* Added CPU simulation-device support to OVRTX cameras. Render products leave GPU selection automatic,
  and camera poses, intrinsics and image reads use host memory without a CUDA stream, so kitless OVXR
  camera panels can run with CPU physics.
