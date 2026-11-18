* Changed Kit ``--xr`` head-panel uploads for the fixed-base G1 task to a 20 Hz cap. With
  ``presets=rgb``, it also shows wrist panels capped at 10 Hz.
* Changed fixed-base G1 with ``presets=rgb`` to re-render three times after each reset
  (``num_rerenders_on_reset = 3``), so its head and wrist images are not stale after a reset.
* Changed the fixed-base G1 policy observation selection, wrist cameras and XR camera feeds to
  presets. Programmatic callers should use ``resolve_task_config`` (or ``resolve_presets(cfg)`` after
  direct construction) before reading or modifying them. The resolved default keeps the
  state-only schema. Camera-panel upload caps do not change the camera sensor's sampling period.
