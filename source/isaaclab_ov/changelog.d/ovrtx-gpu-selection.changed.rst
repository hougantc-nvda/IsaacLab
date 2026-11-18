* Changed OVRTX to reject a rendering GPU selection that differs from the process's first native
  initialization, including after teardown, instead of silently rendering on the first selection.
