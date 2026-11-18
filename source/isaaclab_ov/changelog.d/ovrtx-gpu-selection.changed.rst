* Changed OVRTX to reject a rendering GPU selection that differs from the process's first native
  initialization, including after teardown, instead of silently rendering on the first selection.
  Renderers without an explicit selection request the simulation device, so a CPU-simulation renderer
  followed by a CUDA one, or an ordinary camera beside a renderer on other GPUs, is also rejected.
