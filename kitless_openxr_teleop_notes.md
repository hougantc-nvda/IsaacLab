# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Kitless OpenXR Teleop Notes

Baseline implementation commit: `e980b30858e3d2a023d0a0ffd03d09faa1621d04`

These notes are meant to explain the branch at a technical-direction level
first, then preserve lower-level details for anyone who needs to reproduce,
debug, or continue the work.

## Goal

Support IsaacTeleop without Kit owning the application shell, rendering, or
OpenXR session.

The design target is a kitless teleop path where:

- IsaacLab teleop scripts stay focused on task setup, environment stepping, and
  teleop device creation.
- A shared kitless launcher handles non-Kit lifecycle work that Kit used to
  provide implicitly.
- The selected visualizer owns renderer-specific OpenXR policy.
- Native renderer code owns OpenXR instance/session/swapchains/frame submission.
- Python only initializes the renderer OpenXR bridge, passes settings, pumps
  frames, and exposes IsaacTeleop-compatible handles.

Newton is the first implementation of this model. It should not be the only
one. The branch is structured so a future ovrtx/ovxr visualizer can plug into
the same IsaacLab path instead of adding renderer-specific branches to every
teleop script.

### OpenXR Renderer Requirements

MVP renderer requirements:

- Renderer-owned OpenXR lifecycle: create and own `XrInstance`, `XrSession`,
  spaces, swapchains, frame loop, and shutdown.
- IsaacTeleop input interoperability: expose required OpenXR handles and let
  IsaacTeleop bind action sets and input paths.
- Runtime extension negotiation: support required and optional OpenXR
  extensions, device/vendor extension loading, and clear failure modes.
- Stereo rendering: use `xrLocateViews`, predicted display time, per-eye
  scheduling, per-eye projection/view, correct coordinate conventions, and
  correct compositor submission.
- Quadview rendering: support high-resolution inset views plus low-resolution
  background views, with correct scheduling and composition.
- CloudXR-compatible color/depth submission: submit color plus valid per-eye
  depth behavior required by CloudXR.
- Spatial anchors: support USD prim path anchors, relative and absolute
  anchors, live anchor changes, and persisted recenter state.
- Render/XR settings API: support near/far plane, custom anchor prim, camera
  offset, resolution scale, depth mode, alpha blend mode, render-quality
  preset, and parity with settings used by the Kit XR path.
- Render-quality presets: expose performance, balanced, and quality profiles
  that map renderer features such as RTX/DLSS, reflections, sampled lighting,
  and post effects to predictable product-level defaults.
- Alpha blend modes: support opaque composition for VR and alpha composition
  for AR/passthrough profiles.
- GPU submit path without CPU readback for normal operation.
- Performance parity or better than Kit rendering: benchmark IsaacLab Teleop
  Isaac-PickPlace-GR1T2-Abs-v0 at 45 FPS or better on AMD Threadripper 5595WX
  and Ada RTX 6000.
- Foveated rendering: negotiate runtime/device support for fixed foveation and
  warped foveation modes used by VR/AR profiles.
- No-client/synthetic validation path for CI or local smoke testing.

M2 renderer requirements:

- Basic desktop mirror: show the submitted unwarped XR color image in a 2D
  window.
- Diagnostics: log selected runtime, extensions, formats, swapchain size,
  session state, frame timing, and clear failure details.
- Performance profiling hooks: separate GPU/CPU frame timing for render,
  submit, and physics.

Nice-to-have renderer requirements:

- Dynamic foveated rendering or gaze-driven foveation.
- Color/tone mapping controls: SDR/HDR choice, gamma/color-space correctness,
  and desktop/headset match.
- Rich visual debug overlays: per-eye frustums, controller rays, anchor gizmos,
  and depth preview.
- Advanced capture/debug tooling: dump submitted color/depth/AOVs, OpenXR event
  traces, and replayable frame packets.

Non-renderer platform/product requirements:

- Runtime controls API: start/stop XR, resolution scale, depth mode, mirror
  mode, anchor path/offset, and profiling.
- Full UI panel for XR controls and live render settings.
- User-facing configuration model: defaults, saved profiles, command-line
  overrides, and per-task settings.
- Product-level CloudXR lifecycle policy: launch/stop behavior, certificate
  handling, client connection status, and logs.

## Main Issues And Decisions

### Kit Used To Own Too Much

The Kit path implicitly provided app launch, XR extension startup, OpenXR
handles, CloudXR runtime launch, anchor management, and frame presentation. A
kitless renderer does not get those for free.

Decision: add `KitlessTeleopLauncher` in `isaaclab_teleop.openxr_runtime` as
the shared non-Kit lifecycle coordinator. Teleop scripts call this launcher, but
renderer details live below it.

### OpenXR Must Be Renderer-Owned

Early designs that exposed too much OpenXR functionality through Python would
not scale and would not match the direction of ovrtx/ovxr.

Decision: keep OpenXR instance/session/swapchain ownership in native renderer
code. Python gets only the small surface IsaacTeleop needs: a handles provider,
an anchor provider, and a frame pump.

### CloudXR Still Needs A Runtime Profile

CloudXR launch used to happen through IsaacTeleop on the Kit path. Kitless
Newton needs the same capability, but launcher-owned so the renderer OpenXR
session exists at the right time and is launched only once.

Decision: add a `newton` CloudXR profile shorthand that maps to
`newton-openxr-cloudxr.env`. Normal operation uses `--cloudxr_env newton`.
`--cloudxr_env none` disables launcher-owned CloudXR.

### The OpenXR Loader Cannot Come From Kit

The kitless path should not depend on Isaac Sim or Kit XR libraries being
installed. The OpenXR loader must be available before Newton creates the native
session.

Decision: when the renderer runtime requires it, the kitless launcher re-execs
once with the `isaacteleop` packaged OpenXR loader on `LD_LIBRARY_PATH`.

### Host Copies Were Too Slow

The initial OpenXR submit path staged rendered frames through the host. That was
useful for bring-up but too slow for interactive teleop.

Decision: make Newton OpenXR default to CUDA/Vulkan submit, set Newton OpenXR
swapchain scale to `0.5`, render every third physics step for XR, and avoid a
second full desktop render. The desktop viewer mirrors the submitted OpenXR
color image by default. App-side PQW foveation is not built into the
redistributable native target because the available extension header is not
redistributable.

### Depth And Client UI Had To Be Balanced

CloudXR/WebXR needed depth behavior that did not occlude the browser-side UI.
Submitting scene depth was useful during debugging but made the client controls
hard to use.

Decision: scene depth is a visualizer config option but is disabled by default.
Newton submits explicit far depth in the default path so the client UI remains
usable.

### Camera And Input Need The Same Anchor

The headset view and IsaacTeleop retargeting must agree on where the user is in
the world. Otherwise the image can look correct while the wrist targets drift or
track from the wrong origin.

Decision: Newton OpenXR anchors from the robot pelvis/root body when available,
applies the camera offset from `XrCfg.anchor_pos`, recenters the headset origin,
and provides that same anchor matrix to IsaacTeleop.

### GR1T2 Needed Task-Owned Kitless Policy

The GR1T2 task exposed several Kit-era assumptions: Kit/Isaac Sim could convert
USD assets for Pink IK, Kit rendering showed USD visuals, and Kit physics was
stable with existing tuning. Kitless Newton needed explicit policy for assets,
IK, and safety.

Decision: keep GR1T2-specific behavior in
`isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg`, not in generic
teleop scripts. The task now owns a packaged kinematics-only Pink IK URDF,
optional override fields for alternate URDFs, Newton USD import options for the
table/object, wrist target ramping, and Newton-specific stability guards.

### Newton Physics And Newton Rendering Are Separate Choices

We need both Newton physics plus Newton renderer, and OvPhysX CPU plus Newton
renderer. Those should not accidentally share physics-specific tuning.

Decision: Newton physics activates only when Newton physics presets are
requested or the env config already uses `NewtonCfg`. OvPhysX CPU can still use
Newton visualization/OpenXR for rendering.

## Reproducing This Branch

The commands below use shell locals only to keep the snippets readable. Adjust
paths for the workspace.

### 1. Check Out The Branches

```bash
newton_root=/path/to/newton
lab_root=/path/to/IsaacLab

cd "$newton_root"
git checkout hougantc/renderer-openxr

cd "$lab_root"
git checkout hougantc/newton-renderer-openxr
```

### 2. Build Newton Native OpenXR

Newton's native OpenXR target builds `libnewton_openxr_native.so` into the
Newton package tree. The Newton branch vendors the OpenXR headers needed by
this target, so a CloudXR Runtime source checkout is not required for the
build.

```bash
cd "$newton_root"
cmake -S newton/native/openxr -B build/openxr-native
cmake --build build/openxr-native --parallel
test -f "$newton_root/newton/_native/libnewton_openxr_native.so"
```

If CMake cannot find `vulkan/vulkan.h`, install Vulkan development headers or
pass `-DVULKAN_INCLUDE_DIR=/path/to/vulkan/include`. To build against a
different OpenXR header set, pass `-DOPENXR_INCLUDE_DIR=/path/to/openxr/include`.

### 3. Install Newton Into IsaacLab

Install the Newton checkout into IsaacLab's active Python environment:

```bash
cd "$lab_root"
./isaaclab.sh -p -m pip install -e "$newton_root"
```

If an installed Newton wheel remains ahead of the editable checkout, replace it:

```bash
cd "$lab_root"
./isaaclab.sh -p -m pip uninstall -y newton
./isaaclab.sh -p -m pip install -e "$newton_root"
```

Verify that IsaacLab imports the expected Newton package and sees the native
OpenXR presenter:

```bash
cd "$lab_root"
./isaaclab.sh -p -c "import newton; from newton.openxr import NewtonOpenXRPresenter; print(newton.__file__); print(NewtonOpenXRPresenter.find_library_path()); print(NewtonOpenXRPresenter.is_library_built())"
```

The last line should print `True`.

### 4. Use The Packaged GR1T2 Kitless IK Asset

GR1T2 kitless teleop requires a pre-generated Pink IK URDF because this path
must not depend on Isaac Sim conversion at runtime. The branch includes a
kinematics-only URDF in `isaaclab_tasks.contrib.pick_place.assets`, and
`PickPlaceGR1T2EnvCfg` uses it by default.

Users should not need to pass `kitless_kinematics_urdf_path` for the standard
GR1T2 task. The config fields remain only as escape hatches for debugging or
for testing a replacement URDF.

### 5. Run The Demo

Newton physics with Newton renderer:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-PickPlace-GR1T2-Abs-v0 \
  --xr \
  --visualizer newton \
  --device cuda:0 \
  --cloudxr_env newton \
  presets=newton_mjwarp,newton_renderer
```

OvPhysX CPU physics with Newton renderer:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-PickPlace-GR1T2-Abs-v0 \
  --xr \
  --visualizer newton \
  --device cpu \
  --cloudxr_env newton \
  presets=ovphysx
```

Demo recording follows the same launcher path:

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-PickPlace-GR1T2-Abs-v0 \
  --xr \
  --visualizer newton \
  --device cuda:0 \
  --cloudxr_env newton \
  presets=newton_mjwarp,newton_renderer
```

### 6. Run Targeted Checks

```bash
./isaaclab.sh -p -m pytest \
  source/isaaclab_teleop/test/test_cloudxr_lifecycle.py \
  source/isaaclab_teleop/test/test_teleop_se3_agent_kitless.py \
  source/isaaclab_teleop/test/test_newton_openxr_kitless_contract.py \
  source/isaaclab_teleop/test/test_xr_anchor_manager_kitless.py \
  source/isaaclab_tasks/test/test_gr1t2_pickplace_cfg.py \
  -q
```

## How A Future ovrtx/ovxr Renderer Should Slot In

The intent is that ovxr can integrate by implementing the same renderer-owned
OpenXR contract Newton uses.

IsaacLab integration points:

- Add `isaaclab_visualizers.ovxr` or the chosen visualizer package.
- Add `isaaclab_visualizers.ovxr.teleop_runtime`.
- Expose `get_openxr_teleop_runtime()` from that module.
- Return an object implementing `RendererOpenXRTeleopRuntime` from
  `isaaclab_teleop.openxr_runtime`.
- Make the visualizer selectable with `--visualizer ovxr`.
- Implement `configure_openxr_teleop(env_cfg=..., env=...)` on the visualizer.
- Return a renderer session object from `configure_openxr_teleop()`.
- On that session, provide `openxr_handles_provider()` and
  `anchor_world_matrix()`.
- Keep ovxr-specific knobs on the ovxr visualizer config or runtime module.
- Add a CloudXR profile shorthand only if ovxr needs a different runtime
  profile.

Renderer/native API expectations:

- Native code owns OpenXR instance, session, reference spaces, swapchains,
  frame pacing, event polling, view location, and submission.
- Python can initialize settings, call pump/update, request IsaacTeleop handles,
  and shut down.
- `openxr_handles_provider()` returns IsaacTeleop-compatible
  `OpenXRSessionHandles` or `None` until the native session is ready.
- `anchor_world_matrix()` returns the world transform used to retarget incoming
  headset/controller poses.
- Frame submission should prefer direct GPU paths. For ovxr, this likely means
  rendering directly into OpenXR-compatible Vulkan images or using the ovxr
  native sharing path, not staging through CPU memory.
- Foveation support should come through a renderer/runtime API with
  redistributable headers; do not vendor proprietary extension headers into
  IsaacLab or Newton.
- Desktop presentation should avoid a second expensive full render during XR.
  Mirroring the submitted XR image is the preferred shape.

What should not be needed:

- New visualizer-specific branches in `teleop_se3_agent.py`.
- New visualizer-specific branches in `record_demos.py`.
- Python-level OpenXR instance/session/swapchain ownership.
- Isaac Sim or Kit XR runtime dependencies for the kitless renderer path.

Most reviewers can stop here. The rest of this document keeps implementation
and history details for debugging or for continuing the work in another chat.

## Detailed Continuation Notes

### Current Implementation Map

Shared kitless teleop:

- `isaaclab_teleop.openxr_runtime`
  - `KitlessTeleopLauncher`
  - `RendererOpenXRTeleopRuntime`
  - CloudXR profile resolution
  - packaged OpenXR loader re-exec
  - renderer OpenXR session wiring
  - renderer anchor provider attachment

IsaacTeleop:

- `IsaacTeleopCfg.openxr_handles_provider`
- `IsaacTeleopDevice.set_anchor_world_matrix_provider()`
- `session_lifecycle.py` support for renderer-owned handles
- best-effort Kit anchor/settings behavior for kitless runs

Scripts:

- `teleop_se3_agent.py` uses `KitlessTeleopLauncher` for kitless runs.
- `record_demos.py` uses the same launcher path.
- `teleop_replay_agent.py` only shares CloudXR profile resolution in this
  commit.

Newton visualizer:

- `isaaclab_visualizers.newton.teleop_runtime` implements the runtime policy.
- `isaaclab_visualizers.newton.newton_openxr` bridges Newton rendering to native
  OpenXR.
- `NewtonVisualizer.configure_openxr_teleop(env_cfg=..., env=...)` creates the
  renderer-owned OpenXR teleop session.
- `NewtonVisualizerCfg` carries the OpenXR settings.

GR1T2 task:

- `isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg` owns task-specific
  kitless behavior.
- `isaaclab_tasks.contrib.pick_place.assets.GR1T2_fourier_hand_6dof_kinematics.urdf`
  is the packaged kinematics-only Pink IK model used by default.
- `pink_task_space_actions.py` preserves hand joint order for the IsaacTeleop
  action layout.

Newton package:

- `newton/native/openxr` builds `libnewton_openxr_native.so`.
- `newton.openxr.NewtonOpenXRPresenter` and `NewtonOpenXRContext` expose the
  small Python-facing native presenter surface.
- `newton.openxr.OpenXRFrameRenderer` renders OpenXR eye views into frame data
  consumed by the presenter.

### Renderer-Owned OpenXR Runtime Policy

`RendererOpenXRTeleopRuntime` is intentionally policy-oriented. A renderer
runtime reports:

- `visualizer_type`
- `display_name`
- `cloudxr_label`
- `loader_required`
- `fallback_presets`
- `render_interval`
- whether its visualizer was requested
- whether its physics backend was requested
- how to apply backend defaults
- how to choose visualization device defaults
- how to log/debug backend selection

Newton's policy is:

- visualizer type: `newton`
- fallback presets: `newton`, `newton_mjwarp`, `newton_renderer`
- render interval: `3`
- loader required: `True`
- CloudXR label: renderer-owned Newton XR

Newton physics is enabled only for Newton physics presets or an existing
`NewtonCfg`. Otherwise Newton can still render an OvPhysX CPU simulation.

### Newton OpenXR Details

The Newton OpenXR bridge:

- requires the native OpenXR library to be built;
- creates `NewtonOpenXRContext` with `NewtonOpenXRSettings`;
- asks native code for IsaacTeleop-compatible handles;
- defers IsaacTeleop session creation by returning `None` until native OpenXR is
  ready;
- reads OpenXR eye poses each frame;
- renders through `OpenXRFrameRenderer`;
- prepares color with a display LUT to better match the Newton desktop viewer;
- submits through CUDA/Vulkan by default;
- can use host staging for debugging;
- mirrors submitted XR color in the Newton desktop viewer by default.

Newton's desktop renderer is OpenGL. The XR path is not using OpenGL as the
OpenXR renderer. The XR image is produced as rendered frame data, submitted to
native OpenXR through the selected submit path, and then optionally mirrored in
the desktop viewer.

### Newton Visualizer OpenXR Defaults

Important `NewtonVisualizerCfg` defaults:

| Field | Default | Meaning |
| --- | --- | --- |
| `openxr_submit_mode` | `"cuda-vulkan"` | Native submit mode. |
| `openxr_swapchain_scale` | `0.5` | OpenXR resolution scale. |
| `openxr_scene_depth` | `False` | Use far depth by default for client UI. |
| `openxr_desktop_mirror` | `True` | Show submitted XR color in the desktop viewer. |
| `openxr_desktop_rendering` | `False` | Avoid a second desktop render during XR. |
| `openxr_app_side_pqw` | `False` | Keep disabled in the redistributable build; no proprietary PQW extension header is vendored. |
| `openxr_profile` | `False` | Print frame timing diagnostics when enabled. |

The branch avoids normal-operation IsaacLab environment toggles for this path.
CloudXR behavior is controlled with CLI/profile files, and renderer tuning is
represented by visualizer config fields.

### CloudXR Profiles

`IsaacTeleopCfg` maps CloudXR shorthands to `.env` files:

- `cloudxrjs` -> `cloudxrjs-cloudxr.env`
- `avp` -> `avp-cloudxr.env`
- `newton` -> `newton-openxr-cloudxr.env`

`newton-openxr-cloudxr.env` currently contains CloudXR `NV_*` settings. Those
are runtime profile values, not IsaacLab control-plane switches.

### GR1T2 Kitless Details

GR1T2 kitless teleop currently needs extra task-owned policy because the Kit
path previously hid several assumptions.

Task-owned behavior includes:

- packaged kitless Pink IK URDF default, with optional URDF/mesh-root override
  fields;
- explicit Newton USD import options for `PackingTable` and `Object`;
- Newton-friendly actuator and Pink IK tuning when Newton physics is active;
- wrist target ramping from current robot wrist poses;
- unsafe target detection and concise diagnostics;
- a Pink IK guard for unstable kitless Newton targets;
- explicit IsaacTeleop action order:
  `[left_wrist(7), right_wrist(7), left_hand_joints(11), right_hand_joints(11)]`.

The task does not search machine-local directories for GR1T2 assets. Standard
GR1T2 kitless teleop uses the packaged kinematics-only URDF; missing packaged
assets or invalid explicit overrides produce a clear error.

### Problems Encountered During Bring-Up

This is the historical context that explains why some choices exist:

- OpenXR instance creation failed until the kitless path used the packaged
  `isaacteleop` OpenXR loader instead of assuming Kit or system loader state.
- The client initially showed no image, then blue/magenta debug frames, before
  color/depth submission and CloudXR launch order were corrected.
- Scene depth made the WebXR UI hard to use, so default depth submission moved
  to explicit far depth.
- The headset camera height was too high until the OpenXR anchor was tied to the
  robot pelvis/root plus the configured camera offset.
- Robot wrists did not follow IsaacTeleop wrists until the anchor provider,
  action order, and wrist target conditioning were aligned.
- GR1T2 upper body instability required task-owned wrist ramping and Newton IK
  safety guards.
- The steering wheel/table/bin were missing or collision-only until Newton USD
  import options and task-owned asset policy were added.
- Performance was poor with host-staged frame copies and duplicate desktop
  rendering; CUDA/Vulkan submit, reduced swapchain scale, render interval `3`,
  and desktop mirroring improved interactive performance.
- OvPhysX CPU needed to be treated as a separate physics path from Newton
  physics, while still allowing Newton to render on CUDA.

### Boundaries

This branch adds Newton as the first renderer-owned OpenXR implementation. It
does not add ovrtx/ovxr yet.

The IsaacLab branch does not build the Newton native OpenXR library. Build and
install the Newton branch first, then run the IsaacLab commands above.

`record_demos.py` supports the kitless launcher path. `teleop_replay_agent.py`
only gets shared CloudXR shorthand resolution in this commit.

Scene depth is present as a config field but disabled by default. App-side
PQW remains disabled in the redistributable build because the proprietary
extension header is not vendored.
