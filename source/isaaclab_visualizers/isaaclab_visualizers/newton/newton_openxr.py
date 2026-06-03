# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR teleoperation support for the Newton visualizer."""

from __future__ import annotations

import contextlib
import logging
import time

import numpy as np
import warp as wp

logger = logging.getLogger(__name__)

_OPENXR_RETRY_INTERVAL_S = 2.0
_DEFAULT_PELVIS_CAMERA_OFFSET = (0.0, 0.0, 0.52)
_PELVIS_ANCHOR_BODY_NAMES = ("pelvis", "pelvis_link", "base_link")
_OPENXR_VIEWER_EXPOSURE = 1.6
_OPENXR_SKY_CLEAR_COLOR = 0xFF462B18
_OPENXR_MIRROR_IMAGE_NAME = "OpenXR Mirror"


def _normalize_openxr_submit_mode(value: object) -> str:
    """Return a normalized OVXR-style OpenXR submit mode string."""
    return str(value).strip().lower().replace("_", "-")


def _openxr_uses_cuda_source(submit_mode: str) -> bool:
    """Return whether IsaacLab should submit CUDA source images to Newton OpenXR."""
    return submit_mode not in {"host", "host-staging", "cpu"}


def _openxr_uses_cuda_vulkan_submit(submit_mode: str) -> bool:
    """Return whether Newton OpenXR should submit through CUDA/Vulkan interop."""
    return submit_mode == "cuda-vulkan"


@wp.kernel
def _prepare_openxr_color_rgba_kernel(
    color_image: wp.array(dtype=wp.uint32, ndim=4),
    display_lut: wp.array(dtype=wp.uint8),
    out_rgba: wp.array(dtype=wp.uint8, ndim=4),
):
    view_id, y, x = wp.tid()
    color = color_image[0, view_id, y, x]
    out_rgba[view_id, y, x, 0] = display_lut[wp.int32((color >> wp.uint32(0)) & wp.uint32(0xFF))]
    out_rgba[view_id, y, x, 1] = display_lut[wp.int32((color >> wp.uint32(8)) & wp.uint32(0xFF))]
    out_rgba[view_id, y, x, 2] = display_lut[wp.int32((color >> wp.uint32(16)) & wp.uint32(0xFF))]
    out_rgba[view_id, y, x, 3] = wp.uint8(255)


@wp.kernel
def _fill_openxr_far_depth_kernel(depth_image: wp.array(dtype=wp.float32, ndim=3), far_z: float):
    view_id, y, x = wp.tid()
    depth_image[view_id, y, x] = far_z


@wp.kernel
def _prepare_openxr_reversed_depth_kernel(
    depth_meters: wp.array(dtype=wp.float32, ndim=4),
    out_depth: wp.array(dtype=wp.float32, ndim=3),
    near_z: float,
    far_z: float,
):
    view_id, y, x = wp.tid()
    depth = depth_meters[0, view_id, y, x]
    value = wp.float32(0.0)
    if depth > 0.0:
        if depth <= near_z:
            value = wp.float32(1.0)
        elif depth < far_z:
            value = (far_z - depth) / wp.max(far_z - near_z, 1.0e-6)
    out_depth[view_id, y, x] = wp.min(wp.max(value, 0.0), 1.0)


@wp.kernel
def _fill_openxr_depth_value_kernel(depth_image: wp.array(dtype=wp.float32, ndim=3), value: float):
    view_id, y, x = wp.tid()
    depth_image[view_id, y, x] = value


def _quat_xyzw_to_matrix(quat_xyzw: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Convert an ``xyzw`` quaternion to a 3x3 rotation matrix."""
    x, y, z, w = (float(value) for value in quat_xyzw)
    norm = (x * x + y * y + z * z + w * w) ** 0.5
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to an ``xyzw`` quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = (trace + 1.0) ** 0.5 * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = (1.0 + m[0, 0] - m[1, 1] - m[2, 2]) ** 0.5 * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = (1.0 + m[1, 1] - m[0, 0] - m[2, 2]) ** 0.5 * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = (1.0 + m[2, 2] - m[0, 0] - m[1, 1]) ** 0.5 * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1.0e-12)
    return tuple(float(value) for value in quat)


def _openxr_to_world_matrix(env_cfg) -> np.ndarray:
    """Return the static OpenXR-reference to Isaac world transform."""
    xr_cfg = getattr(getattr(env_cfg, "isaac_teleop", None), "xr_cfg", None)
    anchor_pos = getattr(xr_cfg, "anchor_pos", (0.0, 0.0, 0.0))
    anchor_rot = getattr(xr_cfg, "anchor_rot", (0.0, 0.0, 0.0, 1.0))
    oxr_to_usd_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_xyzw_to_matrix(anchor_rot) @ oxr_to_usd_rotation
    matrix[:3, 3] = np.asarray(anchor_pos, dtype=np.float64)
    return matrix


def _create_openxr_display_lut() -> np.ndarray:
    """Return a LUT matching Newton viewer exposure, tone mapping, and gamma."""
    color = np.linspace(0.0, 1.0, 256, dtype=np.float32) * _OPENXR_VIEWER_EXPOSURE
    tone_mapped = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14)
    display = np.power(np.clip(tone_mapped, 0.0, 1.0), 1.0 / 2.2)
    return np.rint(display * 255.0).astype(np.uint8)


def _kitless_newton_pelvis_camera_offset(xr_cfg) -> np.ndarray:
    """Return the Newton XR camera offset from the pelvis/root body [m]."""
    default_offset = np.asarray(_DEFAULT_PELVIS_CAMERA_OFFSET, dtype=np.float64)
    try:
        anchor_offset = np.asarray(getattr(xr_cfg, "anchor_pos", default_offset), dtype=np.float64)
    except (TypeError, ValueError):
        logger.warning("Invalid XrCfg.anchor_pos for kitless Newton XR; using default pelvis camera offset.")
        return default_offset
    if anchor_offset.shape != (3,):
        logger.warning("Invalid XrCfg.anchor_pos shape for kitless Newton XR; using default pelvis camera offset.")
        return default_offset
    if np.allclose(anchor_offset, 0.0):
        return default_offset
    return anchor_offset


def _newton_pelvis_openxr_to_world_matrix(env, xr_cfg) -> tuple[np.ndarray, str, np.ndarray] | None:
    """Return an OpenXR-reference transform from the robot pelvis/root body."""
    try:
        robot = env.scene["robot"]
        body_names = list(getattr(robot, "body_names", None) or getattr(robot.data, "body_names", []))
        pose_w = _as_torch_tensor(robot.data.body_link_pose_w)[0].detach().cpu().numpy()
    except Exception as exc:
        logger.debug("Unable to read robot body poses for Newton XR pelvis anchoring: %s", exc)
        return None

    candidate_names: list[str] = []
    anchor_prim_path = getattr(xr_cfg, "anchor_prim_path", None)
    if anchor_prim_path:
        candidate_names.append(str(anchor_prim_path).rstrip("/").rsplit("/", 1)[-1])
    candidate_names.extend(_PELVIS_ANCHOR_BODY_NAMES)

    selected_index = None
    selected_name = ""
    for candidate_name in candidate_names:
        lower_candidate = str(candidate_name).lower()
        for index, body_name in enumerate(body_names):
            lower_body_name = str(body_name).lower()
            if lower_body_name == lower_candidate or lower_body_name.endswith(f"_{lower_candidate}"):
                selected_index = index
                selected_name = str(body_name)
                break
        if selected_index is not None:
            break

    if selected_index is None:
        logger.debug("Unable to find a pelvis/root body for Newton XR anchoring in %s.", body_names)
        return None

    pelvis_pos = np.asarray(pose_w[selected_index, :3], dtype=np.float64)
    anchor_offset = _kitless_newton_pelvis_camera_offset(xr_cfg)
    anchor_rot = getattr(xr_cfg, "anchor_rot", (0.0, 0.0, 0.0, 1.0))
    oxr_to_usd_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = _quat_xyzw_to_matrix(anchor_rot) @ oxr_to_usd_rotation
    matrix[:3, 3] = pelvis_pos + anchor_offset
    return matrix, selected_name, anchor_offset


def _as_torch_tensor(value):
    """Return a torch tensor from Isaac Lab or Newton proxy arrays."""
    return value.torch if hasattr(value, "torch") else value


def _as_float3(value) -> np.ndarray:
    """Convert a vector-like object to a 3D float array."""
    return np.array([float(value[0]), float(value[1]), float(value[2])], dtype=np.float64)


def _normalize_vector(value: np.ndarray) -> np.ndarray:
    """Return a normalized 3D vector, or raise when the vector is degenerate."""
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("zero-length vector")
    return value / norm


def _newton_visualizer_openxr_to_world_matrix(visualizer, *, yaw_only: bool = False) -> np.ndarray | None:
    """Return an OpenXR-reference to world transform from the Newton visualizer camera."""
    viewer = getattr(visualizer, "_viewer", None)
    camera = getattr(viewer, "camera", None)
    if camera is None:
        return None

    try:
        position = _as_float3(camera.pos)
        right = _normalize_vector(_as_float3(camera.get_right()))
        up = _normalize_vector(_as_float3(camera.get_up()))
        front = _normalize_vector(_as_float3(camera.get_front()))
    except Exception as exc:
        logger.debug("Unable to read Newton visualizer camera for XR anchoring: %s", exc)
        return None

    if yaw_only:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        front_flat = front.copy()
        front_flat[2] = 0.0
        try:
            front_flat = _normalize_vector(front_flat)
            right = _normalize_vector(np.cross(front_flat, world_up))
            up = world_up
            front = front_flat
        except ValueError as exc:
            logger.debug("Unable to flatten Newton visualizer camera yaw for XR anchoring: %s", exc)
            return None

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, up, -front))
    matrix[:3, 3] = position
    return matrix


class NewtonOpenXRTeleopSession:
    """Renderer-owned OpenXR session bridge used by IsaacTeleop."""

    def __init__(self, *, visualizer, env_cfg, env) -> None:
        """Initialize the Newton OpenXR teleop bridge.

        Args:
            visualizer: Newton visualizer instance that owns the render loop.
            env_cfg: Isaac Lab environment configuration.
            env: Isaac Lab environment instance.
        """
        try:
            from newton.openxr import NewtonOpenXRContext, NewtonOpenXRSettings, OpenXRNativeError
        except ImportError as exc:
            raise RuntimeError(
                "Kitless XR with --visualizer newton requires a Newton package that exposes "
                "newton.openxr.NewtonOpenXRContext."
            ) from exc

        if not NewtonOpenXRContext.is_library_built():
            raise RuntimeError(
                "Newton native OpenXR library is not built or installed. Build Newton native OpenXR target and "
                "reinstall the Newton package into this IsaacLab environment."
            )

        teleop_cfg = getattr(env_cfg, "isaac_teleop", None)
        visualizer_cfg = getattr(visualizer, "cfg", None)
        settings = NewtonOpenXRSettings(
            openxr_submit_mode=getattr(visualizer_cfg, "openxr_submit_mode", "mixed"),
            swapchain_scale=getattr(visualizer_cfg, "openxr_swapchain_scale", 0.5),
            max_view_width=getattr(visualizer_cfg, "openxr_max_view_width", 0),
            max_view_height=getattr(visualizer_cfg, "openxr_max_view_height", 0),
        )
        self._submit_scene_depth = bool(getattr(visualizer_cfg, "openxr_scene_depth", False))
        print(
            "Newton OpenXR settings: "
            f"submit_mode={getattr(visualizer_cfg, 'openxr_submit_mode', 'mixed')}, "
            f"swapchain_scale={getattr(visualizer_cfg, 'openxr_swapchain_scale', 0.5)}, "
            f"desktop_mirror={getattr(visualizer_cfg, 'openxr_desktop_mirror', True)}, "
            f"scene_depth={self._submit_scene_depth}.",
            flush=True,
        )
        self._context = NewtonOpenXRContext(
            application_name=getattr(teleop_cfg, "app_name", "IsaacLabTeleop"),
            settings=settings,
        )
        self._context.initialize()
        self._openxr_native_error_type = OpenXRNativeError
        self._frame_bridge = _NewtonOpenXRFrameBridge(env_cfg, env=env, visualizer=visualizer)
        self._logged_deferred = False
        self._next_retry_time = 0.0

    def openxr_handles_provider(self) -> object | None:
        """Return renderer-owned OpenXR handles for IsaacTeleop, or ``None`` until ready."""
        now = time.monotonic()
        if not self._context.is_ready and now < self._next_retry_time:
            return None
        try:
            self._context.ensure_ready()
            self._logged_deferred = False
            self._next_retry_time = 0.0
            return self._context.isaacteleop_handles()
        except self._openxr_native_error_type as exc:
            self._next_retry_time = now + _OPENXR_RETRY_INTERVAL_S
            if not self._logged_deferred:
                logger.warning(
                    "Newton OpenXR handles are not ready yet; CloudXR/OpenXR runtime may still be starting: %s",
                    exc,
                )
                self._logged_deferred = True
            return None

    def anchor_world_matrix(self) -> np.ndarray:
        """Return the OpenXR-reference to world transform for retargeting inputs."""
        return self._frame_bridge.teleop_anchor_world_matrix()

    def step(self, state) -> bool:
        """Pump one renderer-owned OpenXR frame if the runtime is ready."""
        if not self._context.is_ready:
            self.openxr_handles_provider()
            if not self._context.is_ready:
                return True
        return self._pump_frame(state)

    def close(self) -> None:
        """Release the native OpenXR context."""
        with contextlib.suppress(Exception):
            self._context.shutdown()

    def _pump_frame(self, state) -> bool:
        """Pump a frame through the Newton OVXR-style context facade."""
        if state is None:
            return True
        try:
            should_continue = self._context.pump(
                lambda frame: self._frame_bridge.render_submission(
                    frame,
                    state,
                    submit_scene_depth=self._submit_scene_depth,
                )
            )
            if not should_continue:
                logger.info("Newton OpenXR runtime requested shutdown.")
            return should_continue
        except Exception:
            logger.exception("Newton OpenXR frame pump failed.")
            return False


class _NewtonOpenXRFrameBridge:
    """Render Newton scene images from OpenXR headset eye poses."""

    def __init__(self, env_cfg, *, env=None, visualizer=None) -> None:
        self._env = env
        self._visualizer = visualizer
        visualizer_cfg = getattr(visualizer, "cfg", None)
        self._openxr_submit_mode = _normalize_openxr_submit_mode(getattr(visualizer_cfg, "openxr_submit_mode", "mixed"))
        self._openxr_uses_cuda_source = _openxr_uses_cuda_source(self._openxr_submit_mode)
        self._openxr_uses_cuda_vulkan_submit = _openxr_uses_cuda_vulkan_submit(self._openxr_submit_mode)
        self._app_side_pqw_enabled = bool(getattr(visualizer_cfg, "openxr_app_side_pqw", False))
        self._desktop_mirror_enabled = bool(getattr(visualizer_cfg, "openxr_desktop_mirror", True))
        self._fallback_openxr_to_world = _openxr_to_world_matrix(env_cfg)
        xr_cfg = getattr(getattr(env_cfg, "isaac_teleop", None), "xr_cfg", None)
        self._near_z = float(getattr(xr_cfg, "near_plane", 0.15))
        self._far_z = float(getattr(visualizer_cfg, "openxr_far_z", 1000.0))
        self._sensor = None
        self._renderer = None
        self._color_image = None
        self._ray_depth_image = None
        self._forward_depth_image = None
        self._openxr_color_rgba_device = None
        self._openxr_far_depth_device = None
        self._openxr_normalized_depth_device = None
        self._openxr_display_lut_device = None
        self._size: tuple[int, int, int] | None = None
        self._logged_first_frame = False
        self._logged_far_depth = False
        self._xr_cfg = xr_cfg
        self._pelvis_openxr_to_world: np.ndarray | None = None
        self._pelvis_anchor_body_name: str | None = None
        self._pelvis_anchor_offset: np.ndarray | None = None
        self._logged_pelvis_anchor = False
        self._openxr_display_lut = _create_openxr_display_lut()
        self._logged_color_transfer = False
        self._logged_visualizer_anchor = False
        self._logged_teleop_anchor = False
        self._logged_teleop_recenter = False
        self._logged_pqw_foveation = False
        self._logged_regular_projection = False
        self._logged_submit_stats = False
        self._openxr_anchor_position: np.ndarray | None = None
        self._logged_openxr_recenter = False
        self._logged_desktop_mirror = False
        self._logged_desktop_mirror_error = False
        self._profile_enabled = bool(getattr(visualizer_cfg, "openxr_profile", False))
        self._profile_interval = max(int(getattr(visualizer_cfg, "openxr_profile_interval", 60)), 1)
        self._profile_samples = 0
        self._profile_totals: dict[str, float] = {}

    def _pelvis_openxr_to_world_matrix(self) -> np.ndarray | None:
        """Return the cached initial pelvis/root-body OpenXR transform, if available."""
        if self._pelvis_openxr_to_world is not None:
            return self._pelvis_openxr_to_world
        if self._env is None:
            return None

        pelvis_anchor = _newton_pelvis_openxr_to_world_matrix(self._env, self._xr_cfg)
        if pelvis_anchor is None:
            return None

        matrix, body_name, offset = pelvis_anchor
        self._pelvis_openxr_to_world = matrix
        self._pelvis_anchor_body_name = body_name
        self._pelvis_anchor_offset = offset
        return matrix

    def openxr_to_world_matrix(self) -> np.ndarray:
        """Return the active OpenXR-reference to world transform."""
        matrix = self._pelvis_openxr_to_world_matrix()
        if matrix is not None:
            if not self._logged_pelvis_anchor:
                offset = np.array2string(self._pelvis_anchor_offset, precision=3, suppress_small=True)
                anchor_world = np.array2string(matrix[:3, 3], precision=3, suppress_small=True)
                print(
                    "Newton OpenXR headset anchor starts from robot pelvis body "
                    f"{self._pelvis_anchor_body_name} with camera offset {offset} "
                    f"(anchor_world={anchor_world})."
                )
                self._logged_pelvis_anchor = True
            return matrix

        matrix = _newton_visualizer_openxr_to_world_matrix(self._visualizer)
        if matrix is None:
            return self._fallback_openxr_to_world
        if not self._logged_visualizer_anchor:
            print("Newton OpenXR headset anchor follows the Newton visualizer camera view.")
            self._logged_visualizer_anchor = True
        return matrix

    def _recentered_input_openxr_to_world_matrix(self) -> np.ndarray:
        """Return the OpenXR transform that maps raw runtime input poses into the recentered world."""
        matrix = self.openxr_to_world_matrix().copy()
        if self._openxr_anchor_position is None:
            return matrix

        matrix[:3, 3] = matrix[:3, 3] - matrix[:3, :3] @ self._openxr_anchor_position
        if not self._logged_teleop_recenter:
            local_center = np.array2string(self._openxr_anchor_position, precision=3, suppress_small=True)
            anchor_world = np.array2string(matrix[:3, 3], precision=3, suppress_small=True)
            print(
                "Newton IsaacTeleop input anchor recenters raw OpenXR poses by the initial headset center "
                f"{local_center} (input_anchor_world={anchor_world})."
            )
            self._logged_teleop_recenter = True
        return matrix

    def teleop_anchor_world_matrix(self) -> np.ndarray:
        """Return the OpenXR-reference to world transform for retargeting inputs."""
        if not self._logged_teleop_anchor:
            print("Newton IsaacTeleop anchor uses the active Newton XR camera anchor for retargeting inputs.")
            self._logged_teleop_anchor = True
        return self._recentered_input_openxr_to_world_matrix().astype(np.float32, copy=False)

    def _openxr_origin_relative_position(self, view_position: tuple[float, float, float]) -> np.ndarray:
        """Return a headset position relative to the first located OpenXR headset center."""
        position = np.asarray(view_position, dtype=np.float64)
        if self._openxr_anchor_position is None:
            return position
        return position - self._openxr_anchor_position

    def _ensure_openxr_headset_recenter(self, views: list[object]) -> None:
        """Map the first located OpenXR headset center onto the active Newton camera anchor."""
        if self._openxr_anchor_position is not None:
            return
        positions = [np.asarray(view.position, dtype=np.float64) for view in views]
        if not positions:
            return
        self._openxr_anchor_position = np.mean(np.stack(positions, axis=0), axis=0)
        if not self._logged_openxr_recenter:
            local_center = np.array2string(self._openxr_anchor_position, precision=3, suppress_small=True)
            print(
                "Newton OpenXR recentered headset local origin onto the active Newton camera anchor "
                f"(initial_headset_center={local_center})."
            )
            self._logged_openxr_recenter = True

    def render_submission(self, frame: object, state, *, submit_scene_depth: bool = True) -> object | None:
        """Render Newton scene images and return an OpenXR submit descriptor."""
        views = frame.views
        if not views:
            return None

        width, height, view_count = frame.width, frame.height, frame.view_count
        if len(views) != view_count:
            logger.debug("OpenXR located %d views but swapchain has %d views", len(views), view_count)
            return None

        profile_timings: dict[str, float] = {}
        profile_total_start = time.perf_counter() if self._profile_enabled else 0.0
        profile_start = profile_total_start
        self._ensure_renderer(width, height, view_count)
        if self._profile_enabled:
            profile_timings["ensure"] = time.perf_counter() - profile_start
        if self._renderer is None:
            return None

        profile_start = time.perf_counter() if self._profile_enabled else 0.0
        self._ensure_openxr_headset_recenter(views)
        transformed_views = self._transform_views(views)
        if self._profile_enabled:
            profile_timings["views"] = time.perf_counter() - profile_start
        import newton
        from isaaclab_newton.physics import NewtonManager

        profile_start = time.perf_counter() if self._profile_enabled else 0.0
        model = NewtonManager.get_model()
        if model is not None and model.shape_count > 0:
            if model.bvh_shapes is None:
                newton.geometry.build_bvh_shape(model, state)
            else:
                newton.geometry.refit_bvh_shape(model, state)
        if self._profile_enabled:
            profile_timings["bvh"] = time.perf_counter() - profile_start

        profile_start = time.perf_counter() if self._profile_enabled else 0.0
        self._renderer.render(
            state,
            transformed_views,
            color_image=self._color_image,
            depth_image=self._ray_depth_image if submit_scene_depth else None,
            clear_data=newton.sensors.SensorTiledCamera.ClearData(clear_color=_OPENXR_SKY_CLEAR_COLOR, clear_depth=0.0),
        )
        if self._profile_enabled:
            wp.synchronize_device(self._color_image.device)
            profile_timings["render"] = time.perf_counter() - profile_start

        profile_start = time.perf_counter() if self._profile_enabled else 0.0
        pqw = self._pqw_array_for_submission(transformed_views, width, height)
        cuda_submit_desc = self._make_cuda_source_submit_desc(
            width=width,
            height=height,
            view_count=view_count,
            submit_scene_depth=submit_scene_depth,
            pqw=pqw,
        )
        if cuda_submit_desc is not None:
            if self._profile_enabled:
                profile_timings["submit"] = time.perf_counter() - profile_start
                profile_timings["total"] = time.perf_counter() - profile_total_start
                self._record_profile_sample(profile_timings, width, height, view_count)
            return cuda_submit_desc

        depth_meters = None
        if submit_scene_depth:
            self._forward_depth_image = self._sensor.utils.convert_ray_depth_to_forward_depth(
                self._ray_depth_image,
                self._renderer.camera_transforms,
                self._renderer.camera_rays,
                self._forward_depth_image,
            )
            depth_meters = np.ascontiguousarray(self._forward_depth_image.numpy()[0, :view_count], dtype=np.float32)
        else:
            depth_meters = np.full((view_count, height, width), self._far_z, dtype=np.float32)
            if not self._logged_far_depth:
                print(
                    "Newton OpenXR is submitting explicit far depth while scene depth rendering is disabled "
                    "so client UI remains usable.",
                    flush=True,
                )
                self._logged_far_depth = True

        packed_color = np.ascontiguousarray(self._color_image.numpy()[0, :view_count])
        color_rgba = packed_color.view(np.uint8).reshape(view_count, height, width, 4)
        color_rgba[..., 3] = 255
        color_rgba = self._convert_color_rgba_for_openxr(color_rgba)
        self._log_desktop_mirror(color_rgba)
        if not self._logged_submit_stats:
            depth_desc = (
                "none"
                if depth_meters is None
                else (
                    f"shape={depth_meters.shape}, range=({float(np.nanmin(depth_meters)):.3f}, "
                    f"{float(np.nanmax(depth_meters)):.3f})"
                )
            )
            print(
                "Newton OpenXR staging frame images: "
                f"color_shape={color_rgba.shape}, color_range=({int(color_rgba.min())}, {int(color_rgba.max())}), "
                f"alpha_range=({int(color_rgba[..., 3].min())}, {int(color_rgba[..., 3].max())}), "
                f"depth={depth_desc}",
                flush=True,
            )
        from newton.openxr import OpenXRFrameSubmitDesc

        submit_desc = OpenXRFrameSubmitDesc(
            openxr_submit_mode=self._openxr_submit_mode,
            color_rgba=color_rgba,
            depth_meters=depth_meters,
            pqw=pqw,
            near_z=self._near_z,
            far_z=self._far_z,
        )
        if not self._logged_submit_stats:
            print("Newton OpenXR frame submit descriptor prepared.", flush=True)
            self._logged_submit_stats = True
        if not self._logged_first_frame:
            print(
                f"Newton OpenXR submitted scene color/depth from headset pose ({view_count} views, {width}x{height})."
            )
            self._logged_first_frame = True
        if self._profile_enabled:
            profile_timings["submit"] = time.perf_counter() - profile_start
            profile_timings["total"] = time.perf_counter() - profile_total_start
            self._record_profile_sample(profile_timings, width, height, view_count)
        return submit_desc

    def _can_submit_cuda_source(self) -> bool:
        """Return whether this frame can use native CUDA source staging."""
        if not self._openxr_uses_cuda_source:
            return False
        if self._color_image is None or self._openxr_color_rgba_device is None:
            return False
        return bool(getattr(getattr(self._color_image, "device", None), "is_cuda", False))

    def _make_cuda_source_submit_desc(
        self,
        *,
        width: int,
        height: int,
        view_count: int,
        submit_scene_depth: bool,
        pqw: np.ndarray | None,
    ) -> object | None:
        """Return a CUDA-source submit descriptor, if the frame can use one."""
        if not self._can_submit_cuda_source():
            return None

        wp.launch(
            _prepare_openxr_color_rgba_kernel,
            dim=(view_count, height, width),
            inputs=[self._color_image, self._openxr_display_lut_device, self._openxr_color_rgba_device],
            device=self._color_image.device,
        )

        depth_image = None
        cuda_vulkan_submit = self._openxr_uses_cuda_vulkan_submit
        if submit_scene_depth:
            self._forward_depth_image = self._sensor.utils.convert_ray_depth_to_forward_depth(
                self._ray_depth_image,
                self._renderer.camera_transforms,
                self._renderer.camera_rays,
                self._forward_depth_image,
            )
            if cuda_vulkan_submit:
                wp.launch(
                    _prepare_openxr_reversed_depth_kernel,
                    dim=(view_count, height, width),
                    inputs=[
                        self._forward_depth_image,
                        self._openxr_normalized_depth_device,
                        self._near_z,
                        self._far_z,
                    ],
                    device=self._openxr_normalized_depth_device.device,
                )
                depth_image = self._openxr_normalized_depth_device
            else:
                depth_image = self._forward_depth_image
        else:
            depth_fill_value = 0.0 if cuda_vulkan_submit else self._far_z
            depth_image = self._openxr_normalized_depth_device if cuda_vulkan_submit else self._openxr_far_depth_device
            wp.launch(
                _fill_openxr_depth_value_kernel,
                dim=(view_count, height, width),
                inputs=[depth_image, depth_fill_value],
                device=depth_image.device,
            )
            if not self._logged_far_depth:
                print(
                    "Newton OpenXR is submitting explicit far depth while scene depth rendering is disabled "
                    "so client UI remains usable.",
                    flush=True,
                )
                self._logged_far_depth = True

        if not self._logged_submit_stats:
            depth_label = "depth_normalized_ptr" if cuda_vulkan_submit else "depth_meters_ptr"
            print(
                "Newton OpenXR staging CUDA frame images in native presenter: "
                f"color_ptr=0x{int(self._openxr_color_rgba_device.ptr):x}, "
                f"{depth_label}=0x{int(depth_image.ptr):x}, shape=({view_count}, {height}, {width}).",
                flush=True,
            )
        wp.synchronize_device(self._color_image.device)
        self._log_desktop_mirror(self._openxr_color_rgba_device)

        from newton.openxr import OpenXRFrameSubmitDesc

        submit_desc = OpenXRFrameSubmitDesc(
            openxr_submit_mode=self._openxr_submit_mode,
            color_rgba_cuda_ptr=int(self._openxr_color_rgba_device.ptr),
            color_width=width,
            color_height=height,
            view_count=view_count,
            depth_meters_cuda_ptr=int(depth_image.ptr),
            depth_width=width,
            depth_height=height,
            pqw=pqw,
            near_z=self._near_z,
            far_z=self._far_z,
        )
        if not self._logged_submit_stats:
            print("Newton OpenXR frame submit descriptor prepared.", flush=True)
            self._logged_submit_stats = True
        if not self._logged_first_frame:
            print(
                f"Newton OpenXR submitted CUDA scene color/depth from headset pose "
                f"({view_count} views, {width}x{height})."
            )
            self._logged_first_frame = True
        return submit_desc

    def _convert_color_rgba_for_openxr(self, color_rgba: np.ndarray) -> np.ndarray:
        """Apply Newton viewer-style display mapping for OpenXR swapchain submission."""
        color_rgba[..., :3] = self._openxr_display_lut[color_rgba[..., :3]]
        if not self._logged_color_transfer:
            print(
                "Newton OpenXR converted linear renderer color with viewer exposure/tone mapping "
                "for swapchain submission.",
                flush=True,
            )
            self._logged_color_transfer = True
        return color_rgba

    def _pqw_array_for_submission(self, views: list[object], width: int, height: int) -> np.ndarray | None:
        if not self._app_side_pqw_enabled:
            if not self._logged_regular_projection:
                print("Newton OpenXR app-side PQW foveation disabled; submitting regular projection frames.")
                self._logged_regular_projection = True
            return None

        pqws = [getattr(view, "pqw", None) for view in views]
        if not pqws or any(pqw is None for pqw in pqws):
            return None

        pqw_array = np.ascontiguousarray(pqws, dtype=np.uint16)
        if not (
            np.all(pqw_array[:, 0] > 0)
            and np.all(pqw_array[:, 1] > 0)
            and np.all(pqw_array[:, 4] > 0)
            and np.all(pqw_array[:, 5] > 0)
            and np.all(pqw_array[:, 6] > 0)
            and np.all(pqw_array[:, 7] > 0)
        ):
            return None

        if not self._logged_pqw_foveation:
            first = pqw_array[0]
            print(
                "Newton OpenXR is submitting app-side PQW foveated frames "
                f"(unwarped={int(first[0])}x{int(first[1])}, "
                f"pqw_warped={int(first[6])}x{int(first[7])}, render={width}x{height})."
            )
            self._logged_pqw_foveation = True
        return pqw_array

    def _log_desktop_mirror(self, image) -> None:
        """Mirror the submitted OpenXR color image in the desktop Newton viewer."""
        if not self._desktop_mirror_enabled:
            return

        viewer = getattr(self._visualizer, "_viewer", None)
        if viewer is None:
            return

        try:
            make_current = getattr(getattr(viewer, "renderer", None), "_make_current", None)
            if callable(make_current):
                make_current()
            viewer.log_image(_OPENXR_MIRROR_IMAGE_NAME, image)
            setattr(self._visualizer, "_openxr_mirror_ready", True)
            if not self._logged_desktop_mirror:
                print(
                    "Newton desktop viewer is mirroring submitted OpenXR color frames "
                    f"as {_OPENXR_MIRROR_IMAGE_NAME!r}."
                )
                self._logged_desktop_mirror = True
        except Exception as exc:
            if not self._logged_desktop_mirror_error:
                logger.warning("Newton OpenXR desktop mirror could not upload image: %s", exc)
                self._logged_desktop_mirror_error = True

    def _record_profile_sample(self, timings: dict[str, float], width: int, height: int, view_count: int) -> None:
        """Record and periodically print Newton OpenXR frame timing."""
        if not self._profile_enabled:
            return

        self._profile_samples += 1
        for name, elapsed_s in timings.items():
            self._profile_totals[name] = self._profile_totals.get(name, 0.0) + elapsed_s

        if self._profile_samples % self._profile_interval != 0:
            return

        parts = [
            f"{name}={1000.0 * self._profile_totals.get(name, 0.0) / self._profile_interval:.2f}ms"
            for name in ("ensure", "views", "bvh", "render", "submit", "total")
        ]
        print(
            "Newton OpenXR profile "
            f"{width}x{height}x{view_count} avg/{self._profile_interval} frames: " + ", ".join(parts),
            flush=True,
        )
        self._profile_totals.clear()

    def _ensure_renderer(self, width: int, height: int, view_count: int) -> None:
        if self._size == (width, height, view_count):
            return

        import newton
        from isaaclab_newton.physics import NewtonManager
        from newton.openxr import OpenXRFrameRenderer

        model = NewtonManager.get_model()
        if model is None:
            return

        sensor = newton.sensors.SensorTiledCamera(
            model,
            config=newton.sensors.SensorTiledCamera.RenderConfig(
                enable_textures=False,
                enable_ambient_lighting=True,
                max_distance=self._far_z,
            ),
            load_textures=False,
        )
        sensor.utils.create_default_light(enable_shadows=False)
        plane_shape_indices = np.flatnonzero(model.shape_type.numpy() == int(newton.GeoType.PLANE))
        if plane_shape_indices.size:
            sensor.utils.assign_checkerboard_material_to_shapes(
                plane_shape_indices,
                resolution=64,
                checker_size=32,
                color_0=0xFF462B18,
                color_1=0xFF704626,
                repeat=(0.5, 0.5),
            )
            print(
                f"Newton OpenXR applied blue checkerboard material to {plane_shape_indices.size} ground plane shape(s)."
            )
        self._sensor = sensor
        self._renderer = OpenXRFrameRenderer(sensor, width=width, height=height)
        self._color_image = sensor.utils.create_color_image_output(width, height, view_count)
        self._ray_depth_image = sensor.utils.create_depth_image_output(width, height, view_count)
        self._forward_depth_image = sensor.utils.create_depth_image_output(width, height, view_count)
        self._openxr_color_rgba_device = wp.empty((view_count, height, width, 4), dtype=wp.uint8, device=model.device)
        self._openxr_far_depth_device = wp.empty((view_count, height, width), dtype=wp.float32, device=model.device)
        self._openxr_normalized_depth_device = wp.empty(
            (view_count, height, width), dtype=wp.float32, device=model.device
        )
        self._openxr_display_lut_device = wp.array(self._openxr_display_lut, dtype=wp.uint8, device=model.device)
        self._size = (width, height, view_count)
        print(f"Newton OpenXR scene renderer initialized at {width}x{height} x {view_count} views.")
        logger.info("Newton OpenXR scene renderer initialized at %dx%d x %d views", width, height, view_count)

    def _transform_views(self, views: list[object]) -> list[object]:
        from newton.openxr import OpenXRView

        openxr_to_world = self.openxr_to_world_matrix()
        rotation = openxr_to_world[:3, :3]
        translation = openxr_to_world[:3, 3]
        transformed = []
        for view in views:
            position = rotation @ self._openxr_origin_relative_position(view.position) + translation
            orientation_matrix = rotation @ _quat_xyzw_to_matrix(view.orientation)
            transformed.append(
                OpenXRView(
                    position=tuple(float(value) for value in position),
                    orientation=_matrix_to_quat_xyzw(orientation_matrix),
                    fov=view.fov,
                    pqw=view.pqw if self._app_side_pqw_enabled else None,
                )
            )
        return transformed
