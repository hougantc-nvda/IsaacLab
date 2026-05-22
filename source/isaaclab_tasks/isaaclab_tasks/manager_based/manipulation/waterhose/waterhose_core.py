# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import site
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("PXR_WORK_THREAD_LIMIT", "1")

_ISAACLAB_ROOT = Path(__file__).resolve().parents[6]
_DEFAULT_WATERHOSE_ASSET_ROOT = "source/isaaclab_assets/data/WaterhoseDemo"
_DEFAULT_CABLE_USD_RELATIVE_PATH = "Waterhose/Cable008/curve/cable_SRA_curve03.usda"
_DEFAULT_CABLE_PRIMS = ("/World/cable001/curve_0", "/World/cable002/curve_0")
DEFER_NEWTON_IMPORT_ENV = "ISAACLAB_WATERHOSE_DEFER_NEWTON_IMPORT"
KIT_STATIC_CONTACT_PROXY_ENV = "ISAACLAB_WATERHOSE_KIT_STATIC_CONTACT_PROXY"


def default_waterhose_asset_root() -> Path:
    """Return the root folder for the unpacked waterhose demo asset bundle."""
    return Path(
        os.environ.get(
            "ISAACLAB_WATERHOSE_ASSET_ROOT",
            str(_ISAACLAB_ROOT / _DEFAULT_WATERHOSE_ASSET_ROOT),
        )
    ).expanduser()


@dataclass(frozen=True)
class CableUsdAsset:
    """One cable curve source USD and the BasisCurves prim to import from it."""

    usd_path: Path
    curve_prim_path: str


def make_waterhose_args(**overrides: Any) -> SimpleNamespace:
    """Create the namespace expected by the shared Newton waterhose builder."""
    defaults = dict(
        fps=100.0,
        max_steps=1400,
        num_envs=1,
        env_spacing=2.5,
        asset_root=default_waterhose_asset_root(),
        robot_urdf=None,
        scene_usd=None,
        cable_usds=None,
        cable_usd=None,
        cable_prims=None,
        cable_prim=None,
        hose_radius=0.003,
        gripper_drive_scale=2.0,
        grasp_friction=50.0,
        grasp_margin=0.001,
        grasp_contact_ke=5.0e4,
        sim_substeps=10,
        rigid_substeps=1,
        proxy_iterations=1,
        proxy_mass_scale=1.0,
        vbd_iterations=10,
        rigid_contact_max=100000,
        mujoco_iterations=20,
        mujoco_ls_iterations=10,
        mujoco_ls_parallel=True,
        mujoco_impratio=1000.0,
        mujoco_use_mujoco_contacts=True,
        robot_shape_margin=0.0,
        robot_shape_gap=0.005,
        robot_shape_ke=5.0e4,
        robot_shape_kd=5.0e2,
        robot_shape_mu=2.0,
        robot_joint_target_ke=45000.0,
        robot_joint_target_kd=4500.0,
        robot_joint_effort_limit=1000.0,
        robot_joint_armature=0.2,
        gripper_joint_target_ke=10000.0,
        gripper_joint_target_kd=1000.0,
        gripper_joint_effort_limit=100000.0,
        gripper_joint_armature=0.5,
        vbd_collide_substeps=5,
        vbd_default_contact_ke=1.0e5,
        vbd_default_contact_kd=5.0,
        vbd_default_contact_margin=0.001,
        vbd_solver_friction_epsilon=0.1,
        vbd_rigid_contact_buffer_size=2048,
        vbd_proxy_margin=0.001,
        vbd_cable_density=1200.0,
        vbd_cable_mu=0.8,
        vbd_cable_margin=0.0,
        vbd_cable_gap=0.001,
        vbd_static_margin=1.0e-4,
        vbd_static_gap=0.001,
        vbd_near_tip_mu=1.0e1,
        vbd_far_tip_mu=1.0e5,
        vbd_ground_mu=1.0e5,
        vbd_rigid_avbd_beta=1.0e5,
        vbd_rigid_contact_k_start=1.0e2,
        vbd_rigid_joint_linear_k_start=1.0e4,
        vbd_rigid_joint_angular_k_start=1.0e1,
        cable_stretch_stiffness=3.0e8,
        cable_stretch_damping=8.0e-2,
        cable_num_segments=60,
        cable_bend_rigidity=1.5e-1,
        cable_bend_damping=3.5e-1,
        disable_cuda_graph=False,
        log_interval=0,
        headless=False,
        livestream=-1,
        enable_cameras=False,
        xr=False,
        device="cuda:0",
        visualizer=["newton"],
        cpu=False,
        verbose=False,
        info=False,
        experience="",
        rendering_mode=None,
        kit_args="",
        anim_recording_enabled=False,
        anim_recording_start_time=0.0,
        anim_recording_stop_time=10.0,
        max_visible_envs=None,
    )
    defaults.update(overrides)
    if "grasp_margin" in overrides and "vbd_proxy_margin" not in overrides:
        defaults["vbd_proxy_margin"] = defaults["grasp_margin"]
    for key in ("asset_root", "robot_urdf", "scene_usd", "cable_usd"):
        if defaults[key] is not None:
            defaults[key] = Path(defaults[key]).expanduser()
    return SimpleNamespace(**defaults)


args_cli = make_waterhose_args()


def configure_waterhose_args(**overrides: Any) -> SimpleNamespace:
    """Update the process-local waterhose builder arguments."""
    global args_cli
    args_cli = make_waterhose_args(**overrides)
    return args_cli


def build_waterhose_scene(**overrides: Any):
    """Build the Newton waterhose model and coupled solver configuration."""
    cfg = make_waterhose_args(**overrides)
    import_newton_dependencies()
    scene_builder = WaterhoseSceneBuilder(cfg)
    builder, solver_cfg = scene_builder.build()
    return scene_builder, builder, solver_cfg


np = wp = newton = ik = sim_utils = None
SolverVBD = None
JointTargetMode = None
add_cable_from_usd_curve = None
CoupledProxyCfg = CoupledSolverCfg = CoupledSolverEntryCfg = None
MJWarpSolverCfg = NewtonCfg = NewtonCoupledManager = NewtonManager = NewtonSolverCfg = ProxyCouplingCfg = None

ROBOT_ENTRY = "mjc"
HOSE_ENTRY = "vbd"
ROBOT_PRIM_PATH = "/World/RBY1DF"
KIT_CABLE_CURVE_ROOT = "/World/WaterhoseCableCurves"
KIT_CAMERA_PRIM_PATH = "/OmniverseKit_Persp"
KIT_CAMERA_TRANSLATION = (0.19082, -0.09351, 1.39351)
KIT_CAMERA_ROTATION_XYZ_DEG = (47.78191, -0.0, -107.54558)

RIGHT_EE = "right_gripper_end_effector"
LEFT_EE = "left_gripper_end_effector"
TORSO = "torso_hip_yaw"
PROXY_BODY_NAMES = {
    "right_gripper_base",
    "right_gripper_camera_bracket",
    "right_gripper_leftfinger",
    "right_gripper_rightfinger",
    "left_gripper_base",
    "left_gripper_camera_bracket",
    "left_gripper_leftfinger",
    "left_gripper_rightfinger",
}
_ACTIVE_SCENE_BUILDER = None


def _requested_kit_visualizer() -> bool:
    if os.environ.get(DEFER_NEWTON_IMPORT_ENV) == "1" or os.environ.get(KIT_STATIC_CONTACT_PROXY_ENV) == "1":
        return True

    visualizer_flags = {"--visualizer", "--viz", "--vis"}
    argv = sys.argv[1:]
    for index, token in enumerate(argv):
        value = None
        if token in visualizer_flags and index + 1 < len(argv):
            value = argv[index + 1]
        elif any(token.startswith(f"{flag}=") for flag in visualizer_flags):
            value = token.split("=", 1)[1]
        if value is not None and "kit" in {part.strip().lower() for part in value.replace(",", " ").split()}:
            return True

    try:
        import carb  # noqa: PLC0415

        settings = carb.settings.get_settings()
        visualizer_types = settings.get_as_string("/isaaclab/visualizer/types") or ""
    except Exception:
        return False
    return "kit" in {part.strip().lower() for part in visualizer_types.replace(",", " ").split()}


def _active_scene_builder() -> WaterhoseSceneBuilder:
    if _ACTIVE_SCENE_BUILDER is None:
        raise RuntimeError("Waterhose Newton view was configured before the scene builder was initialized.")
    return _ACTIVE_SCENE_BUILDER


def configure_mujoco_view(view) -> None:
    """Restrict the MuJoCo view to the robot prefix in the shared model."""
    scene_builder = _active_scene_builder()
    view.body_count = scene_builder._mujoco_body_count
    view.shape_count = scene_builder._mujoco_shape_count
    view.joint_count = scene_builder._mujoco_joint_count
    view.joint_coord_count = scene_builder._mujoco_joint_coord_count
    view.joint_dof_count = scene_builder._mujoco_joint_dof_count
    view.articulation_count = scene_builder._mujoco_articulation_count


def configure_vbd_view(view) -> None:
    """Match the reference script's gripper-finger proxy contact material."""
    scene_builder = _active_scene_builder()
    if not scene_builder.proxy_shape_ids:
        return
    proxy_shape_ids = np.asarray(scene_builder.proxy_shape_ids, dtype=np.int32)
    model = view.parent
    for attr, value in (
        ("shape_material_ke", float(scene_builder.cfg.grasp_contact_ke)),
        ("shape_material_kd", float(scene_builder.cfg.vbd_default_contact_kd)),
        ("shape_material_mu", float(scene_builder.cfg.grasp_friction)),
        ("shape_margin", float(scene_builder.cfg.vbd_proxy_margin)),
    ):
        data = getattr(model, attr, None)
        if data is None:
            continue
        data_np = data.numpy().copy()
        data_np[proxy_shape_ids] = value
        setattr(view, attr, wp.array(data_np, dtype=wp.float32, device=model.device))


LEROBOT_INITIAL_STATE_22 = [
    0.0,
    0.872664213180542,
    -1.5707811117172241,
    0.6981245279312134,
    3.796982127823867e-06,
    0.0,
    0.3021828234195709,
    -0.013802030123770237,
    -0.09509921818971634,
    -2.2242417335510254,
    -0.7117632627487183,
    0.14113007485866547,
    0.5137608647346497,
    -0.4555884897708893,
    0.2500312626361847,
    -0.665743887424469,
    -1.3314952850341797,
    -0.19328542053699493,
    -0.5307496786117554,
    0.6565361022949219,
    0.09138019700534642,
    0.09098683297634125,
]


def import_newton_dependencies() -> None:
    """Import Newton modules before Isaac Lab sim touches USD physics bindings."""
    global np, wp, newton, ik, SolverVBD, JointTargetMode

    if newton is not None:
        return

    import newton as newton_module
    import numpy as np_module
    import warp as wp_module
    from newton import JointTargetMode as JointTargetModeClass
    from newton.solvers import SolverVBD as SolverVBDClass

    np = np_module
    wp = wp_module
    newton = newton_module
    SolverVBD = SolverVBDClass
    JointTargetMode = JointTargetModeClass


def prefer_active_python_site_packages() -> None:
    """Keep the active venv ahead of Kit extension prebundles for Newton deps."""
    candidate_paths: list[str] = []
    try:
        candidate_paths.extend(site.getsitepackages())
    except AttributeError:
        pass
    user_site = site.getusersitepackages()
    if user_site:
        candidate_paths.append(user_site)

    for path in reversed(candidate_paths):
        if not path or not Path(path).is_dir() or path not in sys.path:
            continue
        sys.path.remove(path)
        sys.path.insert(0, path)


def _load_cable_curve_importer():
    """Load the task-local USD BasisCurves cable importer."""
    from .usd_cable_curve_import import add_cable_from_usd_curve as importer

    return importer


def import_isaaclab_runtime_dependencies() -> None:
    """Import Isaac Lab simulation/config modules after Newton USD import."""
    global ik, sim_utils
    global CoupledProxyCfg, CoupledSolverCfg, CoupledSolverEntryCfg
    global MJWarpSolverCfg, NewtonCfg, NewtonCoupledManager, NewtonManager, NewtonSolverCfg, ProxyCouplingCfg

    if sim_utils is not None:
        return

    import newton.examples as newton_examples_module
    import newton.ik as ik_module
    from isaaclab_newton.physics import (
        CoupledProxyCfg as CoupledProxyCfgClass,
    )
    from isaaclab_newton.physics import (
        CoupledSolverCfg as CoupledSolverCfgClass,
    )
    from isaaclab_newton.physics import (
        CoupledSolverEntryCfg as CoupledSolverEntryCfgClass,
    )
    from isaaclab_newton.physics import (
        MJWarpSolverCfg as MJWarpSolverCfgClass,
    )
    from isaaclab_newton.physics import (
        NewtonCfg as NewtonCfgClass,
    )
    from isaaclab_newton.physics import (
        NewtonCoupledManager as NewtonCoupledManagerClass,
    )
    from isaaclab_newton.physics import (
        NewtonManager as NewtonManagerClass,
    )
    from isaaclab_newton.physics import (
        ProxyCouplingCfg as ProxyCouplingCfgClass,
    )
    from isaaclab_newton.physics.newton_manager_cfg import NewtonSolverCfg as NewtonSolverCfgClass

    import isaaclab.sim as sim_utils_module

    newton.examples = newton_examples_module
    ik = ik_module
    sim_utils = sim_utils_module
    CoupledProxyCfg = CoupledProxyCfgClass
    CoupledSolverCfg = CoupledSolverCfgClass
    CoupledSolverEntryCfg = CoupledSolverEntryCfgClass
    MJWarpSolverCfg = MJWarpSolverCfgClass
    NewtonCfg = NewtonCfgClass
    NewtonCoupledManager = NewtonCoupledManagerClass
    NewtonManager = NewtonManagerClass
    NewtonSolverCfg = NewtonSolverCfgClass
    ProxyCouplingCfg = ProxyCouplingCfgClass


def _lerobot_22_to_urdf_28(lr: list[float]) -> list[float]:
    """Convert the compact LeRobot RBY1 arm state used by the waterhose script to URDF DOFs."""
    q = [0.0] * 28
    q[0:6] = lr[0:6]
    q[6:13] = lr[6:13]
    q[12] += np.pi / 2.0
    q[13] = lr[20]
    q[14] = -lr[20] / 2.0
    q[15] = lr[20] / 2.0
    q[16:23] = lr[13:20]
    q[22] -= np.pi / 2.0
    q[23] = lr[21]
    q[24] = -lr[21] / 2.0
    q[25] = lr[21] / 2.0
    return q


def _find_label_index(labels: list[str], short_name: str) -> int:
    """Find a label by exact name or URDF namespace suffix."""
    suffix = "/" + short_name
    for index, label in enumerate(labels):
        if label == short_name or label.endswith(suffix):
            return index
    raise ValueError(f"Label {short_name!r} not found.")


def _maybe_find_label_index(labels: list[str], short_name: str) -> int | None:
    try:
        return _find_label_index(labels, short_name)
    except ValueError:
        return None


def _quat_to_vec4(quat) -> object:
    return wp.vec4(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _np_quat_multiply(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def _np_quat_inverse(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def _np_quat_rotate(q, v):
    xyz = np.asarray(q[:3], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    t = 2.0 * np.cross(xyz, v)
    return v + q[3] * t + np.cross(xyz, t)


def _np_quat_from_axis_angle(axis, angle: float):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = axis / norm
    half = 0.5 * angle
    return np.array([axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half), np.cos(half)])


def _np_relative_transform(parent_pos, parent_quat, child_pos, child_quat) -> tuple[np.ndarray, np.ndarray]:
    parent_inv = _np_quat_inverse(parent_quat)
    rel_pos = _np_quat_rotate(
        parent_inv, np.asarray(child_pos, dtype=np.float64) - np.asarray(parent_pos, dtype=np.float64)
    )
    rel_quat = _np_quat_multiply(parent_inv, child_quat)
    return rel_pos, rel_quat / max(np.linalg.norm(rel_quat), 1.0e-12)


def _np_transform_point_quat(parent_pos, parent_quat, rel_pos, rel_quat) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(parent_pos, dtype=np.float64) + _np_quat_rotate(parent_quat, rel_pos)
    quat = _np_quat_multiply(parent_quat, rel_quat)
    return pos, quat / max(np.linalg.norm(quat), 1.0e-12)


def _np_quat_slerp(q0, q1, t):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / max(np.linalg.norm(result), 1.0e-12)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q0 + b * q1


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _create_launcher_sim_cfg():
    """Create the minimal config used by ``launch_simulation`` to decide whether Kit is needed."""
    device = str(args_cli.device)
    if not device.startswith("cuda"):
        raise RuntimeError("The waterhose proxy-coupled demo requires a CUDA device.")
    dummy_newton_cfg = type("NewtonCfg", (), {"class_type": object})()
    return SimpleNamespace(
        dt=1.0 / args_cli.fps,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physics=dummy_newton_cfg,
    )


class WaterhoseSceneBuilder:
    """Build a shared model and Isaac Lab coupled-solver cfg for the demo."""

    def __init__(self, cfg: SimpleNamespace | None = None):
        self.cfg = cfg if cfg is not None else args_cli
        self.num_envs = int(self.cfg.num_envs)
        self.env_spacing = float(self.cfg.env_spacing)
        self.env_origins = [self._env_origin(env_id) for env_id in range(self.num_envs)]
        self.asset_root = self.cfg.asset_root.expanduser().resolve()
        self.robot_urdf = self._resolve_robot_urdf()
        self.scene_usd = self._resolve_scene_usd()
        self.cable_assets = self._resolve_cable_assets()
        self.cable_usd = self.cable_assets[0].usd_path

        self.robot_shape_cfg = newton.ModelBuilder.ShapeConfig(
            margin=float(self.cfg.robot_shape_margin),
            gap=float(self.cfg.robot_shape_gap),
            ke=float(self.cfg.robot_shape_ke),
            kd=float(self.cfg.robot_shape_kd),
            mu=float(self.cfg.robot_shape_mu),
        )
        self.robot_shape_cfg.is_hydroelastic = False
        self.hose_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=float(self.cfg.vbd_cable_density),
            margin=float(self.cfg.vbd_cable_margin),
            gap=float(self.cfg.vbd_cable_gap),
            ke=float(self.cfg.vbd_default_contact_ke),
            kd=float(self.cfg.vbd_default_contact_kd),
            mu=float(self.cfg.vbd_cable_mu),
        )
        self.static_shape_cfg = newton.ModelBuilder.ShapeConfig(
            density=0.0,
            margin=float(self.cfg.vbd_static_margin),
            gap=float(self.cfg.vbd_static_gap),
            ke=float(self.cfg.vbd_default_contact_ke),
            kd=float(self.cfg.vbd_default_contact_kd),
            mu=float(self.cfg.vbd_near_tip_mu),
        )
        self.cable_stretch_stiffness = float(self.cfg.cable_stretch_stiffness)
        self.cable_stretch_damping = float(self.cfg.cable_stretch_damping)
        self.cable_bend_rigidity = float(self.cfg.cable_bend_rigidity)
        self.cable_bend_damping = float(self.cfg.cable_bend_damping)

        self.table_pos = wp.vec3(0.95, -0.051, 0.1925)
        self.table_half_size = (0.55, 0.35, 0.1925)
        self.asset_xform = self._compute_asset_xform()
        self.socket_xform = self._compute_socket_xform(self.asset_xform)
        self.socket_pos = wp.transform_get_translation(self.socket_xform)
        self.socket_rot = wp.transform_get_rotation(self.socket_xform)
        self.hose_plug_tip_pos = self.socket_pos

        self.single_robot_model = None
        self.gripper_dofs: list[int] = []
        self.gripper_driver_dofs: list[int] = []
        self.gripper_finger_dofs: list[int] = []
        self.right_gripper_driver_dofs: list[int] = []
        self.left_gripper_driver_dofs: list[int] = []
        self.right_gripper_dofs: list[int] = []
        self.left_gripper_dofs: list[int] = []
        self.proxy_body_ids: list[int] = []
        self.proxy_shape_ids: list[int] = []
        self.tip_body_id = 0
        self.plug_body_id = 0
        self.grasp_body_id = 0
        self.scene_body_ids: list[int] = []
        self.scene_shape_ids: list[int] = []
        self.cable_body_q_targets: dict[int, tuple[float, float, float, float, float, float, float]] = {}
        self.primary_cable_body_ids: list[int] = []

        self._mujoco_body_count = 0
        self._mujoco_shape_count = 0
        self._mujoco_joint_count = 0
        self._mujoco_joint_coord_count = 0
        self._mujoco_joint_dof_count = 0
        self._mujoco_articulation_count = 0
        self._mujoco_body_ids: list[int] = []
        self._mujoco_shape_ids: list[int] = []
        self._mujoco_joint_ids: list[int] = []
        self._vbd_body_ids: list[int] = []
        self._vbd_shape_ids: list[int] = []
        self._vbd_joint_ids: list[int] = []
        self.cable_body_ids: list[int] = []
        self.cable_head_body_ids: list[int] = []
        self.cable_body_ids_by_curve: list[list[int]] = []
        self.cable_segment_lengths_by_curve: list[list[float]] = []
        self.cable_curve_prim_paths: list[str] = []
        self.cable_head_body_ids_by_env: list[list[int]] = []
        self.cable_body_ids_by_env_by_curve: list[list[list[int]]] = []
        self.cable_segment_lengths_by_env_by_curve: list[list[list[float]]] = []
        self.robot_joint_coord_ids_by_env: list[list[int]] = []

    def _env_origin(self, env_id: int):
        cols = int(np.ceil(np.sqrt(self.num_envs)))
        row = env_id // cols
        col = env_id % cols
        return wp.vec3(float(col) * self.env_spacing, float(row) * self.env_spacing, 0.0)

    def _resolve_asset(self, explicit_path: Path | None, relative_path: str) -> Path:
        path = explicit_path.expanduser().resolve() if explicit_path is not None else self.asset_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required cable robot asset not found: {path}")
        return path

    def _resolve_scene_usd(self) -> Path:
        """Resolve the authored static scene USD required by the waterhose task."""
        return self._resolve_asset(self.cfg.scene_usd, "Waterhose/Cable008/Cable008_Body.usda")

    def _resolve_asset_path(self, path_token: str | Path) -> Path:
        path = Path(path_token).expanduser()
        if not path.is_absolute():
            path = self.asset_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Required cable robot asset not found: {path}")
        return path

    def _resolve_cable_assets(self) -> list[CableUsdAsset]:
        if self.cfg.cable_usd is not None:
            cable_usd = self._resolve_asset_path(self.cfg.cable_usd)
            prim_paths = (
                [str(self.cfg.cable_prim)]
                if self.cfg.cable_prim
                else self._configured_cable_prim_paths(_DEFAULT_CABLE_PRIMS)
            )
            return [CableUsdAsset(cable_usd, curve_prim_path) for curve_prim_path in prim_paths]

        configured_usds = self._split_config_list(getattr(self.cfg, "cable_usds", None))
        if configured_usds:
            usd_paths = [self._resolve_asset_path(path_token) for path_token in configured_usds]
            default_prim_paths = [self._infer_first_basis_curve_prim_path(path) for path in usd_paths]
        else:
            cable_usd = self._resolve_asset(None, _DEFAULT_CABLE_USD_RELATIVE_PATH)
            prim_paths = (
                [str(self.cfg.cable_prim)]
                if self.cfg.cable_prim
                else self._configured_cable_prim_paths(_DEFAULT_CABLE_PRIMS)
            )
            if not prim_paths:
                raise RuntimeError("At least one cable prim must be configured.")
            return [CableUsdAsset(cable_usd, curve_prim_path) for curve_prim_path in prim_paths]

        if self.cfg.cable_prim:
            if len(usd_paths) != 1:
                raise ValueError(
                    "cable_prim can only be used with one cable USD. Use cable_prims to configure one prim per USD."
                )
            prim_paths = [str(self.cfg.cable_prim)]
        else:
            prim_paths = self._configured_cable_prim_paths(default_prim_paths)

        if len(prim_paths) != len(usd_paths):
            raise ValueError(
                "Cable USD and prim counts must match: "
                f"{len(usd_paths)} USD file(s), {len(prim_paths)} prim path(s)."
            )
        if not usd_paths:
            raise RuntimeError("At least one cable USD must be configured.")
        return [CableUsdAsset(usd_path, prim_path) for usd_path, prim_path in zip(usd_paths, prim_paths)]

    def _configured_cable_prim_paths(self, default_prim_paths: str | list[str]) -> list[str]:
        configured_prim_paths = self._split_config_list(self.cfg.cable_prims)
        if configured_prim_paths:
            return configured_prim_paths
        if isinstance(default_prim_paths, str):
            return self._split_config_list(default_prim_paths)
        return list(default_prim_paths)

    @staticmethod
    def _split_config_list(value: object | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            tokens: list[str] = []
            for item in value:
                tokens.extend(WaterhoseSceneBuilder._split_config_list(item))
            return tokens
        return [token.strip() for token in str(value).split(",") if token.strip()]

    @staticmethod
    def _infer_first_basis_curve_prim_path(usd_path: Path) -> str:
        from pxr import Usd  # noqa: PLC0415

        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise RuntimeError(f"Failed to open cable USD stage: {usd_path}")
        for prim in stage.Traverse():
            if prim.GetTypeName() == "BasisCurves":
                return str(prim.GetPath())
        raise ValueError(f"No BasisCurves prim found in cable USD stage: {usd_path}")

    def _resolve_robot_urdf(self) -> Path:
        if self.cfg.robot_urdf is not None:
            path = self.cfg.robot_urdf.expanduser().resolve()
        else:
            path = self.asset_root / "RBY1DF/urdf/robot_edited.urdf"
        if not path.is_file():
            raise FileNotFoundError(f"RBY1DF URDF not found: {path}")
        return path

    @staticmethod
    def _compute_asset_xform():
        """Match the bundle scripts' Cable008 placement relative to the robot."""
        table_half_z = 0.5 * (0.6 - 0.215)
        table_top_z = table_half_z + table_half_z
        z_offset = 0.902 + table_top_z
        y_offset = (0.293 - 0.395) / 2.0
        quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi / 2.0)
        return wp.transform(wp.vec3(0.95, y_offset, z_offset), quat)

    @staticmethod
    def _compute_socket_xform(asset_xform):
        socket_offset = wp.vec3(-0.259404, 0.362961, -0.262711)
        pos = wp.transform_point(asset_xform, socket_offset)
        rot = wp.transform_get_rotation(asset_xform) * wp.quat_from_axis_angle(
            wp.vec3(1.0, 0.0, 0.0), 20.0 * wp.pi / 180.0
        )
        return wp.transform(pos, rot)

    def build(self):
        """Return ``(builder, solver_cfg)`` for Isaac Lab's Newton manager."""
        static_scene_builder = self._build_static_scene_template()
        robot = self._build_robot()
        proto_builder = self._build_single_env_proto(robot, static_scene_builder)
        proto_meta = self._capture_proto_metadata(proto_builder, robot)
        builder = self._build_replicated_builder(proto_builder, proto_meta, robot, static_scene_builder)
        # Finalize the standalone IK model after USD scene parsing.  The USD
        # physics importer for Cable008 is fragile if called after CUDA/Warp
        # solver modules have already initialized during robot.finalize().
        self.single_robot_model = robot.finalize(device=str(self.cfg.device))
        builder.color()
        import_isaaclab_runtime_dependencies()
        NewtonManager._num_envs = self.num_envs
        return builder, self._make_solver_cfg()

    def _build_single_env_proto(self, robot, static_scene_builder):
        builder = newton.ModelBuilder()
        SolverVBD.register_custom_attributes(builder)
        builder.default_shape_cfg = self.robot_shape_cfg
        builder.bound_mass = robot.bound_mass
        builder.bound_inertia = robot.bound_inertia
        builder.add_builder(robot)

        self._mujoco_body_count = builder.body_count
        self._mujoco_shape_count = builder.shape_count
        self._mujoco_joint_count = builder.joint_count
        self._mujoco_joint_coord_count = builder.joint_coord_count
        self._mujoco_joint_dof_count = builder.joint_dof_count
        self._mujoco_articulation_count = builder.articulation_count
        self._mujoco_body_ids = list(range(self._mujoco_body_count))
        self._mujoco_shape_ids = list(range(self._mujoco_shape_count))
        self._mujoco_joint_ids = list(range(self._mujoco_joint_count))

        self._add_waterhose_world(builder, static_scene_builder)
        self._select_proxy_bodies(builder)
        return builder

    def _capture_proto_metadata(self, proto_builder, robot) -> dict[str, object]:
        return {
            "mujoco_body_ids": list(self._mujoco_body_ids),
            "mujoco_shape_ids": list(self._mujoco_shape_ids),
            "mujoco_joint_ids": list(self._mujoco_joint_ids),
            "mujoco_joint_coord_count": self._mujoco_joint_coord_count,
            "mujoco_joint_dof_count": self._mujoco_joint_dof_count,
            "mujoco_articulation_count": self._mujoco_articulation_count,
            "vbd_body_ids": list(self._vbd_body_ids),
            "vbd_shape_ids": list(self._vbd_shape_ids),
            "vbd_joint_ids": list(self._vbd_joint_ids),
            "proxy_body_ids": list(self.proxy_body_ids),
            "proxy_shape_ids": list(self.proxy_shape_ids),
            "scene_body_ids": list(self.scene_body_ids),
            "scene_shape_ids": list(self.scene_shape_ids),
            "cable_body_ids": list(self.cable_body_ids),
            "cable_head_body_ids": list(self.cable_head_body_ids),
            "cable_body_ids_by_curve": [list(body_ids) for body_ids in self.cable_body_ids_by_curve],
            "cable_segment_lengths_by_curve": [
                list(segment_lengths) for segment_lengths in self.cable_segment_lengths_by_curve
            ],
            "cable_curve_prim_paths": list(self.cable_curve_prim_paths),
            "primary_cable_body_ids": list(self.primary_cable_body_ids),
            "tip_body_id": self.tip_body_id,
            "plug_body_id": self.plug_body_id,
            "grasp_body_id": self.grasp_body_id,
            "cable_body_q_targets": dict(self.cable_body_q_targets),
            "robot_joint_coord_count": robot.joint_coord_count,
        }

    def _build_replicated_builder(self, proto_builder, proto_meta: dict[str, object], robot, static_scene_builder):
        self._reset_replicated_metadata()
        if self.num_envs == 1:
            self._append_env_metadata(proto_meta, (0, 0, 0, 0, 0), self.env_origins[0], is_primary=True)
            return proto_builder

        builder = newton.ModelBuilder()
        SolverVBD.register_custom_attributes(builder)
        builder.default_shape_cfg = self.robot_shape_cfg
        builder.bound_mass = proto_builder.bound_mass
        builder.bound_inertia = proto_builder.bound_inertia
        for env_id, origin in enumerate(self.env_origins):
            body_offset = builder.body_count
            shape_offset = builder.shape_count
            joint_offset = builder.joint_count
            joint_coord_offset = builder.joint_coord_count
            builder.add_builder(robot, xform=wp.transform(origin, wp.quat_identity()))
            self._mujoco_body_ids.extend(range(body_offset, body_offset + robot.body_count))
            self._mujoco_shape_ids.extend(range(shape_offset, shape_offset + robot.shape_count))
            self._mujoco_joint_ids.extend(range(joint_offset, joint_offset + robot.joint_count))
            self._mujoco_body_count += robot.body_count
            self._mujoco_shape_count += robot.shape_count
            self._mujoco_joint_count += robot.joint_count
            self._mujoco_joint_coord_count += robot.joint_coord_count
            self._mujoco_joint_dof_count += robot.joint_dof_count
            self._mujoco_articulation_count += robot.articulation_count
            self.robot_joint_coord_ids_by_env.append(
                [joint_coord_offset + joint_id for joint_id in range(robot.joint_coord_count)]
            )

        self._select_proxy_bodies(builder)

        vbd_body_ids: list[int] = []
        vbd_shape_ids: list[int] = []
        vbd_joint_ids: list[int] = []
        primary_env_metadata = None
        for env_id, origin in enumerate(self.env_origins):
            self._add_waterhose_world(builder, static_scene_builder, origin=origin)
            vbd_body_ids.extend(self._vbd_body_ids)
            vbd_shape_ids.extend(self._vbd_shape_ids)
            vbd_joint_ids.extend(self._vbd_joint_ids)
            self._append_current_cable_env_metadata()
            if env_id == 0:
                primary_env_metadata = {
                    "scene_body_ids": list(self.scene_body_ids),
                    "scene_shape_ids": list(self.scene_shape_ids),
                    "cable_body_ids": list(self.cable_body_ids),
                    "cable_head_body_ids": list(self.cable_head_body_ids),
                    "cable_body_ids_by_curve": [list(body_ids) for body_ids in self.cable_body_ids_by_curve],
                    "cable_segment_lengths_by_curve": [
                        list(segment_lengths) for segment_lengths in self.cable_segment_lengths_by_curve
                    ],
                    "cable_curve_prim_paths": list(self.cable_curve_prim_paths),
                    "primary_cable_body_ids": list(self.primary_cable_body_ids),
                    "tip_body_id": self.tip_body_id,
                    "plug_body_id": self.plug_body_id,
                    "grasp_body_id": self.grasp_body_id,
                }

        self._vbd_body_ids = vbd_body_ids
        self._vbd_shape_ids = vbd_shape_ids
        self._vbd_joint_ids = vbd_joint_ids
        if primary_env_metadata is not None:
            self.scene_body_ids = primary_env_metadata["scene_body_ids"]
            self.scene_shape_ids = primary_env_metadata["scene_shape_ids"]
            self.cable_body_ids = primary_env_metadata["cable_body_ids"]
            self.cable_head_body_ids = primary_env_metadata["cable_head_body_ids"]
            self.cable_body_ids_by_curve = primary_env_metadata["cable_body_ids_by_curve"]
            self.cable_segment_lengths_by_curve = primary_env_metadata["cable_segment_lengths_by_curve"]
            self.cable_curve_prim_paths = primary_env_metadata["cable_curve_prim_paths"]
            self.primary_cable_body_ids = primary_env_metadata["primary_cable_body_ids"]
            self.tip_body_id = primary_env_metadata["tip_body_id"]
            self.plug_body_id = primary_env_metadata["plug_body_id"]
            self.grasp_body_id = primary_env_metadata["grasp_body_id"]
        return builder

    def _reset_replicated_metadata(self) -> None:
        self._mujoco_body_count = 0
        self._mujoco_shape_count = 0
        self._mujoco_joint_count = 0
        self._mujoco_joint_coord_count = 0
        self._mujoco_joint_dof_count = 0
        self._mujoco_articulation_count = 0
        self._mujoco_body_ids = []
        self._mujoco_shape_ids = []
        self._mujoco_joint_ids = []
        self._vbd_body_ids = []
        self._vbd_shape_ids = []
        self._vbd_joint_ids = []
        self.proxy_body_ids = []
        self.proxy_shape_ids = []
        self.cable_body_q_targets = {}
        self.cable_body_ids_by_curve = []
        self.cable_segment_lengths_by_curve = []
        self.cable_curve_prim_paths = []
        self.cable_head_body_ids_by_env = []
        self.cable_body_ids_by_env_by_curve = []
        self.cable_segment_lengths_by_env_by_curve = []
        self.robot_joint_coord_ids_by_env = []

    def _append_current_cable_env_metadata(self) -> None:
        self.cable_head_body_ids_by_env.append(list(self.cable_head_body_ids))
        self.cable_body_ids_by_env_by_curve.append([list(body_ids) for body_ids in self.cable_body_ids_by_curve])
        self.cable_segment_lengths_by_env_by_curve.append(
            [list(segment_lengths) for segment_lengths in self.cable_segment_lengths_by_curve]
        )

    def tip_body_ids_by_env(self) -> list[int]:
        """Return the simulated cable tip body id for each environment."""
        if not self.cable_body_ids_by_env_by_curve:
            return [self.tip_body_id]

        body_ids: list[int] = []
        for env_curves in self.cable_body_ids_by_env_by_curve:
            if not env_curves or not env_curves[0]:
                raise RuntimeError("Cable tip body metadata is incomplete.")
            body_ids.append(int(env_curves[0][0]))
        if len(body_ids) != self.num_envs:
            raise RuntimeError(f"Expected {self.num_envs} cable tip bodies, found {body_ids}.")
        return body_ids

    def plug_body_ids_by_env(self) -> list[int]:
        """Return the simulated cable plug/head body id for each environment."""
        if not self.cable_head_body_ids_by_env:
            return [self.plug_body_id]

        body_ids: list[int] = []
        for env_id, head_body_ids in enumerate(self.cable_head_body_ids_by_env):
            if head_body_ids:
                body_ids.append(int(head_body_ids[0]))
                continue
            env_curves = self.cable_body_ids_by_env_by_curve[env_id]
            if not env_curves or not env_curves[0]:
                raise RuntimeError("Cable plug body metadata is incomplete.")
            body_ids.append(int(env_curves[0][0]))
        if len(body_ids) != self.num_envs:
            raise RuntimeError(f"Expected {self.num_envs} cable plug bodies, found {body_ids}.")
        return body_ids

    def _append_env_metadata(
        self,
        proto_meta: dict[str, object],
        offsets: tuple[int, int, int, int, int],
        origin,
        *,
        is_primary: bool,
    ) -> None:
        body_offset, shape_offset, joint_offset, joint_coord_offset, _joint_dof_offset = offsets

        def offset_ids(ids: list[int], offset: int) -> list[int]:
            return [offset + int(idx) for idx in ids]

        self._mujoco_body_ids.extend(offset_ids(proto_meta["mujoco_body_ids"], body_offset))
        self._mujoco_shape_ids.extend(offset_ids(proto_meta["mujoco_shape_ids"], shape_offset))
        self._mujoco_joint_ids.extend(offset_ids(proto_meta["mujoco_joint_ids"], joint_offset))
        self._vbd_body_ids.extend(offset_ids(proto_meta["vbd_body_ids"], body_offset))
        self._vbd_shape_ids.extend(offset_ids(proto_meta["vbd_shape_ids"], shape_offset))
        self._vbd_joint_ids.extend(offset_ids(proto_meta["vbd_joint_ids"], joint_offset))
        self.proxy_body_ids.extend(offset_ids(proto_meta["proxy_body_ids"], body_offset))
        self.proxy_shape_ids.extend(offset_ids(proto_meta["proxy_shape_ids"], shape_offset))

        self._mujoco_body_count += len(proto_meta["mujoco_body_ids"])
        self._mujoco_shape_count += len(proto_meta["mujoco_shape_ids"])
        self._mujoco_joint_count += len(proto_meta["mujoco_joint_ids"])
        self._mujoco_joint_coord_count += int(proto_meta["mujoco_joint_coord_count"])
        self._mujoco_joint_dof_count += int(proto_meta["mujoco_joint_dof_count"])
        self._mujoco_articulation_count += int(proto_meta["mujoco_articulation_count"])

        robot_joint_coord_count = int(proto_meta["robot_joint_coord_count"])
        self.robot_joint_coord_ids_by_env.append(
            [joint_coord_offset + joint_id for joint_id in range(robot_joint_coord_count)]
        )

        origin_np = np.array([float(origin[0]), float(origin[1]), float(origin[2]), 0.0, 0.0, 0.0, 0.0])
        for proto_body_id, target_q in proto_meta["cable_body_q_targets"].items():
            target_np = np.asarray(target_q, dtype=np.float64) + origin_np
            self.cable_body_q_targets[body_offset + int(proto_body_id)] = tuple(float(v) for v in target_np)

        if not is_primary:
            return
        self.scene_body_ids = offset_ids(proto_meta["scene_body_ids"], body_offset)
        self.scene_shape_ids = offset_ids(proto_meta["scene_shape_ids"], shape_offset)
        self.cable_body_ids = offset_ids(proto_meta["cable_body_ids"], body_offset)
        self.cable_head_body_ids = offset_ids(proto_meta["cable_head_body_ids"], body_offset)
        cable_body_ids_by_curve = [
            offset_ids(list(body_ids), body_offset) for body_ids in proto_meta["cable_body_ids_by_curve"]
        ]
        cable_segment_lengths_by_curve = [
            [float(length) for length in segment_lengths]
            for segment_lengths in proto_meta["cable_segment_lengths_by_curve"]
        ]
        self.cable_head_body_ids_by_env.append(list(self.cable_head_body_ids))
        self.cable_body_ids_by_env_by_curve.append([list(body_ids) for body_ids in cable_body_ids_by_curve])
        self.cable_segment_lengths_by_env_by_curve.append(
            [list(segment_lengths) for segment_lengths in cable_segment_lengths_by_curve]
        )

        self.cable_body_ids_by_curve = cable_body_ids_by_curve
        self.cable_segment_lengths_by_curve = cable_segment_lengths_by_curve
        self.cable_curve_prim_paths = list(proto_meta["cable_curve_prim_paths"])
        self.primary_cable_body_ids = offset_ids(proto_meta["primary_cable_body_ids"], body_offset)
        self.tip_body_id = body_offset + int(proto_meta["tip_body_id"])
        self.plug_body_id = body_offset + int(proto_meta["plug_body_id"])
        self.grasp_body_id = body_offset + int(proto_meta["grasp_body_id"])

    def _build_static_scene_template(self):
        """Build the static contact scene before robot setup."""
        builder = newton.ModelBuilder()
        SolverVBD.register_custom_attributes(builder)
        builder.default_shape_cfg = self.static_shape_cfg
        self._add_static_scene_contacts(builder)
        return builder

    def _build_robot(self):
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.default_shape_cfg = self.robot_shape_cfg
        robot.bound_mass = 1.0e-4
        robot.bound_inertia = 1.0e-6
        robot.add_urdf(
            str(self.robot_urdf),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            ignore_inertial_definitions=True,
        )

        self._label_robot_bodies(robot)
        self._patch_zero_mass_children(robot)
        self._disable_gripper_mimic_constraints(robot)
        self._discover_gripper_dofs(robot)
        self._configure_robot_dofs(robot)
        self._seed_initial_joint_state(robot)
        self._enable_gravity_compensation(robot)
        return robot

    @staticmethod
    def _label_robot_bodies(robot) -> None:
        """Give imported RBY1 bodies stable absolute labels for selector-based coupling."""
        for body_id, label in enumerate(robot.body_label):
            body_name = str(label).rsplit("/", 1)[-1]
            robot.body_label[body_id] = f"{ROBOT_PRIM_PATH}/{body_name}"

    @staticmethod
    def _patch_zero_mass_children(robot) -> None:
        """Patch invalid zero-mass driven links without editing the URDF asset."""
        min_inertia = wp.mat33(
            robot.bound_inertia,
            0.0,
            0.0,
            0.0,
            robot.bound_inertia,
            0.0,
            0.0,
            0.0,
            robot.bound_inertia,
        )
        for child in {int(value) for value in robot.joint_child if int(value) >= 0}:
            if robot.body_mass[child] <= 0.0:
                robot.body_mass[child] = robot.bound_mass
                robot.body_inv_mass[child] = 1.0 / robot.bound_mass
                robot.body_inertia[child] = min_inertia
                robot.body_inv_inertia[child] = wp.inverse(min_inertia)

    @staticmethod
    def _disable_gripper_mimic_constraints(robot) -> None:
        """Match the reference Newton waterhose examples' gripper behavior."""
        mimic_enabled = getattr(robot, "constraint_mimic_enabled", [])
        mimic_count = len(mimic_enabled)
        if mimic_count > 0:
            robot.constraint_mimic_enabled[-mimic_count:] = [False] * mimic_count

    def _discover_gripper_dofs(self, robot) -> None:
        self.right_gripper_driver_dofs = self._joint_dofs_by_name(robot, ["right_gripper_finger_joint_1"])
        self.left_gripper_driver_dofs = self._joint_dofs_by_name(robot, ["left_gripper_finger_joint_1"])
        self.right_gripper_dofs = self._joint_dofs_by_name(
            robot, ["right_gripper_left_finger_joint", "right_gripper_right_finger_joint"]
        )
        self.left_gripper_dofs = self._joint_dofs_by_name(
            robot, ["left_gripper_left_finger_joint", "left_gripper_right_finger_joint"]
        )
        self.gripper_driver_dofs = [*self.right_gripper_driver_dofs, *self.left_gripper_driver_dofs]
        self.gripper_finger_dofs = [*self.right_gripper_dofs, *self.left_gripper_dofs]
        self.gripper_dofs = [*self.gripper_driver_dofs, *self.gripper_finger_dofs]

    @staticmethod
    def _joint_dofs_by_name(robot, names: list[str]) -> list[int]:
        dofs = []
        for name in names:
            joint_id = _maybe_find_label_index(robot.joint_label, name)
            if joint_id is not None:
                dofs.append(int(robot.joint_qd_start[joint_id]))
        return dofs

    def _configure_robot_dofs(self, robot) -> None:
        gripper_set = set(self.gripper_driver_dofs) | set(self.gripper_finger_dofs)
        gripper_scale = float(self.cfg.gripper_drive_scale)
        for dof in range(robot.joint_dof_count):
            if dof in gripper_set:
                robot.joint_target_ke[dof] = float(self.cfg.gripper_joint_target_ke) * gripper_scale
                robot.joint_target_kd[dof] = float(self.cfg.gripper_joint_target_kd) * gripper_scale
                robot.joint_effort_limit[dof] = float(self.cfg.gripper_joint_effort_limit) * gripper_scale
                robot.joint_armature[dof] = float(self.cfg.gripper_joint_armature)
            else:
                robot.joint_target_ke[dof] = float(self.cfg.robot_joint_target_ke)
                robot.joint_target_kd[dof] = float(self.cfg.robot_joint_target_kd)
                robot.joint_effort_limit[dof] = float(self.cfg.robot_joint_effort_limit)
                robot.joint_armature[dof] = float(self.cfg.robot_joint_armature)
            robot.joint_target_mode[dof] = int(JointTargetMode.POSITION)

    def _seed_initial_joint_state(self, robot) -> None:
        q = _lerobot_22_to_urdf_28(LEROBOT_INITIAL_STATE_22)
        if len(q) != robot.joint_coord_count:
            q = (q + [0.0] * robot.joint_coord_count)[: robot.joint_coord_count]
        robot.joint_q = q
        robot.joint_target_pos = list(q)

    def _enable_gravity_compensation(self, robot) -> None:
        gravcomp_body = robot.custom_attributes.get("mujoco:gravcomp")
        if gravcomp_body is not None:
            gravcomp_body.values = gravcomp_body.values or {}
            for body_id in range(1, robot.body_count):
                gravcomp_body.values[body_id] = 1.0

        gravcomp_joint = robot.custom_attributes.get("mujoco:jnt_actgravcomp")
        if gravcomp_joint is not None:
            gravcomp_joint.values = gravcomp_joint.values or {}
            for dof_id in range(robot.joint_dof_count):
                if dof_id not in self.gripper_dofs:
                    gravcomp_joint.values[dof_id] = True

    def _add_waterhose_world(self, builder, static_scene_builder, origin=None) -> None:
        builder.default_shape_cfg = self.hose_shape_cfg
        builder.rigid_contact_margin = 0.0
        builder.rigid_gap = 0.001
        vbd_body_start = builder.body_count
        vbd_shape_start = builder.shape_count
        vbd_joint_start = builder.joint_count

        scene_body_start = builder.body_count
        scene_shape_start = builder.shape_count
        if origin is None:
            builder.add_builder(static_scene_builder)
        else:
            builder.add_builder(static_scene_builder, xform=wp.transform(origin, wp.quat_identity()))
        self.scene_body_ids = list(range(scene_body_start, builder.body_count))
        self.scene_shape_ids = list(range(scene_shape_start, builder.shape_count))
        for shape_index, shape_a in enumerate(self.scene_shape_ids):
            for shape_b in self.scene_shape_ids[shape_index + 1 :]:
                builder.add_shape_collision_filter_pair(shape_a, shape_b)

        self._add_hoses_from_usd(builder, origin=origin)

        self._vbd_body_ids = list(range(vbd_body_start, builder.body_count))
        self._vbd_shape_ids = list(range(vbd_shape_start, builder.shape_count))
        self._vbd_joint_ids = list(range(vbd_joint_start, builder.joint_count))

    def _add_hoses_from_usd(self, builder, origin=None) -> None:
        global add_cable_from_usd_curve

        if add_cable_from_usd_curve is None:
            add_cable_from_usd_curve = _load_cable_curve_importer()

        cfg = self.hose_shape_cfg.copy()
        head_cfg = newton.ModelBuilder.ShapeConfig(
            density=float(self.cfg.vbd_cable_density),
            ke=float(self.cfg.vbd_default_contact_ke),
            kd=float(self.cfg.vbd_default_contact_kd),
            mu=float(self.cfg.vbd_cable_mu),
        )
        cable_joint_ids: list[int] = []
        fixed_body_ids: list[int] = []
        self.cable_body_ids = []
        self.cable_head_body_ids = []
        self.cable_body_ids_by_curve = []
        self.cable_segment_lengths_by_curve = []
        self.cable_curve_prim_paths = []

        for cable_index, cable_asset in enumerate(self.cable_assets):
            cable_bend_stiffness = self._cable_bend_stiffness(cable_asset)
            result = add_cable_from_usd_curve(
                builder=builder,
                source_usd_path=str(cable_asset.usd_path),
                curve_prim_path=cable_asset.curve_prim_path,
                cable_label=f"water_hose_cable_{cable_index}",
                cable_cfg=cfg,
                stretch_stiffness=self.cable_stretch_stiffness,
                stretch_damping=self.cable_stretch_damping,
                bend_stiffness=cable_bend_stiffness,
                bend_damping=self.cable_bend_damping,
                wrap_in_articulation=False,
                head_shape_mode="mesh",
                head_cfg=head_cfg,
                head_mass=0.0,
                resample_segments=int(self.cfg.cable_num_segments),
            )
            self.cable_body_ids.extend(result.cable_body_ids)
            self.cable_head_body_ids.extend(result.head_body_ids)
            self.cable_body_ids_by_curve.append(list(result.cable_body_ids))
            self.cable_segment_lengths_by_curve.append(self._cable_segment_lengths(builder, result.cable_body_ids))
            self.cable_curve_prim_paths.append(cable_asset.curve_prim_path)
            cable_joint_ids.extend([*result.cable_joint_ids, *result.head_fixed_joint_ids])
            fixed_body_ids.extend(int(v) for v in result.fixed_body_ids)
            self._sanitize_imported_labels(builder, result, cable_index)
            self._filter_cable_self_collisions(builder, [*result.cable_body_ids, *result.head_body_ids])
            self._filter_head_parent_neighbor_collisions(builder, result, cable_index)
            self._cache_asset_transformed_body_targets(builder, [*result.cable_body_ids, *result.head_body_ids], origin)

            if cable_index == 0:
                self.primary_cable_body_ids = list(result.cable_body_ids)
                self.tip_body_id = result.cable_body_ids[0]
                self.plug_body_id = result.head_body_ids[0] if result.head_body_ids else result.cable_body_ids[0]
                self.grasp_body_id = self.plug_body_id

        if not self.cable_body_ids:
            raise RuntimeError("At least one cable prim must be configured.")
        if cable_joint_ids:
            builder.add_articulation(cable_joint_ids, label="waterhose_articulation")
        for body_id in sorted(set(fixed_body_ids)):
            builder.body_mass[body_id] = 0.0
            builder.body_inv_mass[body_id] = 0.0
            builder.body_inertia[body_id] = wp.mat33()
            builder.body_inv_inertia[body_id] = wp.mat33()

    def _cable_bend_stiffness(self, cable_asset: CableUsdAsset) -> float:
        """Convert the proxy-coupled example's bend rigidity to per-joint stiffness."""
        return self.cable_bend_rigidity / self._cable_mean_edge_length(cable_asset)

    @staticmethod
    def _cable_segment_lengths(builder, body_ids: list[int]) -> list[float]:
        """Return each rod segment length from its capsule shape."""
        lengths: list[float] = []
        for index, body_id in enumerate(body_ids):
            segment_length = None
            for shape_id in builder.body_shapes.get(body_id, []):
                shape_scale = builder.shape_scale[shape_id]
                half_length = float(shape_scale[1])
                if half_length > 0.0:
                    segment_length = 2.0 * half_length
                    break
            if segment_length is None:
                if index + 1 < len(body_ids):
                    p0 = wp.transform_get_translation(builder.body_q[body_id])
                    p1 = wp.transform_get_translation(builder.body_q[body_ids[index + 1]])
                    segment_length = float(wp.length(p1 - p0))
                elif lengths:
                    segment_length = lengths[-1]
                else:
                    segment_length = 0.01
            lengths.append(float(segment_length))
        return lengths

    def _cable_mean_edge_length(self, cable_asset: CableUsdAsset) -> float:
        """Return the mean authored cable edge length [m] for one USD BasisCurves prim."""
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(cable_asset.usd_path))
        if stage is None:
            raise RuntimeError(f"Failed to open cable USD stage: {cable_asset.usd_path}")
        curve_prim_path = cable_asset.curve_prim_path
        curve_prim = stage.GetPrimAtPath(curve_prim_path)
        if not curve_prim or not curve_prim.IsValid():
            raise ValueError(f"Curve prim {curve_prim_path!r} is not valid in stage {cable_asset.usd_path}.")

        points_attr = curve_prim.GetAttribute("points")
        points_raw = points_attr.Get() if points_attr else None
        if points_raw is None:
            raise ValueError(f"BasisCurves {curve_prim_path!r} is missing points.")

        points = np.asarray(points_raw, dtype=np.float64).reshape(-1, 3)
        if UsdGeom.StageHasAuthoredMetersPerUnit(stage):
            meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        else:
            meters_per_unit = 1.0
        points *= meters_per_unit

        connections_attr = curve_prim.GetAttribute("connections")
        connections_raw = connections_attr.Get() if connections_attr else None
        if connections_raw is None:
            edges = [(index, index + 1) for index in range(points.shape[0] - 1)]
        else:
            edges = [(int(pair[0]), int(pair[1])) for pair in connections_raw]

        lengths = [float(np.linalg.norm(points[v] - points[u])) for u, v in edges]
        positive_lengths = [length for length in lengths if length > 0.0]
        if not positive_lengths:
            raise ValueError(f"BasisCurves {curve_prim_path!r} has no positive-length edges.")
        if int(self.cfg.cable_num_segments) > 0:
            return float(sum(positive_lengths) / int(self.cfg.cable_num_segments))
        return float(np.mean(positive_lengths))

    @staticmethod
    def _sanitize_imported_labels(builder, result, cable_index: int) -> None:
        """Replace importer labels containing USD-path punctuation with valid visualization paths."""
        body_ids = [*result.cable_body_ids, *result.head_body_ids]
        joint_ids = [*result.cable_joint_ids, *result.head_fixed_joint_ids]
        shape_ids: list[int] = []
        for body_id in body_ids:
            shape_ids.extend(builder.body_shapes.get(body_id, []))
        for index, body_id in enumerate(body_ids):
            builder.body_label[body_id] = f"/World/Waterhose/Cable_{cable_index}/Body_{index:03d}"
        for index, shape_id in enumerate(shape_ids):
            builder.shape_label[shape_id] = f"/World/Waterhose/Cable_{cable_index}/Shape_{index:03d}"
        for index, joint_id in enumerate(joint_ids):
            builder.joint_label[joint_id] = f"/World/Waterhose/Cable_{cable_index}/Joint_{index:03d}"

    @staticmethod
    def _filter_cable_self_collisions(builder, body_ids: list[int]) -> None:
        shape_ids: list[int] = []
        for body_id in body_ids:
            shape_ids.extend(builder.body_shapes.get(body_id, []))
        for i, shape_a in enumerate(shape_ids):
            for shape_b in shape_ids[i + 1 :]:
                builder.add_shape_collision_filter_pair(shape_a, shape_b)

    @staticmethod
    def _filter_head_parent_neighbor_collisions(builder, result, cable_index: int) -> None:
        if not result.head_body_ids or len(result.cable_body_ids) < 2:
            return
        neighbor_index = 1 if cable_index == 0 else -2
        neighbor_body = result.cable_body_ids[neighbor_index]
        for head_body in result.head_body_ids:
            for head_shape in builder.body_shapes.get(head_body, []):
                for neighbor_shape in builder.body_shapes.get(neighbor_body, []):
                    builder.add_shape_collision_filter_pair(head_shape, neighbor_shape)

    def apply_runtime_cable_asset_xform(self, sync_solver_prev: bool = True) -> None:
        """Apply the authored cable scene transform to model/state buffers after Isaac Lab reset."""
        if not self.cable_body_q_targets:
            return

        def apply_to_array(body_q_arr) -> None:
            body_q_np = body_q_arr.numpy()
            for body_id, target_q in self.cable_body_q_targets.items():
                body_q_np[body_id] = target_q
            wp.copy(body_q_arr, wp.array(body_q_np, dtype=wp.transform, device=body_q_arr.device))

        apply_to_array(NewtonManager.get_model().body_q)
        apply_to_array(NewtonManager.get_state_0().body_q)
        apply_to_array(NewtonManager.get_state_1().body_q)

        if sync_solver_prev:
            vbd_solver = NewtonCoupledManager.get_entry_solver(HOSE_ENTRY)
            body_q_prev = getattr(vbd_solver, "body_q_prev", None)
            if body_q_prev is not None:
                apply_to_array(body_q_prev)

    def configure_runtime_vbd_solver(self) -> None:
        """Match the waterhose reference: VBD joints run in soft penalty mode."""
        vbd_solver = NewtonCoupledManager.get_entry_solver(HOSE_ENTRY)
        set_mode = getattr(vbd_solver, "set_joint_constraint_mode", None)
        if set_mode is None:
            return
        for joint_id in self._vbd_joint_ids:
            set_mode(joint_id, hard=False)

    def _cache_asset_transformed_body_targets(self, builder, body_ids: list[int], origin=None) -> None:
        rot = wp.transform_get_rotation(self.asset_xform)
        pos = wp.transform_get_translation(self.asset_xform)
        if origin is not None:
            pos = pos + origin
        for body_id in body_ids:
            body_q = builder.body_q[body_id]
            old_pos = wp.transform_get_translation(body_q)
            old_rot = wp.transform_get_rotation(body_q)
            new_pos = wp.quat_rotate(rot, old_pos) + pos
            new_rot = rot * old_rot
            self.cable_body_q_targets[body_id] = (
                float(new_pos[0]),
                float(new_pos[1]),
                float(new_pos[2]),
                float(new_rot[0]),
                float(new_rot[1]),
                float(new_rot[2]),
                float(new_rot[3]),
            )

    def _add_static_scene_contacts(self, builder) -> None:
        if _requested_kit_visualizer():
            self._add_kit_static_contact_proxy(builder)
            return
        self._add_static_scene_from_usd(builder)

    def _add_static_scene_from_usd(self, builder) -> None:
        scene_result = builder.add_usd(
            str(self.scene_usd),
            xform=self.asset_xform,
            root_path="/root",
            load_sites=False,
            load_visual_shapes=True,
            hide_collision_shapes=False,
            parse_mujoco_options=False,
            only_load_enabled_joints=True,
            only_load_enabled_rigid_bodies=False,
        )
        self.scene_body_ids = sorted({int(v) for v in scene_result["path_body_map"].values()})
        for body_id in self.scene_body_ids:
            builder.body_mass[body_id] = 0.0
            builder.body_inv_mass[body_id] = 0.0
            builder.body_inertia[body_id] = wp.mat33()
            builder.body_inv_inertia[body_id] = wp.mat33()
        self.scene_shape_ids = sorted(int(v) for v in scene_result["path_shape_map"].values())

    def _add_kit_static_contact_proxy(self, builder) -> None:
        """Use simple Newton contact shapes while Kit owns the visual USD scene."""
        shape_start = builder.shape_count
        table_cfg = self.static_shape_cfg.copy()
        socket_cfg = self.static_shape_cfg.copy()
        socket_cfg.mu = float(self.cfg.vbd_near_tip_mu)

        builder.add_shape_box(
            body=-1,
            xform=wp.transform(self.table_pos, wp.quat_identity()),
            hx=float(self.table_half_size[0]),
            hy=float(self.table_half_size[1]),
            hz=float(self.table_half_size[2]),
            cfg=table_cfg,
            label="waterhose_static_table_contact",
        )

        insertion_dir = wp.quat_rotate(self.socket_rot, wp.vec3(0.0, 0.0, 1.0))
        socket_center = self.socket_pos - insertion_dir * 0.025
        builder.add_shape_box(
            body=-1,
            xform=wp.transform(socket_center, self.socket_rot),
            hx=0.045,
            hy=0.045,
            hz=0.025,
            cfg=socket_cfg,
            label="waterhose_static_socket_contact",
        )

        self.scene_body_ids = []
        self.scene_shape_ids = list(range(shape_start, builder.shape_count))

    def _select_proxy_bodies(self, builder) -> None:
        self.proxy_body_ids = []
        self.proxy_shape_ids = []
        for body_id in self._mujoco_body_ids:
            label = builder.body_label[body_id] if body_id < len(builder.body_label) else ""
            short_label = label.rsplit("/", 1)[-1]
            if short_label not in PROXY_BODY_NAMES:
                continue
            shape_ids = [
                int(shape_id)
                for shape_id in self._mujoco_shape_ids
                for shape_body in [builder.shape_body[shape_id]]
                if int(shape_body) == body_id
                and (int(builder.shape_flags[shape_id]) & int(newton.ShapeFlags.COLLIDE_SHAPES))
            ]
            if shape_ids:
                self.proxy_body_ids.append(body_id)
                self.proxy_shape_ids.extend(shape_ids)
        if not self.proxy_body_ids:
            raise RuntimeError("No RBY1 gripper proxy bodies found for waterhose coupling.")

    def _make_solver_cfg(self):
        global _ACTIVE_SCENE_BUILDER
        _ACTIVE_SCENE_BUILDER = self
        rigid_contact_max_per_world = max(1, int(self.cfg.rigid_contact_max) // max(1, self.num_envs))

        def create_proxy_collision_pipeline(model_view):
            return newton.examples.create_collision_pipeline(
                model_view,
                self.cfg,
                contact_matching="sticky",
                contact_matching_pos_threshold=0.005,
                contact_matching_normal_dot_threshold=0.95,
            )

        return CoupledSolverCfg(
            entries=[
                CoupledSolverEntryCfg(
                    name=ROBOT_ENTRY,
                    solver_cfg=MJWarpSolverCfg(
                        solver="newton",
                        integrator="implicitfast",
                        cone="elliptic",
                        njmax=rigid_contact_max_per_world,
                        nconmax=rigid_contact_max_per_world,
                        iterations=int(self.cfg.mujoco_iterations),
                        ls_iterations=int(self.cfg.mujoco_ls_iterations),
                        ls_parallel=bool(self.cfg.mujoco_ls_parallel),
                        impratio=float(self.cfg.mujoco_impratio),
                        use_mujoco_contacts=bool(self.cfg.mujoco_use_mujoco_contacts),
                    ),
                    bodies=self._mujoco_body_ids,
                    joints=self._mujoco_joint_ids,
                    shapes=self._mujoco_shape_ids,
                    configure_view=configure_mujoco_view,
                    substeps=self.cfg.rigid_substeps,
                ),
                CoupledSolverEntryCfg(
                    name=HOSE_ENTRY,
                    solver_cfg=NewtonSolverCfg(),
                    solver_class=SolverVBD,
                    solver_kwargs={
                        "iterations": self.cfg.vbd_iterations,
                        "friction_epsilon": float(self.cfg.vbd_solver_friction_epsilon),
                        "rigid_avbd_beta": float(self.cfg.vbd_rigid_avbd_beta),
                        "rigid_contact_history": True,
                        "rigid_contact_k_start": float(self.cfg.vbd_rigid_contact_k_start),
                        "rigid_body_contact_buffer_size": int(self.cfg.vbd_rigid_contact_buffer_size),
                        "rigid_joint_linear_k_start": float(self.cfg.vbd_rigid_joint_linear_k_start),
                        "rigid_joint_angular_k_start": float(self.cfg.vbd_rigid_joint_angular_k_start),
                    },
                    bodies=self._vbd_body_ids,
                    joints=self._vbd_joint_ids,
                    shapes=self._vbd_shape_ids,
                    configure_view=configure_vbd_view,
                ),
            ],
            use_collision_pipeline=False,
            proxy_coupling=ProxyCouplingCfg(
                proxies=[
                    CoupledProxyCfg(
                        source=ROBOT_ENTRY,
                        destination=HOSE_ENTRY,
                        bodies=self.proxy_body_ids,
                        mass_scale=float(self.cfg.proxy_mass_scale),
                        mode="lagged",
                        collision_pipeline_factory=create_proxy_collision_pipeline,
                        collide_interval=int(self.cfg.vbd_collide_substeps),
                    )
                ],
                iterations=self.cfg.proxy_iterations,
            ),
        )


class WaterhoseIKController:
    """Scripted Newton IK controller layered on Isaac Lab's Newton state/control buffers."""

    _PHASES = [
        ("approach_hose", 2.0),
        ("engage_hose", 0.9),
        ("grasp_hose", 0.4),
        ("hold_grasp", 0.25),
        ("retract", 0.9),
        ("settle", 0.1),
        ("approach_socket", 3.0),
        ("align_axes", 1.5),
        ("verify_align", 2.0),
        ("insert_hose", 3.0),
        ("release_hose", 1.0),
        ("withdraw", 2.0),
        ("wait_after_withdraw", 1.0),
        ("reapproach", 2.0),
        ("reengage", 0.4),
        ("regrasp", 3.0),
        ("pull", 4.0),
        ("final_release", 2.0),
        ("done", 999.0),
    ]

    def __init__(self, scene_builder: WaterhoseSceneBuilder):
        self.scene_builder = scene_builder
        self.cfg = scene_builder.cfg
        self.model = NewtonManager.get_model()
        self.state = NewtonManager.get_state_0()
        self.control = NewtonManager.get_control()
        self.single_robot_model = scene_builder.single_robot_model
        self.ik_iters = 24
        self.start_time = 0.0
        self.last_task = ""
        self.phase_index = 0
        self.phase_start_time = 0.0
        self.last_target_pos = None
        self.last_target_quat = None
        self.live_hose_target_pos = None
        self.live_hose_target_quat = None
        self.live_hose_target_alpha = 0.35
        self.max_target_step = 0.018
        self.max_joint_step = 0.02
        self.max_gripper_joint_step = 0.20
        self.max_gripper_joint_velocity = 0.03
        self.gripper_centering_k = 0.4
        self.gripper_axis_centering_k = 0.8
        self.gripper_centering_max_step = 0.003
        self.phase_start_ee_pos = np.zeros(3, dtype=np.float64)
        self.phase_start_ee_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.phase_start_plug_pos = np.zeros(3, dtype=np.float64)
        self.phase_start_plug_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.phase_target_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.verify_align_retries = 0
        self.max_verify_align_retries = 1000000
        self.align_lateral_threshold = 0.0005
        self.align_axis_cosine_threshold = -0.997
        self.align_lateral_gain = 1.0
        self.insert_start_depth = 0.005
        self.insert_final_depth = 0.035
        self.pull_distance = 0.06
        self.transfer_retract_standoff = 0.045
        self.transfer_retract_lift = 0.035
        self.socket_approach_standoff = 0.055
        self.socket_approach_lift = 0.035
        self.hose_approach_offset_local = np.array([0.0, 0.08, 0.0], dtype=np.float64)
        self.hose_engage_offset_local = np.array([0.01, 0.0, 0.0], dtype=np.float64)
        self.socket_pos_np = np.array([float(scene_builder.socket_pos[i]) for i in range(3)], dtype=np.float64)
        self.socket_rot_np = np.array([float(scene_builder.socket_rot[i]) for i in range(4)], dtype=np.float64)
        self.insertion_dir_np = _np_quat_rotate(self.socket_rot_np, np.array([0.0, 0.0, 1.0]))
        self.insertion_dir_np /= max(np.linalg.norm(self.insertion_dir_np), 1.0e-12)
        self.desired_tip_quat_np = _np_quat_multiply(
            self.socket_rot_np,
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        self.clearance_dir_np = -self.insertion_dir_np.copy()
        self.clearance_dir_np[2] = 0.0
        clearance_norm = np.linalg.norm(self.clearance_dir_np)
        if clearance_norm < 1.0e-8:
            self.clearance_dir_np = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        else:
            self.clearance_dir_np /= clearance_norm
        self.grasp_orientation_offset = _np_quat_multiply(
            _np_quat_from_axis_angle([1.0, 0.0, 0.0], np.pi / 2.0),
            _np_quat_from_axis_angle([0.0, 0.0, 1.0], -np.pi / 2.0),
        )
        self.grasp_shift = 0.004
        self.grasp_local_offset = np.array([0.0, -self.cfg.hose_radius + 0.002, 0.0], dtype=np.float64)
        self.right_open_driver, self.right_closed_driver = self._gripper_driver_targets(
            scene_builder.right_gripper_driver_dofs
        )
        self.left_open_driver, _ = self._gripper_driver_targets(scene_builder.left_gripper_driver_dofs)

        self.ee_indices = {
            "right": _find_label_index(self.model.body_label, RIGHT_EE),
            "left": _find_label_index(self.model.body_label, LEFT_EE),
            "torso": _find_label_index(self.model.body_label, TORSO),
        }
        self.right_finger_body_ids = [
            body_id
            for body_id in (
                _maybe_find_label_index(self.model.body_label, "right_gripper_leftfinger"),
                _maybe_find_label_index(self.model.body_label, "right_gripper_rightfinger"),
            )
            if body_id is not None
        ]
        self._setup_ik()
        self._seed_control_targets()
        self._enter_phase(0, 0.0)

    def _gripper_driver_targets(self, driver_dofs: list[int]) -> tuple[float, float]:
        if not driver_dofs:
            return 0.0, 0.0
        dof = driver_dofs[0]
        lower = float(self.model.joint_limit_lower.numpy()[dof])
        upper = float(self.model.joint_limit_upper.numpy()[dof])
        open_target = 0.5 * upper
        closed_target = 2.0 * 0.0036
        return max(lower, min(upper, open_target)), max(lower, min(upper, closed_target))

    def _setup_ik(self) -> None:
        body_q_np = self.state.body_q.numpy()
        weights = {"right": 1.0, "left": 1.0, "torso": 50.0}
        self.pos_objs = []
        self.rot_objs = []
        self.nominal_tfs = {}
        for name in ("right", "left", "torso"):
            body_id = self.ee_indices[name]
            tf = wp.transform(*body_q_np[body_id])
            self.nominal_tfs[name] = tf
            self.pos_objs.append(
                ik.IKObjectivePosition(
                    link_index=body_id,
                    link_offset=wp.vec3(0.0, 0.0, 0.0),
                    target_positions=wp.array([wp.transform_get_translation(tf)], dtype=wp.vec3),
                    weight=weights[name],
                )
            )
            quat = wp.transform_get_rotation(tf)
            self.rot_objs.append(
                ik.IKObjectiveRotation(
                    link_index=body_id,
                    link_offset_rotation=wp.quat_identity(),
                    target_rotations=wp.array([_quat_to_vec4(quat)], dtype=wp.vec4),
                    weight=weights[name],
                )
            )

        joint_limits = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.single_robot_model.joint_limit_lower,
            joint_limit_upper=self.single_robot_model.joint_limit_upper,
            weight=10.0,
        )
        initial_joint_q = self.single_robot_model.joint_q.numpy().astype(np.float32, copy=False)
        self.ik_joint_q = wp.array(
            initial_joint_q,
            shape=(1, self.single_robot_model.joint_coord_count),
            dtype=wp.float32,
            device=self.model.device,
        )
        self.ik_solver = ik.IKSolver(
            model=self.single_robot_model,
            n_problems=1,
            objectives=[*self.pos_objs, *self.rot_objs, joint_limits],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

    def _seed_control_targets(self) -> None:
        initial = self.single_robot_model.joint_q.numpy().astype(np.float32, copy=False)
        target_np = self.control.joint_target_pos.numpy().astype(np.float32, copy=True)
        for joint_coord_ids in self.scene_builder.robot_joint_coord_ids_by_env:
            target_np[joint_coord_ids] = initial
        wp.copy(self.control.joint_target_pos, wp.array(target_np, dtype=wp.float32, device=self.model.device))
        self.last_control_q = initial.astype(np.float32, copy=True)

    def update(self, sim_time: float) -> str:
        self.state = NewtonManager.get_state_0()
        while self._should_advance_phase(sim_time):
            self._enter_phase(min(self.phase_index + 1, len(self._PHASES) - 1), sim_time)
        target_pos, target_quat, gripper_value, task = self._target_for_phase(sim_time)
        target_pos, target_quat = self._filter_ik_target(target_pos, target_quat)
        self._set_ik_target(0, target_pos, target_quat)
        self._hold_nominal_target(1, "left")
        self._hold_nominal_target(2, "torso")
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        self._write_joint_targets(gripper_value)
        self.last_task = task
        return task

    def _enter_phase(self, phase_index: int, sim_time: float) -> None:
        self.phase_index = phase_index
        self.phase_start_time = sim_time
        self.phase_start_ee_pos, self.phase_start_ee_quat = self._ee_pose_np()
        self.phase_start_plug_pos, self.phase_start_plug_quat = self._plug_pose_np()
        phase = self._PHASES[self.phase_index][0]
        if phase in ("approach_hose", "reapproach"):
            self._reset_live_hose_target()
            self._update_grasp_local_offset()
        if phase in ("engage_hose", "reengage"):
            self._update_grasp_local_offset()
        if phase in ("align_axes", "verify_align", "insert_hose", "release_hose") or phase in ("approach_socket",):
            self.phase_target_quat = _np_quat_multiply(self.socket_rot_np, self.grasp_orientation_offset)
        else:
            # Hose grasp phases override this with a filtered live cable orientation.
            self.phase_target_quat = self.phase_start_ee_quat.copy()

    def _should_advance_phase(self, sim_time: float) -> bool:
        phase, duration = self._PHASES[self.phase_index]
        if phase == "done":
            return False
        elapsed = sim_time - self.phase_start_time
        if phase == "verify_align" and elapsed > 0.5:
            if self._verify_alignment_ok():
                self.verify_align_retries = 0
                return True
            if elapsed >= duration:
                self.verify_align_retries += 1
                self._enter_phase(self._phase_index("align_axes"), sim_time)
                return False
        return elapsed >= duration

    def _target_for_phase(self, sim_time: float):
        phase, duration = self._PHASES[self.phase_index]
        elapsed = sim_time - self.phase_start_time
        alpha = _smoothstep(elapsed / max(duration, 1.0e-6))

        if phase == "approach_hose":
            hose_pos, hose_quat = self._hose_target_pose("approach", live=True)
            target_pos = self._lerp(
                self.phase_start_ee_pos,
                hose_pos,
                alpha,
            )
            target_quat = _np_quat_slerp(self.phase_start_ee_quat, hose_quat, alpha)
            grip = 0.0
        elif phase == "engage_hose":
            hose_pos, hose_quat = self._hose_target_pose("engage", live=True)
            target_pos = self._lerp(
                self.phase_start_ee_pos,
                hose_pos,
                alpha,
            )
            target_pos = self._apply_gripper_centering(target_pos)
            target_quat = hose_quat
            grip = 0.0
        elif phase == "grasp_hose":
            target_pos, target_quat = self._hose_target_pose("engage", live=True)
            target_pos = self._apply_gripper_centering(target_pos)
            grip = _smoothstep(alpha / 0.35)
        elif phase == "hold_grasp":
            target_pos, target_quat = self._hose_target_pose("engage", live=True)
            target_pos = self._apply_gripper_centering(target_pos)
            grip = 1.0
        elif phase == "retract":
            target_pos = self.phase_start_ee_pos + alpha * self._transfer_retract_offset()
            target_quat = self.phase_start_ee_quat
            grip = 1.0
        elif phase == "settle":
            target_pos = self.phase_start_ee_pos
            target_quat = self.phase_start_ee_quat
            grip = 1.0
        elif phase == "approach_socket":
            target_pos = self._lerp(self.phase_start_ee_pos, self._socket_approach_pos(), alpha)
            target_quat = _np_quat_slerp(self.phase_start_ee_quat, self.phase_target_quat, alpha)
            grip = 1.0
        elif phase == "align_axes":
            ee_pos, ee_quat = self._ee_target_for_desired_tip(self._socket_start_pos(), self.desired_tip_quat_np)
            target_pos = self._lerp(self.phase_start_ee_pos, ee_pos, alpha)
            target_quat = _np_quat_slerp(self.phase_start_ee_quat, ee_quat, alpha)
            grip = 1.0
        elif phase == "verify_align":
            target_pos, target_quat = self._verify_alignment_target()
            grip = 1.0
        elif phase == "insert_hose":
            target_pos, target_quat = self._insert_target(alpha)
            grip = 1.0
        elif phase == "release_hose":
            target_pos = self.phase_start_ee_pos
            target_quat = self.phase_start_ee_quat
            grip = 1.0 - alpha
        elif phase == "withdraw":
            target_pos = self._lerp(
                self.phase_start_ee_pos,
                self.phase_start_ee_pos + 0.055 * self.clearance_dir_np + np.array([0.0, 0.0, 0.035]),
                alpha,
            )
            target_quat = self.phase_start_ee_quat
            grip = 0.0
        elif phase == "wait_after_withdraw":
            target_pos = self.phase_start_ee_pos
            target_quat = self.phase_start_ee_quat
            grip = 0.0
        elif phase == "reapproach":
            hose_pos, hose_quat = self._hose_target_pose("approach", live=True)
            target_pos = self._lerp(self.phase_start_ee_pos, hose_pos, alpha)
            target_quat = _np_quat_slerp(self.phase_start_ee_quat, hose_quat, alpha)
            grip = 0.0
        elif phase == "reengage":
            hose_pos, hose_quat = self._hose_target_pose("engage", live=True)
            target_pos = self._lerp(self.phase_start_ee_pos, hose_pos, alpha)
            target_pos = self._apply_gripper_centering(target_pos)
            target_quat = hose_quat
            grip = 0.0
        elif phase == "regrasp":
            target_pos, target_quat = self._hose_target_pose("engage", live=True)
            target_pos = self._apply_gripper_centering(target_pos)
            grip = _smoothstep(alpha / 0.35)
        elif phase == "pull":
            pull_vec = -self.pull_distance * self.insertion_dir_np + np.array([-0.03, 0.0, 0.0])
            target_pos = self.phase_start_ee_pos + alpha * pull_vec
            target_quat = self.phase_start_ee_quat
            grip = 1.0
        elif phase == "final_release":
            target_pos = self.phase_start_ee_pos + alpha * np.array([-0.08, 0.0, 0.04])
            target_quat = self.phase_start_ee_quat
            grip = 1.0 - alpha
        else:
            target_pos = self.phase_start_ee_pos
            target_quat = self.phase_start_ee_quat
            grip = 0.0

        return self._vec3(target_pos), self._quat(target_quat), grip, phase

    def _phase_index(self, name: str) -> int:
        for index, (phase, _duration) in enumerate(self._PHASES):
            if phase == name:
                return index
        raise ValueError(name)

    def _filter_ik_target(self, target_pos, target_quat):
        target_pos_np = np.array([float(target_pos[i]) for i in range(3)], dtype=np.float64)
        target_quat_np = np.array([float(target_quat[i]) for i in range(4)], dtype=np.float64)
        if self.last_target_pos is None:
            self.last_target_pos = target_pos_np
            self.last_target_quat = target_quat_np
        else:
            delta = target_pos_np - self.last_target_pos
            dist = np.linalg.norm(delta)
            if dist > self.max_target_step:
                target_pos_np = self.last_target_pos + delta * (self.max_target_step / dist)
            target_quat_np = _np_quat_slerp(self.last_target_quat, target_quat_np, 0.25)
            self.last_target_pos = target_pos_np
            self.last_target_quat = target_quat_np
        return self._vec3(target_pos_np), self._quat(target_quat_np)

    def _plug_pose_np(self) -> tuple[np.ndarray, np.ndarray]:
        return self._body_pose_np(self.scene_builder.plug_body_id)

    def _tip_pose_np(self) -> tuple[np.ndarray, np.ndarray]:
        return self._body_pose_np(self.scene_builder.tip_body_id)

    def _ee_pose_np(self) -> tuple[np.ndarray, np.ndarray]:
        return self._body_pose_np(self.ee_indices["right"])

    def _body_pose_np(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        body_q = self.state.body_q.numpy()[body_id]
        return body_q[:3].astype(np.float64), body_q[3:].astype(np.float64)

    def _reset_live_hose_target(self) -> None:
        plug_pos, plug_quat = self._plug_pose_np()
        self.live_hose_target_pos = plug_pos.copy()
        self.live_hose_target_quat = plug_quat.copy()

    def _tracked_plug_pose(self, live: bool) -> tuple[np.ndarray, np.ndarray]:
        if not live:
            return self.phase_start_plug_pos, self.phase_start_plug_quat
        plug_pos, plug_quat = self._plug_pose_np()
        if self.live_hose_target_pos is None or self.live_hose_target_quat is None:
            self.live_hose_target_pos = plug_pos.copy()
            self.live_hose_target_quat = plug_quat.copy()
        else:
            alpha = self.live_hose_target_alpha
            self.live_hose_target_pos = (1.0 - alpha) * self.live_hose_target_pos + alpha * plug_pos
            self.live_hose_target_quat = _np_quat_slerp(self.live_hose_target_quat, plug_quat, alpha)
        return self.live_hose_target_pos.copy(), self.live_hose_target_quat.copy()

    def _grasp_quat(self, plug_quat: np.ndarray) -> np.ndarray:
        return _np_quat_multiply(plug_quat, self.grasp_orientation_offset)

    def _update_grasp_local_offset(self) -> None:
        body_ids = self.scene_builder.primary_cable_body_ids
        if not body_ids:
            return
        plug_pos, plug_quat = self._plug_pose_np()
        body_q = self.state.body_q.numpy()
        nearest_pos = None
        nearest_dist = float("inf")
        for body_id in body_ids:
            cable_pos = body_q[body_id, :3].astype(np.float64)
            dist = float(np.linalg.norm(cable_pos - plug_pos))
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_pos = cable_pos
        if nearest_pos is None:
            return
        toward_cable_world = nearest_pos - plug_pos
        norm = np.linalg.norm(toward_cable_world)
        if norm <= 1.0e-8:
            return
        toward_cable_world /= norm
        toward_cable_local = _np_quat_rotate(_np_quat_inverse(plug_quat), toward_cable_world)
        self.grasp_local_offset = np.array(
            [
                toward_cable_local[0] * self.grasp_shift,
                -self.cfg.hose_radius + 0.002 + toward_cable_local[1] * self.grasp_shift,
                toward_cable_local[2] * self.grasp_shift,
            ],
            dtype=np.float64,
        )

    def _hose_target_pose(self, mode: str, live: bool) -> tuple[np.ndarray, np.ndarray]:
        if live:
            self._update_grasp_local_offset()
        plug_pos, plug_quat = self._tracked_plug_pose(live)
        grasp_world = _np_quat_rotate(plug_quat, self._grasp_local_offset())
        if mode == "approach":
            extra_world = _np_quat_rotate(plug_quat, self.hose_approach_offset_local)
        elif mode == "engage":
            extra_world = _np_quat_rotate(plug_quat, self.hose_engage_offset_local)
        else:
            raise ValueError(mode)
        return plug_pos + grasp_world + extra_world, self._grasp_quat(plug_quat)

    def _apply_gripper_centering(self, target_pos: np.ndarray) -> np.ndarray:
        if len(self.right_finger_body_ids) != 2:
            return target_pos
        plug_pos, plug_quat = self._plug_pose_np()
        finger_0_pos, _ = self._body_pose_np(self.right_finger_body_ids[0])
        finger_1_pos, _ = self._body_pose_np(self.right_finger_body_ids[1])

        closing_axis = finger_1_pos - finger_0_pos
        closing_axis_norm = np.linalg.norm(closing_axis)
        if closing_axis_norm < 1.0e-8:
            return target_pos
        closing_axis /= closing_axis_norm

        finger_mid = 0.5 * (finger_0_pos + finger_1_pos)
        hose_axis = _np_quat_rotate(plug_quat, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        hose_axis /= max(np.linalg.norm(hose_axis), 1.0e-12)

        delta = -self.gripper_centering_k * np.dot(plug_pos - finger_mid, closing_axis) * closing_axis
        mid_to_hose = finger_mid - plug_pos
        radial_error = mid_to_hose - np.dot(mid_to_hose, hose_axis) * hose_axis
        delta -= self.gripper_axis_centering_k * radial_error
        delta -= np.dot(delta, hose_axis) * hose_axis

        delta_norm = np.linalg.norm(delta)
        if delta_norm > self.gripper_centering_max_step:
            delta *= self.gripper_centering_max_step / delta_norm
        return target_pos + delta

    def _grasp_local_offset(self) -> np.ndarray:
        return self.grasp_local_offset.copy()

    def _transfer_retract_offset(self) -> np.ndarray:
        return self.transfer_retract_standoff * self.clearance_dir_np + np.array(
            [0.0, 0.0, self.transfer_retract_lift], dtype=np.float64
        )

    def _socket_approach_pos(self) -> np.ndarray:
        return (
            self.socket_pos_np
            + self.socket_approach_standoff * self.clearance_dir_np
            + np.array([0.0, 0.0, self.socket_approach_lift], dtype=np.float64)
        )

    def _socket_start_pos(self) -> np.ndarray:
        return self.socket_pos_np + self.insert_start_depth * self.insertion_dir_np

    def _tip_axis_np(self) -> np.ndarray:
        _, tip_quat = self._tip_pose_np()
        tip_axis = _np_quat_rotate(tip_quat, np.array([0.0, 0.0, 1.0], dtype=np.float64))
        return tip_axis / max(np.linalg.norm(tip_axis), 1.0e-12)

    def _alignment_errors(self) -> tuple[np.ndarray, float, float]:
        tip_pos, _ = self._tip_pose_np()
        delta = tip_pos - self._socket_start_pos()
        lateral = delta - np.dot(delta, self.insertion_dir_np) * self.insertion_dir_np
        axis_cosine = float(np.dot(self._tip_axis_np(), self.insertion_dir_np))
        return lateral, float(np.linalg.norm(lateral)), axis_cosine

    def _ee_target_for_desired_tip(
        self,
        desired_tip_pos: np.ndarray,
        desired_tip_quat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        tip_pos, tip_quat = self._tip_pose_np()
        ee_pos, ee_quat = self._ee_pose_np()
        tip_to_ee_pos, tip_to_ee_quat = _np_relative_transform(tip_pos, tip_quat, ee_pos, ee_quat)
        return _np_transform_point_quat(desired_tip_pos, desired_tip_quat, tip_to_ee_pos, tip_to_ee_quat)

    def _verify_alignment_target(self) -> tuple[np.ndarray, np.ndarray]:
        return self._ee_target_for_desired_tip(self._socket_start_pos(), self.desired_tip_quat_np)

    def _verify_alignment_ok(self) -> bool:
        _, lateral_error, axis_cosine = self._alignment_errors()
        return lateral_error < self.align_lateral_threshold and axis_cosine < self.align_axis_cosine_threshold

    def _insert_target(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        depth = self.insert_start_depth + alpha * (self.insert_final_depth - self.insert_start_depth)
        desired_tip_pos = self.socket_pos_np + depth * self.insertion_dir_np
        return self._ee_target_for_desired_tip(desired_tip_pos, self.desired_tip_quat_np)

    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
        return (1.0 - alpha) * a + alpha * b

    @staticmethod
    def _vec3(v: np.ndarray):
        return wp.vec3(float(v[0]), float(v[1]), float(v[2]))

    @staticmethod
    def _quat(q: np.ndarray):
        q = np.asarray(q, dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1.0e-12)
        return wp.quat(float(q[0]), float(q[1]), float(q[2]), float(q[3]))

    def _set_ik_target(self, objective_idx: int, pos, quat) -> None:
        self.pos_objs[objective_idx].set_target_position(0, pos)
        self.rot_objs[objective_idx].set_target_rotation(0, _quat_to_vec4(quat))

    def _hold_nominal_target(self, objective_idx: int, name: str) -> None:
        tf = self.nominal_tfs[name]
        self._set_ik_target(objective_idx, wp.transform_get_translation(tf), wp.transform_get_rotation(tf))

    def _write_joint_targets(self, gripper_alpha: float) -> None:
        q = self.ik_joint_q.numpy().reshape(-1).astype(np.float32, copy=True)
        self._set_gripper_targets(q, gripper_alpha)
        max_step = np.full_like(q, self.max_joint_step)
        gripper_dofs = self.scene_builder.gripper_dofs
        if gripper_dofs:
            max_step[gripper_dofs] = self._gripper_joint_step_limit()
        q = self.last_control_q + np.clip(q - self.last_control_q, -max_step, max_step)
        self.last_control_q = q.astype(np.float32, copy=True)
        limited_ik_q = wp.array(
            q,
            shape=(1, self.single_robot_model.joint_coord_count),
            dtype=wp.float32,
            device=self.model.device,
        )
        wp.copy(self.ik_joint_q, limited_ik_q)
        target_np = self.control.joint_target_pos.numpy().astype(np.float32, copy=True)
        for joint_coord_ids in self.scene_builder.robot_joint_coord_ids_by_env:
            target_np[joint_coord_ids] = q
        wp.copy(self.control.joint_target_pos, wp.array(target_np, dtype=wp.float32, device=self.model.device))

    def _set_gripper_targets(self, q: np.ndarray, right_alpha: float) -> None:
        right_alpha = max(0.0, min(1.0, right_alpha))
        right_driver = (1.0 - right_alpha) * self.right_open_driver + right_alpha * self.right_closed_driver
        self._set_gripper_side(
            q, self.scene_builder.right_gripper_driver_dofs, self.scene_builder.right_gripper_dofs, right_driver
        )
        self._set_gripper_side(
            q, self.scene_builder.left_gripper_driver_dofs, self.scene_builder.left_gripper_dofs, self.left_open_driver
        )

    @staticmethod
    def _set_gripper_side(q: np.ndarray, driver_dofs: list[int], finger_dofs: list[int], driver_target: float) -> None:
        if driver_dofs:
            q[driver_dofs[0]] = driver_target
        if len(finger_dofs) >= 2:
            q[finger_dofs[0]] = -0.5 * driver_target
            q[finger_dofs[1]] = 0.5 * driver_target

    def _gripper_joint_step_limit(self) -> float:
        if self.max_gripper_joint_velocity <= 0.0:
            return self.max_gripper_joint_step
        step_dt = 1.0 / max(float(self.cfg.fps), 1.0)
        return min(self.max_gripper_joint_step, self.max_gripper_joint_velocity * step_dt)


def setup_kit_scene(sim, scene_builder: WaterhoseSceneBuilder) -> None:
    """Create Kit-side authored scene geometry when Kit visualization is active."""
    if "kit" not in sim.resolve_visualizer_types():
        return
    _spawn_kit_robot_visuals(scene_builder)
    for env_id, origin in enumerate(scene_builder.env_origins):
        if scene_builder.num_envs == 1:
            scene_prim_path = "/World/Cable008Scene"
        else:
            scene_prim_path = f"/World/Env_{env_id}/Cable008Scene"
        if sim_utils.get_current_stage().GetPrimAtPath(scene_prim_path).IsValid():
            continue
        scene_cfg = sim_utils.UsdFileCfg(usd_path=str(scene_builder.scene_usd))
        scene_pos = wp.transform_get_translation(scene_builder.asset_xform)
        scene_rot = wp.transform_get_rotation(scene_builder.asset_xform)
        scene_cfg.func(
            scene_prim_path,
            scene_cfg,
            translation=(
                float(scene_pos[0] + origin[0]),
                float(scene_pos[1] + origin[1]),
                float(scene_pos[2] + origin[2]),
            ),
            orientation=(float(scene_rot[0]), float(scene_rot[1]), float(scene_rot[2]), float(scene_rot[3])),
        )
    _spawn_kit_cable_curve_visuals(scene_builder)
    install_kit_cable_curve_pre_render_sync(scene_builder)
    if not sim_utils.get_current_stage().GetPrimAtPath("/World/DomeLight").IsValid():
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/DomeLight", light_cfg)
    configure_kit_camera_view(sim)


def configure_kit_camera_view(sim) -> None:
    """Set the Kit perspective camera to the hardcoded waterhose task view."""
    if "kit" not in sim.resolve_visualizer_types():
        return

    stage = sim_utils.get_current_stage()
    if stage is None:
        return

    for visualizer in getattr(sim, "visualizers", ()):
        viewport_api = getattr(visualizer, "_viewport_api", None)
        if viewport_api is not None:
            try:
                viewport_api.set_active_camera(KIT_CAMERA_PRIM_PATH)
            except Exception:
                pass
        if hasattr(visualizer, "_controlled_camera_path"):
            visualizer._controlled_camera_path = KIT_CAMERA_PRIM_PATH
    _set_kit_camera_prim_transform(stage)


def _set_kit_camera_prim_transform(stage) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: PLC0415

    edit_layer = stage.GetSessionLayer() or stage.GetRootLayer()
    with Usd.EditContext(stage, edit_layer):
        prim = stage.GetPrimAtPath(KIT_CAMERA_PRIM_PATH)
        if not prim.IsValid():
            prim = UsdGeom.Camera.Define(stage, KIT_CAMERA_PRIM_PATH).GetPrim()
        if not prim.IsValid():
            return

        translate_attr = _typed_camera_attr(prim, "xformOp:translate", Sdf.ValueTypeNames.Double3)
        rotate_attr = prim.GetAttribute("xformOp:rotateXYZ")
        if rotate_attr.IsValid() and rotate_attr.GetTypeName() not in (
            Sdf.ValueTypeNames.Float3,
            Sdf.ValueTypeNames.Double3,
        ):
            rotate_attr.SetTypeName(Sdf.ValueTypeNames.Float3)
        if not rotate_attr.IsValid() or rotate_attr.GetTypeName() not in (
            Sdf.ValueTypeNames.Float3,
            Sdf.ValueTypeNames.Double3,
        ):
            rotate_attr = prim.CreateAttribute("xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3, False)
        order_attr = _typed_camera_attr(prim, "xformOpOrder", Sdf.ValueTypeNames.TokenArray)

        translate_attr.Set(Gf.Vec3d(*[float(v) for v in KIT_CAMERA_TRANSLATION]))
        if rotate_attr.GetTypeName() == Sdf.ValueTypeNames.Double3:
            rotate_attr.Set(Gf.Vec3d(*[float(v) for v in KIT_CAMERA_ROTATION_XYZ_DEG]))
        else:
            rotate_attr.Set(Gf.Vec3f(*[float(v) for v in KIT_CAMERA_ROTATION_XYZ_DEG]))
        order_attr.Set(["xformOp:translate", "xformOp:rotateXYZ"])


def _typed_camera_attr(prim, name: str, type_name):
    attr = prim.GetAttribute(name)
    if attr.IsValid() and attr.GetTypeName() != type_name:
        attr.SetTypeName(type_name)
    if not attr.IsValid() or attr.GetTypeName() != type_name:
        attr = prim.CreateAttribute(name, type_name, False)
    return attr


def install_kit_cable_curve_pre_render_sync(scene_builder: WaterhoseSceneBuilder) -> None:
    """Update Kit cable curves before Newton flushes Fabric transforms for rendering."""
    register = getattr(NewtonManager, "register_pre_render_callback", None)
    if register is None:
        return
    callback_name = _kit_cable_curve_sync_callback_name(scene_builder)

    def sync_curves() -> None:
        sync_kit_cable_curves_from_newton(scene_builder)

    register(callback_name, sync_curves)


def remove_kit_cable_curve_pre_render_sync(scene_builder: WaterhoseSceneBuilder) -> None:
    deregister = getattr(NewtonManager, "deregister_pre_render_callback", None)
    if deregister is not None:
        deregister(_kit_cable_curve_sync_callback_name(scene_builder))


def sync_kit_cable_curves_from_newton(scene_builder: WaterhoseSceneBuilder) -> None:
    """Mirror the simulated Newton hose centerlines into the Kit BasisCurves."""
    if sim_utils is None or NewtonManager is None:
        return
    cable_body_ids_by_env_by_curve = _cable_body_ids_by_env_by_curve(scene_builder)
    if not cable_body_ids_by_env_by_curve:
        return

    stage = sim_utils.get_current_stage()
    if not stage:
        return
    try:
        body_q = NewtonManager.get_state_0().body_q.numpy()
    except Exception:
        return

    segment_lengths_by_env_by_curve = _cable_segment_lengths_by_env_by_curve(scene_builder)
    for env_id, cable_body_ids_by_curve in enumerate(cable_body_ids_by_env_by_curve):
        if env_id >= len(segment_lengths_by_env_by_curve):
            continue
        for curve_index, body_ids in enumerate(cable_body_ids_by_curve):
            if (
                not body_ids
                or curve_index >= len(scene_builder.cable_curve_prim_paths)
                or curve_index >= len(segment_lengths_by_env_by_curve[env_id])
            ):
                continue
            curve_prim_path = scene_builder.cable_curve_prim_paths[curve_index]
            segment_lengths = segment_lengths_by_env_by_curve[env_id][curve_index]
            kit_curve_path = _kit_cable_curve_path(scene_builder, curve_prim_path, env_id)
            prim = stage.GetPrimAtPath(kit_curve_path)
            if not prim.IsValid():
                continue
            points = _cable_curve_points_from_newton_body_q(body_q, body_ids, segment_lengths)
            if points is None:
                continue
            _set_kit_basis_curve_points(stage, prim, points, radius=float(scene_builder.cfg.hose_radius))


def _spawn_kit_cable_curve_visuals(scene_builder: WaterhoseSceneBuilder) -> None:
    """Spawn the authored cable BasisCurves as Kit-only deformable visuals."""
    stage = sim_utils.get_current_stage()
    spawned_assets: set[tuple[str, str]] = set()
    for env_id in range(scene_builder.num_envs):
        root_path = _kit_cable_curve_root_path(scene_builder, env_id)
        if scene_builder.num_envs > 1:
            _define_xform_path(stage, f"/World/Env_{env_id}")
        for cable_asset in scene_builder.cable_assets:
            spawn_path = _kit_cable_asset_spawn_path(scene_builder, cable_asset, env_id)
            asset_key = (str(cable_asset.usd_path), spawn_path)
            if asset_key in spawned_assets or stage.GetPrimAtPath(spawn_path).IsValid():
                continue
            spawned_assets.add(asset_key)
            parent_path = spawn_path.rsplit("/", 1)[0]
            if parent_path:
                _define_xform_path(stage, parent_path)
            cable_cfg = sim_utils.UsdFileCfg(usd_path=str(cable_asset.usd_path))
            cable_cfg.func(
                spawn_path,
                cable_cfg,
                translation=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0, 1.0),
            )
            _disable_kit_robot_physics(stage, spawn_path)
        _prepare_kit_cable_curve_payload(stage, root_path)
        _disable_kit_robot_physics(stage, root_path)


def _prepare_kit_cable_curve_payload(stage, root_path: str) -> None:
    """Keep cable payload visuals renderable while Newton drives dynamic parts."""
    from pxr import UsdGeom  # noqa: PLC0415

    for rel_path in ("plug_mesh", "plug_mesh/plug_mesh", "plug_mesh02", "plug_mesh02/plug_mesh"):
        prim = stage.GetPrimAtPath(f"{root_path}/{rel_path}")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeVisible()


def _kit_cable_curve_sync_callback_name(scene_builder: WaterhoseSceneBuilder) -> str:
    return f"waterhose_kit_cable_curves_{id(scene_builder)}"


def _cable_body_ids_by_env_by_curve(scene_builder: WaterhoseSceneBuilder) -> list[list[list[int]]]:
    by_env = getattr(scene_builder, "cable_body_ids_by_env_by_curve", [])
    if by_env:
        return by_env
    return [[list(body_ids) for body_ids in scene_builder.cable_body_ids_by_curve]]


def _cable_segment_lengths_by_env_by_curve(scene_builder: WaterhoseSceneBuilder) -> list[list[list[float]]]:
    by_env = getattr(scene_builder, "cable_segment_lengths_by_env_by_curve", [])
    if by_env:
        return by_env
    return [[list(lengths) for lengths in scene_builder.cable_segment_lengths_by_curve]]


def _kit_cable_curve_root_path(scene_builder: WaterhoseSceneBuilder, env_id: int = 0) -> str:
    if scene_builder.num_envs == 1:
        return KIT_CABLE_CURVE_ROOT
    return f"/World/Env_{env_id}/WaterhoseCableCurves"


def _kit_cable_asset_spawn_path(
    scene_builder: WaterhoseSceneBuilder, cable_asset: CableUsdAsset, env_id: int = 0
) -> str:
    root_path = _kit_cable_curve_root_path(scene_builder, env_id)
    first_component = cable_asset.curve_prim_path.strip("/").split("/", 1)[0]
    if not first_component or first_component == "World":
        return root_path
    return f"{root_path}/{first_component}"


def _kit_cable_curve_path(scene_builder: WaterhoseSceneBuilder, source_curve_prim_path: str, env_id: int = 0) -> str:
    root_path = _kit_cable_curve_root_path(scene_builder, env_id)
    suffix = source_curve_prim_path.strip("/")
    if suffix == "World":
        suffix = ""
    elif suffix.startswith("World/"):
        suffix = suffix[len("World/") :]
    return root_path if not suffix else f"{root_path}/{suffix}"


def _cable_curve_points_from_newton_body_q(
    body_q: object, body_ids: list[int], segment_lengths: list[float]
) -> object | None:
    """Return BasisCurves points from Newton rod body poses."""
    body_indices = np.asarray(body_ids, dtype=np.int64)
    if np.any(body_indices < 0) or np.any(body_indices >= body_q.shape[0]):
        return None
    poses = np.asarray(body_q[body_indices], dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] < 7:
        return None

    starts = poses[:, 0:3]
    quats_xyzw = poses[:, 3:7]
    lengths = np.asarray(segment_lengths, dtype=np.float64)
    if lengths.shape[0] != starts.shape[0]:
        if lengths.shape[0] > starts.shape[0]:
            lengths = lengths[: starts.shape[0]]
        elif lengths.shape[0] > 0:
            lengths = np.pad(lengths, (0, starts.shape[0] - lengths.shape[0]), mode="edge")
        else:
            lengths = np.full(starts.shape[0], 0.01, dtype=np.float64)

    points = np.empty((starts.shape[0] + 1, 3), dtype=np.float32)
    points[:-1] = starts.astype(np.float32)
    last_offset = _quat_rotate_xyzw(quats_xyzw[-1], np.array([0.0, 0.0, float(lengths[-1])], dtype=np.float64))
    points[-1] = (starts[-1] + last_offset).astype(np.float32)

    adjacent_steps = np.linalg.norm(points[1:].astype(np.float64) - points[:-1].astype(np.float64), axis=1)
    max_segment_length = float(np.max(lengths)) if lengths.size else 0.01
    max_reasonable_step = max(0.05, 20.0 * max_segment_length)
    invalid_points = (
        not np.all(np.isfinite(points))
        or float(np.max(np.abs(points))) > 20.0
        or bool(adjacent_steps.size and float(np.max(adjacent_steps)) > max_reasonable_step)
    )
    if invalid_points:
        return None
    return points


def _quat_rotate_xyzw(quat_xyzw: object, vec: object) -> object:
    q = np.asarray(quat_xyzw, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        return v
    q = q / norm
    q_xyz = q[:3]
    q_w = float(q[3])
    t = 2.0 * np.cross(q_xyz, v)
    return v + q_w * t + np.cross(q_xyz, t)


def _set_kit_basis_curve_points(stage, prim, points: object, radius: float) -> None:
    from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415

    curve = UsdGeom.BasisCurves(prim)
    if not curve:
        return

    points_np = np.ascontiguousarray(points, dtype=np.float32)
    point_count = int(points_np.shape[0])
    widths = [float(2.0 * radius)] * point_count
    extent_np = np.ascontiguousarray(
        np.stack((points_np.min(axis=0) - radius, points_np.max(axis=0) + radius), axis=0), dtype=np.float32
    )

    points_value = Vt.Vec3fArray.FromNumpy(points_np)
    extent_value = Vt.Vec3fArray.FromNumpy(extent_np)
    curve.GetPointsAttr().Set(points_value)
    curve.GetCurveVertexCountsAttr().Set([point_count])
    curve.GetWidthsAttr().Set(widths)
    UsdGeom.Boundable(prim).CreateExtentAttr().Set(extent_value)

    connections_attr = prim.GetAttribute("connections")
    if connections_attr:
        try:
            connections_attr.Set([Gf.Vec2i(index, index + 1) for index in range(point_count - 1)])
        except Exception:
            pass

    gprim = UsdGeom.Gprim(prim)
    if gprim and not gprim.GetDisplayColorAttr().HasAuthoredValue():
        gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(0.02, 0.16, 0.28)])


def _spawn_kit_robot_visuals(scene_builder: WaterhoseSceneBuilder) -> None:
    """Spawn the RBY1DF visual USD so Newton body transforms have visible Kit geometry."""
    stage = sim_utils.get_current_stage()
    robot_usd_dir = scene_builder.asset_root / "_generated" / "RBY1DF"
    robot_usd_path = robot_usd_dir / "robot_edited" / "robot_edited.usda"
    robot_cfg = None
    if robot_usd_path.is_file():
        robot_cfg = sim_utils.UsdFileCfg(usd_path=str(robot_usd_path), variants={"Physics": "none"})
    else:
        robot_cfg = sim_utils.UrdfFileCfg(
            asset_path=str(scene_builder.robot_urdf),
            usd_dir=str(robot_usd_dir),
            fix_base=False,
            merge_fixed_joints=True,
            make_instanceable=True,
            self_collision=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        )

    for env_id, origin in enumerate(scene_builder.env_origins):
        root_path = ROBOT_PRIM_PATH if scene_builder.num_envs == 1 else f"/World/Env_{env_id}/RBY1DF"
        if stage.GetPrimAtPath(root_path).IsValid():
            continue
        if scene_builder.num_envs > 1:
            _define_xform_path(stage, f"/World/Env_{env_id}")
        robot_cfg.func(
            root_path,
            robot_cfg,
            translation=(float(origin[0]), float(origin[1]), float(origin[2])),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )
        _disable_kit_robot_physics(stage, root_path)


def _disable_kit_robot_physics(stage, root_path: str) -> None:
    """Keep the spawned robot as Kit visual geometry; Newton drives the transforms."""
    from pxr import Sdf, Usd  # noqa: PLC0415

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return

    bool_type = Sdf.ValueTypeNames.Bool
    for prim in Usd.PrimRange(root_prim):
        applied_schemas = set(prim.GetAppliedSchemas())
        if "PhysicsRigidBodyAPI" in applied_schemas:
            prim.CreateAttribute("physics:rigidBodyEnabled", bool_type, False).Set(False)
            prim.CreateAttribute("physics:kinematicEnabled", bool_type, False).Set(True)
        if "PhysicsCollisionAPI" in applied_schemas:
            prim.CreateAttribute("physics:collisionEnabled", bool_type, False).Set(False)
        if "PhysicsArticulationRootAPI" in applied_schemas or "PhysxArticulationAPI" in applied_schemas:
            prim.CreateAttribute("physxArticulation:articulationEnabled", bool_type, False).Set(False)


def prepare_kit_scene_for_newton_sync(sim, scene_builder: WaterhoseSceneBuilder, builder) -> None:
    """Create Kit prims and relabel Newton bodies before Fabric sync is initialized."""
    if "kit" not in sim.resolve_visualizer_types():
        return

    import_isaaclab_runtime_dependencies()
    setup_kit_scene(sim, scene_builder)
    stage = sim_utils.get_current_stage()

    _relabel_kit_robot_bodies(stage, scene_builder, builder)
    _relabel_kit_scene_bodies(stage, scene_builder, builder)
    _relabel_kit_cable_head_bodies(stage, scene_builder, builder)
    _define_missing_kit_body_prims(stage, builder)


def _define_xform_path(stage, prim_path: str) -> None:
    """Define an Xform and any missing parents at ``prim_path``."""
    from pxr import UsdGeom  # noqa: PLC0415

    current_path = ""
    for component in prim_path.strip("/").split("/"):
        if not component:
            continue
        current_path += "/" + component
        if not stage.GetPrimAtPath(current_path).IsValid():
            UsdGeom.Xform.Define(stage, current_path)


def _resolve_kit_body_prim_path(stage, root_path: str, body_name: str) -> str:
    """Find a body prim below ``root_path``, falling back to a new Xform."""
    from pxr import Usd  # noqa: PLC0415

    direct_path = f"{root_path}/{body_name}"
    if stage.GetPrimAtPath(direct_path).IsValid():
        return direct_path

    root_prim = stage.GetPrimAtPath(root_path)
    if root_prim.IsValid():
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == body_name:
                return str(prim.GetPath())

    _define_xform_path(stage, direct_path)
    return direct_path


def _relabel_kit_robot_bodies(stage, scene_builder: WaterhoseSceneBuilder, builder) -> None:
    """Relabel Newton robot bodies to Kit Xforms for Fabric transform sync."""
    robot_body_ids = list(scene_builder._mujoco_body_ids)
    if not robot_body_ids:
        return
    robot_body_count = max(1, len(robot_body_ids) // max(1, scene_builder.num_envs))
    for env_id in range(scene_builder.num_envs):
        root_path = ROBOT_PRIM_PATH if scene_builder.num_envs == 1 else f"/World/Env_{env_id}/RBY1DF"
        start = env_id * robot_body_count
        stop = min(start + robot_body_count, len(robot_body_ids))
        for body_id in robot_body_ids[start:stop]:
            body_name = builder.body_label[body_id].rsplit("/", 1)[-1]
            builder.body_label[body_id] = _resolve_kit_body_prim_path(stage, root_path, body_name)


def _relabel_kit_scene_bodies(stage, scene_builder: WaterhoseSceneBuilder, builder) -> None:
    """Relabel imported static scene bodies to the spawned Kit scene prims."""
    scene_body_ids = [body_id for body_id, label in enumerate(builder.body_label) if label.startswith("/root/")]
    if not scene_body_ids:
        return
    bodies_per_env = max(1, len(scene_body_ids) // max(1, scene_builder.num_envs))
    for index, body_id in enumerate(scene_body_ids):
        env_id = min(index // bodies_per_env, scene_builder.num_envs - 1)
        root_path = "/World/Cable008Scene" if scene_builder.num_envs == 1 else f"/World/Env_{env_id}/Cable008Scene"
        body_name = builder.body_label[body_id].rsplit("/", 1)[-1]
        builder.body_label[body_id] = _resolve_kit_body_prim_path(stage, root_path, body_name)


def _relabel_kit_cable_head_bodies(stage, scene_builder: WaterhoseSceneBuilder, builder) -> None:
    """Relabel Newton cable head bodies to the authored Kit plug mesh Xforms."""
    head_body_ids_by_env = getattr(scene_builder, "cable_head_body_ids_by_env", [])
    if not head_body_ids_by_env:
        head_body_ids_by_env = [scene_builder.cable_head_body_ids]

    for env_id, head_body_ids in enumerate(head_body_ids_by_env):
        root_path = _kit_cable_curve_root_path(scene_builder, env_id)
        head_visual_paths = [
            f"{root_path}/plug_mesh/plug_mesh",
            f"{root_path}/plug_mesh02/plug_mesh",
        ]
        for body_id, visual_path in zip(head_body_ids, head_visual_paths):
            if body_id >= len(builder.body_label):
                continue
            if stage.GetPrimAtPath(visual_path).IsValid():
                builder.body_label[body_id] = visual_path


def _define_missing_kit_body_prims(stage, builder) -> None:
    """Ensure every Newton body label is an existing absolute USD prim path."""
    for body_id, body_label in enumerate(builder.body_label):
        if body_label.startswith("/") and stage.GetPrimAtPath(body_label).IsValid():
            continue
        if body_label.startswith("/") and not stage.GetPrimAtPath(body_label).IsValid():
            prim_path = body_label
        else:
            body_name = body_label.rsplit("/", 1)[-1]
            prim_path = f"/World/NewtonBodies/Body_{body_id:04d}_{body_name}"
            builder.body_label[body_id] = prim_path
        _define_xform_path(stage, prim_path)


def configure_newton_viewer(sim) -> None:
    """Enable useful interaction and contact display in the Newton viewer."""
    visualizer_types = set(sim.resolve_visualizer_types()) if hasattr(sim, "resolve_visualizer_types") else set()
    if "newton" in visualizer_types and not sim.visualizers and hasattr(sim, "initialize_visualizers"):
        sim.initialize_visualizers()

    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is None:
            continue
        if hasattr(viewer, "show_contacts"):
            viewer.show_contacts = True
        if hasattr(viewer, "picking_enabled"):
            viewer.picking_enabled = True
        if hasattr(viewer, "set_camera"):
            viewer.set_camera(wp.vec3(-1.2, -2.8, 1.6), pitch=-18.0, yaw=-300.0)


def apply_viewer_forces(sim) -> None:
    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is not None and hasattr(viewer, "apply_forces"):
            viewer.apply_forces(NewtonManager.get_state_0())


def keep_running(sim, step_count: int) -> bool:
    if args_cli.max_steps >= 0 and step_count >= args_cli.max_steps:
        return False
    if not sim.visualizers:
        return True
    return any(not viz.is_closed and viz.is_running() for viz in sim.visualizers)


def log_progress(step_count: int, task: str, scene_builder: WaterhoseSceneBuilder) -> None:
    if args_cli.log_interval <= 0 or step_count % args_cli.log_interval != 0:
        return
    state = NewtonManager.get_state_0()
    body_q = state.body_q.numpy()
    hose_pos = body_q[scene_builder.grasp_body_id, 0:3]
    tip_q = body_q[scene_builder.tip_body_id]
    tip_pos = tip_q[0:3]
    tip_axis = _np_quat_rotate(tip_q[3:], np.array([0.0, 0.0, 1.0], dtype=np.float64))
    socket_pos = np.array([float(scene_builder.socket_pos[i]) for i in range(3)], dtype=np.float64)
    socket_rot = np.array([float(scene_builder.socket_rot[i]) for i in range(4)], dtype=np.float64)
    insertion_dir = _np_quat_rotate(socket_rot, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    insertion_dir /= max(np.linalg.norm(insertion_dir), 1.0e-12)
    tip_delta = tip_pos - (socket_pos + 0.005 * insertion_dir)
    tip_lateral = tip_delta - np.dot(tip_delta, insertion_dir) * insertion_dir
    tip_lateral_mm = 1000.0 * float(np.linalg.norm(tip_lateral))
    tip_axis_dot = float(np.dot(tip_axis / max(np.linalg.norm(tip_axis), 1.0e-12), insertion_dir))
    right_ee_id = _find_label_index(NewtonManager.get_model().body_label, RIGHT_EE)
    right_ee_pos = body_q[right_ee_id, 0:3]
    ee_to_hose = float(np.linalg.norm(right_ee_pos - hose_pos))
    wrenches = NewtonCoupledManager.get_proxy_body_wrenches(ROBOT_ENTRY, HOSE_ENTRY)
    wrench_norm = 0.0
    if wrenches is not None:
        wrench_np = wrenches.numpy()
        proxy_body_ids = [body_id for body_id in scene_builder.proxy_body_ids if body_id < wrench_np.shape[0]]
        if proxy_body_ids:
            wrench_norm = float(np.linalg.norm(wrench_np[proxy_body_ids, 0:3]))
    print(
        "[INFO]: "
        f"step={step_count:05d} task={task:<15} "
        f"hose_grasp=({hose_pos[0]:.3f}, {hose_pos[1]:.3f}, {hose_pos[2]:.3f}) "
        f"ee_dist={ee_to_hose:.3f}m "
        f"tip_lat={tip_lateral_mm:.2f}mm tip_dot={tip_axis_dot:.3f} "
        f"|F_proxy|={wrench_norm:.2f}N",
        flush=True,
    )


def run_simulator(sim, controller: WaterhoseIKController, scene_builder: WaterhoseSceneBuilder) -> None:
    step_count = 0
    while keep_running(sim, step_count):
        apply_viewer_forces(sim)
        task = controller.update(step_count / args_cli.fps)
        sim.step(render=False)
        log_progress(step_count, task, scene_builder)
        if sim.is_rendering:
            sync_kit_cable_curves_from_newton(scene_builder)
            sim.render()
        step_count += 1


def initialize_waterhose_runtime(sim, scene_builder: WaterhoseSceneBuilder, builder) -> None:
    """Install the Newton builder and run waterhose-specific post-reset hooks."""
    NewtonManager.set_builder(builder)
    prefer_active_python_site_packages()
    sim.reset()
    scene_builder.configure_runtime_vbd_solver()
    scene_builder.apply_runtime_cable_asset_xform()
    if sim.is_rendering:
        sync_kit_cable_curves_from_newton(scene_builder)
        NewtonManager.sync_transforms_to_usd()
    configure_newton_viewer(sim)
    configure_kit_camera_view(sim)


def body_pose_np(body_id: int):
    """Return a Newton body pose as numpy ``(position, quaternion_xyzw)``."""
    state = NewtonManager.get_state_0()
    q = state.body_q.numpy()[body_id]
    return q[:3].copy(), q[3:].copy()
