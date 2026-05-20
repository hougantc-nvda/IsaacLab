# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Run ANYmal-C walking over implicit MPM sand through Isaac Lab.

This mirrors Newton's ``mpm_anymal`` example, but the application, simulation
context, visualizer selection, and Newton manager lifecycle are owned by Isaac
Lab.  The robot is advanced by Isaac Lab's Newton MJWarp manager, while a
``SolverImplicitMPM`` instance is attached to the same Newton model to update
the sand after each robot step.

.. code-block:: bash

    # headless/no visualizer, suitable for training-style runs
    ./isaaclab.sh -p scripts/demos/newton_anymal_mpm_sand.py

    # native Newton visualizer for debugging particles and robot motion
    ./isaaclab.sh -p scripts/demos/newton_anymal_mpm_sand.py --viz newton

    # Isaac Sim Kit plus Newton visualizers
    ./isaaclab.sh -p scripts/demos/newton_anymal_mpm_sand.py --viz kit,newton

The demo spawns Isaac Lab's ANYmal-C USD for Kit visuals and drives those prims
from Newton body transforms.  MPM particles are shown in Kit as a USD
``Points`` cloud, and the Newton visualizer can render the native Newton model
and particles directly.
"""

import argparse
from dataclasses import dataclass
from types import SimpleNamespace

from isaaclab_tasks.utils.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="ANYmal-C walking over Newton implicit MPM sand.")
parser.add_argument("--voxel-size", type=float, default=0.05, help="MPM grid voxel size in meters.")
parser.add_argument("--particles-per-cell", type=float, default=3.0, help="Sand particles per grid cell.")
parser.add_argument("--grid-type", choices=["sparse", "dense", "fixed"], default="sparse", help="MPM grid type.")
parser.add_argument("--tolerance", type=float, default=1.0e-6, help="MPM rheology solver tolerance.")
parser.add_argument("--mpm-iterations", type=int, default=50, help="Maximum MPM rheology iterations.")
parser.add_argument("--robot-substeps", type=int, default=4, help="MJWarp substeps per policy/control frame.")
parser.add_argument("--fps", type=float, default=50.0, help="Control/render frames per second.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of independent ANYmal/sand environments to spawn.")
parser.add_argument("--env-spacing", type=float, default=4.0, help="Spacing between replicated environments in meters.")
parser.add_argument("--command-forward", type=float, default=1.0, help="Commanded forward velocity.")
parser.add_argument("--command-lateral", type=float, default=0.0, help="Commanded lateral velocity.")
parser.add_argument("--command-yaw", type=float, default=0.0, help="Commanded yaw velocity.")
parser.add_argument("--max-steps", type=int, default=-1, help="Stop after this many frames; negative runs forever.")
parser.add_argument("--policy", type=str, default=None, help="Optional path to a TorchScript ANYmal policy.")
parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable MJWarp CUDA graph capture.")
parser.add_argument(
    "--kit-sand-stride",
    type=int,
    default=1,
    help="Render every Nth sand particle in Kit; 1 renders every particle.",
)
add_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be >= 1.")
if args_cli.env_spacing <= 0.0:
    parser.error("--env-spacing must be > 0.")

import warnings
import xml.etree.ElementTree as ET

np = torch = wp = newton = sim_utils = None
SolverImplicitMPM = SolverMuJoCo = None
MJWarpSolverCfg = NewtonCfg = NewtonManager = None


LAB_TO_MUJOCO = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
MUJOCO_TO_LAB = [0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11]


@dataclass
class DemoModelMetadata:
    """Index metadata for the replicated Newton worlds."""

    joint_q_ids: list[list[int]]
    joint_qd_ids: list[list[int]]
    particle_ids: list[list[int]]
    env_origins: list[tuple[float, float, float]]


def import_runtime_dependencies() -> None:
    """Import Newton/Isaac Lab modules after Kit has launched when requested."""
    global np, torch, wp, newton, sim_utils, SolverImplicitMPM, SolverMuJoCo
    global MJWarpSolverCfg, NewtonCfg, NewtonManager

    import newton as newton_module
    import newton.utils  # noqa: F401
    import numpy as np_module
    import torch as torch_module
    import warp as wp_module
    from isaaclab_newton.physics import (
        MJWarpSolverCfg as MJWarpSolverCfgClass,
    )
    from isaaclab_newton.physics import (
        NewtonCfg as NewtonCfgClass,
    )
    from isaaclab_newton.physics import (
        NewtonManager as NewtonManagerClass,
    )
    from newton.solvers import SolverImplicitMPM as SolverImplicitMPMClass
    from newton.solvers import SolverMuJoCo as SolverMuJoCoClass

    import isaaclab.sim as sim_utils_module

    np = np_module
    torch = torch_module
    wp = wp_module
    newton = newton_module
    sim_utils = sim_utils_module
    SolverImplicitMPM = SolverImplicitMPMClass
    SolverMuJoCo = SolverMuJoCoClass
    MJWarpSolverCfg = MJWarpSolverCfgClass
    NewtonCfg = NewtonCfgClass
    NewtonManager = NewtonManagerClass


def quat_mul(q_a: np.ndarray, q_b: np.ndarray) -> np.ndarray:
    """Multiply quaternions in XYZW order."""
    ax, ay, az, aw = q_a
    bx, by, bz, bw = q_b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a vector by a quaternion in XYZW order."""
    q_vec = q[:3]
    t = 2.0 * np.cross(q_vec, v)
    return v + q[3] * t + np.cross(q_vec, t)


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Convert URDF RPY angles to an XYZW quaternion."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def compose_transform(
    parent_pos: np.ndarray, parent_quat: np.ndarray, child_pos: np.ndarray, child_quat: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compose parent and child transforms represented as position plus XYZW quaternion."""
    pos = parent_pos + quat_rotate(parent_quat, child_pos)
    quat = quat_mul(parent_quat, child_quat)
    quat /= np.linalg.norm(quat)
    return pos, quat


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` by inverse quaternion ``q``. Quaternions use XYZW order."""
    q_w = q[..., 3]
    q_vec = q[..., :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


def compute_obs(
    actions: torch.Tensor,
    state: newton.State,
    joint_pos_initial: torch.Tensor,
    joint_q_ids: torch.Tensor,
    joint_qd_ids: torch.Tensor,
    indices: torch.Tensor,
    gravity_vec: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Build the observation vector expected by the Newton ANYmal walking policy."""
    joint_q = wp.to_torch(state.joint_q)[joint_q_ids]
    joint_qd = wp.to_torch(state.joint_qd)[joint_qd_ids]
    root_quat_w = joint_q[:, 3:7]
    root_lin_vel_w = joint_qd[:, :3]
    root_ang_vel_w = joint_qd[:, 3:6]
    joint_pos_current = joint_q[:, 7:]
    joint_vel_current = joint_qd[:, 6:]

    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    ang_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav_b = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = torch.index_select(joint_pos_current - joint_pos_initial, 1, indices)
    joint_vel_rel = torch.index_select(joint_vel_current, 1, indices)
    return torch.cat([vel_b, ang_vel_b, grav_b, command, joint_pos_rel, joint_vel_rel, actions], dim=1)


def spawn_sand(builder: newton.ModelBuilder, voxel_size: float, particles_per_cell: float) -> None:
    """Add a shallow sand bed in front of ANYmal."""
    density = 2500.0
    particle_lo = np.array([-0.5, -0.5, 0.0])
    particle_hi = np.array([0.5, 2.5, 0.15])
    particle_res = np.array(np.ceil(particles_per_cell * (particle_hi - particle_lo) / voxel_size), dtype=int)
    cell_size = (particle_hi - particle_lo) / particle_res
    cell_volume = float(np.prod(cell_size))
    radius = float(np.max(cell_size) * 0.5)
    mass = float(cell_volume * density)

    builder.add_particle_grid(
        pos=wp.vec3(particle_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(particle_res[0]) + 1,
        dim_y=int(particle_res[1]) + 1,
        dim_z=int(particle_res[2]) + 1,
        cell_x=float(cell_size[0]),
        cell_y=float(cell_size[1]),
        cell_z=float(cell_size[2]),
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
    )


def setup_kit_scene(sim: sim_utils.SimulationContext) -> None:
    """Create simple Kit-only scene visuals for the demo."""
    if "kit" not in sim.resolve_visualizer_types():
        return

    stage = sim_utils.get_current_stage()
    if not stage.GetPrimAtPath("/World/Ground").IsValid():
        ground_cfg = sim_utils.GroundPlaneCfg(size=(12.0, 12.0), color=(0.46, 0.38, 0.24))
        ground_cfg.func("/World/Ground", ground_cfg)

    if not stage.GetPrimAtPath("/World/DomeLight").IsValid():
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/DomeLight", light_cfg)


def resolve_kit_body_prim_path(stage, root_path: str, body_name: str) -> str:
    """Find the USD prim that owns a robot body visual, falling back to a new Xform."""
    from pxr import Usd, UsdGeom  # noqa: PLC0415

    direct_path = f"{root_path}/{body_name}"
    if stage.GetPrimAtPath(direct_path).IsValid():
        return direct_path

    root_prim = stage.GetPrimAtPath(root_path)
    if root_prim.IsValid():
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == body_name:
                return str(prim.GetPath())

    UsdGeom.Xform.Define(stage, direct_path)
    return direct_path


def parse_fixed_link_offsets(
    urdf_path: str, dynamic_body_names: set[str]
) -> dict[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Map each dynamic Newton body link to fixed-descendant visual links and their body-frame offsets."""
    root = ET.parse(urdf_path).getroot()
    fixed_children: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}

    for joint in root.findall("joint"):
        if joint.attrib.get("type") != "fixed":
            continue
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        origin = joint.find("origin")
        xyz = np.zeros(3, dtype=np.float64)
        rpy = np.zeros(3, dtype=np.float64)
        if origin is not None:
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ", dtype=np.float64)
        fixed_children.setdefault(parent.attrib["link"], []).append((child.attrib["link"], xyz, rpy_to_quat(rpy)))

    offsets: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {}
    for body_name in dynamic_body_names:
        stack = [(body_name, np.zeros(3, dtype=np.float64), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))]
        visited = {body_name}
        while stack:
            parent_link, parent_pos, parent_quat = stack.pop()
            for child_link, child_pos_local, child_quat_local in fixed_children.get(parent_link, []):
                if child_link in visited or child_link in dynamic_body_names:
                    continue
                child_pos, child_quat = compose_transform(parent_pos, parent_quat, child_pos_local, child_quat_local)
                offsets.setdefault(body_name, []).append((child_link, child_pos, child_quat))
                visited.add(child_link)
                stack.append((child_link, child_pos, child_quat))

    return offsets


def compute_env_origins(num_envs: int, spacing: float) -> list[np.ndarray]:
    """Lay out environment origins on a compact XY grid."""
    columns = int(np.ceil(np.sqrt(num_envs)))
    origins = []
    for env_id in range(num_envs):
        row, column = divmod(env_id, columns)
        origins.append(np.array([column * spacing, row * spacing, 0.0], dtype=np.float64))
    return origins


def create_kit_body_prims(
    builder: newton.ModelBuilder,
    root_path: str,
    translation: tuple[float, float, float],
    body_start: int,
    body_end: int,
) -> None:
    """Spawn ANYmal-C USD visuals and relabel Newton bodies to valid USD paths."""
    from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # noqa: PLC0415

    stage = sim_utils.get_current_stage()
    parent_path = root_path.rsplit("/", 1)[0]
    if parent_path and parent_path != "/World" and not stage.GetPrimAtPath(parent_path).IsValid():
        sim_utils.create_prim(parent_path, "Xform")

    ANYMAL_C_CFG.spawn.func(
        root_path,
        ANYMAL_C_CFG.spawn,
        translation=translation,
        orientation=(0.0, 0.0, 0.70710678, 0.70710678),
    )
    stage = sim_utils.get_current_stage()

    for body_id in range(body_start, body_end):
        body_label = builder.body_label[body_id]
        body_name = body_label.rsplit("/", 1)[-1]
        builder.body_label[body_id] = resolve_kit_body_prim_path(stage, root_path, body_name)


def configure_anymal_sand_proto(builder: newton.ModelBuilder) -> tuple[str, str]:
    """Populate a single-environment ANYmal/sand prototype builder."""
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=0.06,
        limit_ke=1.0e3,
        limit_kd=1.0e1,
    )
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    asset_path = newton.utils.download_asset("anybotics_anymal_c")
    urdf_path = str(asset_path / "urdf" / "anymal.urdf")
    builder.add_urdf(
        urdf_path,
        xform=wp.transform(
            wp.vec3(0.0, 0.0, 0.62),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), wp.pi * 0.5),
        ),
        floating=True,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        ignore_inertial_definitions=False,
    )

    # Only the shank collision shapes interact with particles, matching Newton's
    # reference ANYmal-MPM setup.
    for body_id, body_label in enumerate(builder.body_label):
        if "SHANK" not in body_label:
            for shape_id in builder.body_shapes[body_id]:
                builder.shape_flags[shape_id] &= ~newton.ShapeFlags.COLLIDE_PARTICLES

    builder.add_ground_plane()

    initial_q = {
        "RH_HAA": 0.0,
        "RH_HFE": -0.4,
        "RH_KFE": 0.8,
        "LH_HAA": 0.0,
        "LH_HFE": -0.4,
        "LH_KFE": 0.8,
        "RF_HAA": 0.0,
        "RF_HFE": 0.4,
        "RF_KFE": -0.8,
        "LF_HAA": 0.0,
        "LF_HFE": 0.4,
        "LF_KFE": -0.8,
    }
    for name, value in initial_q.items():
        joint_index = next(i for i, label in enumerate(builder.joint_label) if label.endswith(f"/{name}"))
        q_id = builder.joint_q_start[joint_index]
        qd_id = builder.joint_qd_start[joint_index]
        builder.joint_q[q_id] = value
        builder.joint_target_pos[qd_id] = value

    for joint_dof_index in range(builder.joint_dof_count):
        builder.joint_target_ke[joint_dof_index] = 150.0
        builder.joint_target_kd[joint_dof_index] = 5.0

    spawn_sand(builder, args_cli.voxel_size, args_cli.particles_per_cell)
    return str(asset_path / "rl_policies" / "anymal_walking_policy_physx.pt"), urdf_path


def relabel_newton_bodies(builder: newton.ModelBuilder, env_id: int, body_start: int, body_end: int) -> None:
    """Give replicated bodies unique labels for non-Kit visualizers/debugging."""
    for body_id in range(body_start, body_end):
        body_name = builder.body_label[body_id].rsplit("/", 1)[-1]
        builder.body_label[body_id] = f"env_{env_id}/{body_name}"


def build_anymal_sand_model(use_kit_visuals: bool) -> tuple[str, str, DemoModelMetadata]:
    """Populate ``NewtonManager`` with replicated ANYmal/sand worlds."""
    proto = NewtonManager.create_builder()
    SolverMuJoCo.register_custom_attributes(proto)
    SolverImplicitMPM.register_custom_attributes(proto)
    default_policy_path, urdf_path = configure_anymal_sand_proto(proto)

    builder = NewtonManager.create_builder()
    SolverMuJoCo.register_custom_attributes(builder)
    SolverImplicitMPM.register_custom_attributes(builder)

    env_origins = compute_env_origins(args_cli.num_envs, args_cli.env_spacing)
    joint_q_ids: list[list[int]] = []
    joint_qd_ids: list[list[int]] = []
    particle_ids: list[list[int]] = []

    for env_id, origin in enumerate(env_origins):
        builder.begin_world(label=f"env_{env_id}")
        body_offset = builder.body_count
        particle_offset = builder.particle_count
        joint_q_offset = builder.joint_coord_count
        joint_qd_offset = builder.joint_dof_count

        builder.add_builder(proto, xform=wp.transform(wp.vec3(*origin), wp.quat_identity()))

        body_end = body_offset + proto.body_count
        if use_kit_visuals:
            root_path = "/World/Robot" if args_cli.num_envs == 1 else f"/World/Env_{env_id}/Robot"
            create_kit_body_prims(
                builder,
                root_path,
                translation=(float(origin[0]), float(origin[1]), 0.62),
                body_start=body_offset,
                body_end=body_end,
            )
        else:
            relabel_newton_bodies(builder, env_id, body_offset, body_end)

        joint_q_ids.append(list(range(joint_q_offset, joint_q_offset + proto.joint_coord_count)))
        joint_qd_ids.append(list(range(joint_qd_offset, joint_qd_offset + proto.joint_dof_count)))
        particle_ids.append(list(range(particle_offset, particle_offset + proto.particle_count)))
        builder.end_world()

    NewtonManager._num_envs = args_cli.num_envs
    NewtonManager.set_builder(builder)
    metadata = DemoModelMetadata(
        joint_q_ids=joint_q_ids,
        joint_qd_ids=joint_qd_ids,
        particle_ids=particle_ids,
        env_origins=[tuple(float(v) for v in origin) for origin in env_origins],
    )
    return default_policy_path, urdf_path, metadata


def create_mpm_solver(model: newton.Model, state: newton.State) -> SolverImplicitMPM:
    """Create and configure the one-way coupled sand solver."""
    mpm_cfg = SolverImplicitMPM.Config()
    mpm_cfg.voxel_size = args_cli.voxel_size
    mpm_cfg.tolerance = args_cli.tolerance
    mpm_cfg.transfer_scheme = "pic"
    mpm_cfg.grid_type = args_cli.grid_type
    mpm_cfg.grid_padding = 50 if args_cli.grid_type == "fixed" else 0
    mpm_cfg.max_active_cell_count = 1 << 15 if args_cli.grid_type == "fixed" else -1
    mpm_cfg.strain_basis = "P0"
    mpm_cfg.max_iterations = args_cli.mpm_iterations
    mpm_cfg.critical_fraction = 0.0
    mpm_cfg.air_drag = 1.0
    mpm_cfg.collider_velocity_mode = "backward"
    # The installed Newton 1.2.0.dev0 package expects the long solver name.
    mpm_cfg.solver = "gauss-seidel"

    mpm_solver = SolverImplicitMPM(model, mpm_cfg)
    mpm_solver.setup_collider(body_mass=wp.zeros_like(model.body_mass), body_q=state.body_q)
    return mpm_solver


class KitRobotVisuals:
    """Fallback USD transform sync for the spawned ANYmal-C Kit visual asset."""

    def __init__(self, model: newton.Model, urdf_path: str):
        from pxr import UsdGeom  # noqa: PLC0415

        stage = sim_utils.get_current_stage()
        self._body_indices: list[int] = []
        self._offset_pos: list[np.ndarray] = []
        self._offset_quat: list[np.ndarray] = []
        self._xform_ops = []

        body_names = [label.rsplit("/", 1)[-1] for label in model.body_label]
        fixed_offsets = parse_fixed_link_offsets(urdf_path, set(body_names))

        visual_targets: list[tuple[int, str, np.ndarray, np.ndarray]] = []
        for body_id, body_path in enumerate(model.body_label):
            body_name = body_names[body_id]
            identity_pos = np.zeros(3, dtype=np.float64)
            identity_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            visual_targets.append((body_id, body_path, identity_pos, identity_quat))
            for link_name, offset_pos, offset_quat in fixed_offsets.get(body_name, []):
                link_path = resolve_kit_body_prim_path(stage, infer_robot_root_path(body_path), link_name)
                visual_targets.append((body_id, link_path, offset_pos, offset_quat))

        seen_paths = set()
        for body_id, prim_path, offset_pos, offset_quat in visual_targets:
            if prim_path in seen_paths:
                continue
            seen_paths.add(prim_path)
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                xformable = UsdGeom.Xformable(prim)
                xformable.ClearXformOpOrder()
                xformable.SetResetXformStack(True)
                self._xform_ops.append(xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "newton_world"))
                self._body_indices.append(body_id)
                self._offset_pos.append(offset_pos)
                self._offset_quat.append(offset_quat)

    def update(self, state: newton.State) -> None:
        """Write Newton body poses into the corresponding USD body Xforms."""
        from pxr import Gf, Sdf  # noqa: PLC0415

        body_q = state.body_q.numpy()
        with Sdf.ChangeBlock():
            for body_id, offset_pos, offset_quat, xform_op in zip(
                self._body_indices, self._offset_pos, self._offset_quat, self._xform_ops
            ):
                transform = body_q[body_id]
                body_pos = transform[:3].astype(np.float64, copy=False)
                body_quat = transform[3:7].astype(np.float64, copy=False)
                visual_pos, visual_quat = compose_transform(body_pos, body_quat, offset_pos, offset_quat)
                matrix = Gf.Matrix4d(1.0)
                matrix.SetRotate(
                    Gf.Quatd(
                        float(visual_quat[3]),
                        Gf.Vec3d(float(visual_quat[0]), float(visual_quat[1]), float(visual_quat[2])),
                    )
                )
                matrix.SetTranslateOnly(Gf.Vec3d(float(visual_pos[0]), float(visual_pos[1]), float(visual_pos[2])))
                xform_op.Set(matrix)


def infer_robot_root_path(body_path: str) -> str:
    """Infer the USD robot root from a body prim path."""
    marker = "/Robot/"
    if marker in body_path:
        return body_path.split(marker, 1)[0] + "/Robot"
    if body_path.endswith("/Robot"):
        return body_path
    return "/World/Robot"


class KitSandPoints:
    """Small USD ``Points`` helper for visualizing MPM particles in Kit."""

    def __init__(self, prim_path: str, widths: np.ndarray):
        from pxr import Gf, UsdGeom, Vt  # noqa: PLC0415

        stage = sim_utils.get_current_stage()
        self._points = UsdGeom.Points.Define(stage, prim_path)
        self._widths_np = widths.astype(np.float32, copy=False)
        self._color = Gf.Vec3f(0.72, 0.60, 0.38)
        self._points_attr = self._points.GetPointsAttr()
        self._widths_attr = self._points.CreateWidthsAttr(Vt.FloatArray())
        self._color_attr = self._points.CreateDisplayColorAttr(Vt.Vec3fArray())
        self._particle_count = -1
        self._widths = Vt.FloatArray()
        self._colors = Vt.Vec3fArray()

    def update(self, positions: torch.Tensor) -> None:
        from pxr import Sdf, Vt  # noqa: PLC0415

        positions_np = positions.detach().cpu().numpy().astype(np.float32, copy=False)
        particle_count = int(positions_np.shape[0])
        with Sdf.ChangeBlock():
            self._points_attr.Set(Vt.Vec3fArray.FromNumpy(positions_np))
            if particle_count != self._particle_count:
                # RTX treats widths/colors as vertex data, so cache them until the particle count changes.
                self._particle_count = particle_count
                self._widths = Vt.FloatArray(self._widths_np[:particle_count].tolist())
                self._colors = Vt.Vec3fArray([self._color] * particle_count)
                self._widths_attr.Set(self._widths)
                self._color_attr.Set(self._colors)


def create_kit_robot_visuals(
    sim: sim_utils.SimulationContext, model: newton.Model, urdf_path: str
) -> KitRobotVisuals | None:
    """Create a Kit-side fallback synchronizer for the spawned ANYmal-C visual asset."""
    if "kit" not in sim.resolve_visualizer_types():
        return None
    return KitRobotVisuals(model, urdf_path)


def create_sand_points(sim: sim_utils.SimulationContext, model: newton.Model) -> KitSandPoints | None:
    """Create Kit points for the MPM particles when Kit visualization is active."""
    if "kit" not in sim.resolve_visualizer_types():
        return None
    if args_cli.kit_sand_stride < 1:
        raise ValueError("--kit-sand-stride must be >= 1.")
    particle_radius = wp.to_torch(model.particle_radius)
    if args_cli.kit_sand_stride == 1:
        rendered_radius = particle_radius
    else:
        rendered_radius = particle_radius[:: args_cli.kit_sand_stride]
    widths = 2.0 * rendered_radius.detach().cpu().numpy().astype(np.float32, copy=False)
    sim_utils.create_prim("/World/Visuals", "Xform")
    return KitSandPoints("/World/Visuals/SandParticles", widths=widths)


def update_sand_points(points: KitSandPoints | None, state: newton.State) -> None:
    """Push particle positions into the Kit USD points cloud."""
    if points is None:
        return
    particle_q = wp.to_torch(state.particle_q)
    if args_cli.kit_sand_stride == 1:
        points.update(particle_q)
    else:
        points.update(particle_q[:: args_cli.kit_sand_stride])


def update_kit_robot_visuals(robot_visuals: KitRobotVisuals | None, state: newton.State) -> None:
    """Push Newton robot body poses into Kit USD prims."""
    if robot_visuals is not None:
        robot_visuals.update(state)


def enable_newton_particle_visualization(sim: sim_utils.SimulationContext) -> None:
    """Turn on native particle rendering for active Newton-family visualizers."""
    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is not None and hasattr(viewer, "show_particles"):
            viewer.show_particles = True


def keep_running(sim: sim_utils.SimulationContext, count: int) -> bool:
    """Return whether the demo loop should continue."""
    if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
        return False
    if not sim.visualizers:
        return True
    return any(not viz.is_closed and viz.is_running() for viz in sim.visualizers)


def load_policy(default_policy_path: str, device: torch.device) -> torch.jit.ScriptModule:
    """Load the walking policy used by Newton's ANYmal example."""
    policy_path = args_cli.policy or default_policy_path
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`torch\.jit\.load` is deprecated\. Please switch to `torch\.export`\.",
            category=DeprecationWarning,
        )
        return torch.jit.load(policy_path, map_location=device)


def apply_policy(
    policy: torch.jit.ScriptModule,
    control: newton.Control,
    state: newton.State,
    joint_pos_initial: torch.Tensor,
    action: torch.Tensor,
    joint_q_ids: torch.Tensor,
    joint_qd_ids: torch.Tensor,
    lab_to_mujoco_indices: torch.Tensor,
    mujoco_to_lab_indices: torch.Tensor,
    gravity_vec: torch.Tensor,
    command: torch.Tensor,
) -> torch.Tensor:
    """Run the walking policy and write joint targets into Newton control."""
    obs = compute_obs(
        action,
        state,
        joint_pos_initial,
        joint_q_ids,
        joint_qd_ids,
        lab_to_mujoco_indices,
        gravity_vec,
        command,
    )
    with torch.no_grad():
        action = policy(obs)
        rearranged_action = torch.gather(action, 1, mujoco_to_lab_indices.unsqueeze(0).expand(action.shape[0], -1))
        target = joint_pos_initial + 0.5 * rearranged_action
        target_with_free_joint = torch.cat(
            [torch.zeros((target.shape[0], 6), device=target.device, dtype=torch.float32), target],
            dim=1,
        )
        wp.to_torch(control.joint_target_pos)[joint_qd_ids] = target_with_free_joint
    return action


def run_simulator(
    sim: sim_utils.SimulationContext,
    mpm_solver: SolverImplicitMPM,
    policy: torch.jit.ScriptModule,
    robot_visuals: KitRobotVisuals | None,
    sand_points: KitSandPoints | None,
    metadata: DemoModelMetadata,
) -> None:
    """Run the coupled robot/sand simulation loop."""
    model = NewtonManager.get_model()
    state = NewtonManager.get_state_0()
    control = NewtonManager.get_control()
    torch_device = wp.device_to_torch(model.device)

    joint_q_ids = torch.tensor(metadata.joint_q_ids, device=torch_device, dtype=torch.long)
    joint_qd_ids = torch.tensor(metadata.joint_qd_ids, device=torch_device, dtype=torch.long)
    joint_pos_initial = wp.to_torch(state.joint_q)[joint_q_ids][:, 7:].detach().clone()
    action = torch.zeros(args_cli.num_envs, 12, device=torch_device, dtype=torch.float32)
    lab_to_mujoco_indices = torch.tensor(LAB_TO_MUJOCO, device=torch_device)
    mujoco_to_lab_indices = torch.tensor(MUJOCO_TO_LAB, device=torch_device)
    gravity_vec = torch.tensor([[0.0, 0.0, -1.0]], device=torch_device, dtype=torch.float32).expand(
        args_cli.num_envs, -1
    )
    command = torch.tensor(
        [[args_cli.command_forward, args_cli.command_lateral, args_cli.command_yaw]],
        device=torch_device,
        dtype=torch.float32,
    ).expand(args_cli.num_envs, -1)

    count = 0
    frame_dt = 1.0 / args_cli.fps
    while keep_running(sim, count):
        state = NewtonManager.get_state_0()
        action = apply_policy(
            policy,
            control,
            state,
            joint_pos_initial,
            action,
            joint_q_ids,
            joint_qd_ids,
            lab_to_mujoco_indices,
            mujoco_to_lab_indices,
            gravity_vec,
            command,
        )

        sim.step(render=False)
        state = NewtonManager.get_state_0()
        mpm_solver.step(state, state, control=None, contacts=None, dt=frame_dt)
        if sim.is_rendering:
            update_kit_robot_visuals(robot_visuals, state)
            update_sand_points(sand_points, state)
            sim.render()
        count += 1


def create_launcher_sim_cfg():
    """Create the minimal config used to decide whether Kit is required."""
    from isaaclab_newton.physics import NewtonCfg as NewtonCfgClass

    import isaaclab.sim as sim_utils_module

    device = str(args_cli.device)
    if not device.startswith("cuda"):
        raise RuntimeError("Newton implicit MPM ANYmal demo requires a CUDA device.")
    frame_dt = 1.0 / args_cli.fps
    return sim_utils_module.SimulationCfg(
        dt=frame_dt,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfgClass(num_substeps=args_cli.robot_substeps, use_cuda_graph=not args_cli.disable_cuda_graph),
    )


def main() -> None:
    """Set up and run the Isaac Lab ANYmal-on-sand demo."""
    sim_cfg = create_launcher_sim_cfg()

    with launch_simulation(SimpleNamespace(sim=sim_cfg), args_cli):
        import_runtime_dependencies()
        sim_cfg.physics = NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                njmax=50,
                nconmax=100,
                ls_iterations=50,
                use_mujoco_contacts=True,
            ),
            num_substeps=args_cli.robot_substeps,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        try:
            setup_kit_scene(sim)
            sim.set_camera_view(eye=[4.5, -4.5, 2.2], target=[0.0, 0.8, 0.4])

            use_kit_visuals = "kit" in sim.resolve_visualizer_types()
            default_policy_path, urdf_path, metadata = build_anymal_sand_model(use_kit_visuals)
            sim.reset()
            enable_newton_particle_visualization(sim)

            model = NewtonManager.get_model()
            state = NewtonManager.get_state_0()
            mpm_solver = create_mpm_solver(model, state)
            robot_visuals = create_kit_robot_visuals(sim, model, urdf_path)
            update_kit_robot_visuals(robot_visuals, state)
            sand_points = create_sand_points(sim, model)
            update_sand_points(sand_points, state)
            policy = load_policy(default_policy_path, wp.device_to_torch(model.device))

            print("[INFO]: Isaac Lab Newton ANYmal MPM sand demo ready.")
            print(f"[INFO]: Running {args_cli.num_envs} environment(s) with {args_cli.env_spacing:g} m spacing.")
            print("[INFO]: Use --command-forward/--command-lateral/--command-yaw to change the fixed velocity command.")
            run_simulator(sim, mpm_solver, policy, robot_visuals, sand_points, metadata)
        finally:
            sim.clear_instance()


if __name__ == "__main__":
    main()
