# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Single rigid box two-way coupled with MPM sand through Newton proxy coupling.

This demo is intentionally small: one MuJoCo-Warp rigid box, one implicit-MPM
sand bed, and one lagged proxy mapping from the rigid solver into the MPM solver.
The Newton OpenGL viewer can right-mouse drag the box, and the live "Plots"
window shows the sand reaction wrench harvested by the proxy coupler.

.. code-block:: bash

    ./isaaclab.sh -p scripts/demos/newton_box_mpm_twoway.py --viz newton
"""

import argparse
from types import SimpleNamespace

from isaaclab_tasks.utils.sim_launcher import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Newton proxy-coupled box interacting with MPM sand.")
parser.add_argument("--fps", type=float, default=60.0, help="Simulation/control frames per second.")
parser.add_argument("--max-steps", type=int, default=900, help="Stop after this many frames; negative runs forever.")
parser.add_argument("--voxel-size", type=float, default=0.05, help="MPM grid voxel size in meters.")
parser.add_argument("--particles-per-cell", type=float, default=3.0, help="Sand particles per grid cell.")
parser.add_argument("--mpm-iterations", type=int, default=50, help="Maximum MPM rheology iterations.")
parser.add_argument("--proxy-iterations", type=int, default=1, help="Proxy relaxation passes per coupled step.")
parser.add_argument("--proxy-mass-relaxation", type=float, default=1.0, help="Scale proxy box mass inside MPM.")
parser.add_argument("--rigid-substeps", type=int, default=4, help="MJWarp substeps inside each coupled step.")
parser.add_argument("--box-mass", type=float, default=150.0, help="Rigid box mass in kg.")
parser.add_argument("--box-size", type=float, default=0.5, help="Box side length in meters.")
parser.add_argument("--box-height", type=float, default=2.0, help="Initial box center height in meters.")
parser.add_argument("--sand-friction", type=float, default=0.35, help="MPM Drucker-Prager friction coefficient.")
parser.add_argument("--sand-damping", type=float, default=0.30, help="MPM elastic damping relaxation time in seconds.")
parser.add_argument("--sand-young-modulus", type=float, default=1.0e15, help="MPM Young's modulus in Pa.")
parser.add_argument("--sand-yield-pressure", type=float, default=1.0e15, help="MPM compressive yield pressure in Pa.")
parser.add_argument("--sand-tensile-yield-ratio", type=float, default=0.0, help="MPM tensile yield ratio.")
parser.add_argument(
    "--collider-margin",
    type=float,
    default=0.0,
    help="MPM collider thickness/margin in meters; Newton's reference example uses 0.0.",
)
parser.add_argument("--log-interval", type=int, default=60, help="Print simulation progress every N steps; 0 disables.")
parser.add_argument("--disable-cuda-graph", action="store_true", help="Disable Newton CUDA graph capture.")
parser.add_argument(
    "--kit-sand-stride",
    type=int,
    default=1,
    help="Render every Nth sand particle in Kit; 1 renders every particle.",
)
add_launcher_args(parser)
args_cli = parser.parse_args()

np = torch = wp = newton = sim_utils = None
SolverImplicitMPM = None
CoupledProxyCfg = CoupledSolverCfg = CoupledSolverEntryCfg = None
MJWarpSolverCfg = MPMSolverCfg = NewtonCfg = NewtonCoupledManager = NewtonManager = ProxyCouplingCfg = None


BOX_BODY_PATH = "/World/ProxyCoupledBox"
RIGID_ENTRY = "rigid"
SAND_ENTRY = "sand"


def import_runtime_dependencies() -> None:
    """Import Newton/Isaac Lab modules after Kit has launched when requested."""
    global np, torch, wp, newton, sim_utils, SolverImplicitMPM
    global CoupledProxyCfg, CoupledSolverCfg, CoupledSolverEntryCfg
    global MJWarpSolverCfg, MPMSolverCfg, NewtonCfg, NewtonCoupledManager, NewtonManager, ProxyCouplingCfg

    import newton as newton_module
    import numpy as np_module
    import torch as torch_module
    import warp as wp_module
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
        MPMSolverCfg as MPMSolverCfgClass,
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
    from newton.solvers import SolverImplicitMPM as SolverImplicitMPMClass

    import isaaclab.sim as sim_utils_module

    np = np_module
    torch = torch_module
    wp = wp_module
    newton = newton_module
    sim_utils = sim_utils_module
    SolverImplicitMPM = SolverImplicitMPMClass
    CoupledProxyCfg = CoupledProxyCfgClass
    CoupledSolverCfg = CoupledSolverCfgClass
    CoupledSolverEntryCfg = CoupledSolverEntryCfgClass
    MJWarpSolverCfg = MJWarpSolverCfgClass
    MPMSolverCfg = MPMSolverCfgClass
    NewtonCfg = NewtonCfgClass
    NewtonCoupledManager = NewtonCoupledManagerClass
    NewtonManager = NewtonManagerClass
    ProxyCouplingCfg = ProxyCouplingCfgClass


def solid_box_inertia(mass: float, half_extent: float) -> wp.mat33:
    """Return the COM inertia tensor for a uniform cube."""
    side = 2.0 * half_extent
    diagonal = (1.0 / 6.0) * mass * side * side
    return wp.mat33(diagonal, 0.0, 0.0, 0.0, diagonal, 0.0, 0.0, 0.0, diagonal)


def spawn_sand(builder: newton.ModelBuilder) -> tuple[int, int]:
    """Add a compact MPM sand bed and return the particle index range."""
    density = 2500.0
    sand_lo = np.array([-1.0, -1.0, 0.0])
    sand_hi = np.array([1.0, 1.0, 0.5])
    resolution = np.maximum(np.ceil(args_cli.particles_per_cell * (sand_hi - sand_lo) / args_cli.voxel_size), 1).astype(
        int
    )
    cell_size = (sand_hi - sand_lo) / resolution
    radius = float(np.max(cell_size) * 0.5)
    mass = float(np.prod(cell_size) * density)

    particle_start = builder.particle_count
    builder.add_particle_grid(
        pos=wp.vec3(sand_lo),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=int(resolution[0]) + 1,
        dim_y=int(resolution[1]) + 1,
        dim_z=int(resolution[2]) + 1,
        cell_x=float(cell_size[0]),
        cell_y=float(cell_size[1]),
        cell_z=float(cell_size[2]),
        mass=mass,
        jitter=2.0 * radius,
        radius_mean=radius,
        custom_attributes={
            "mpm:friction": args_cli.sand_friction,
            "mpm:damping": args_cli.sand_damping,
            "mpm:young_modulus": args_cli.sand_young_modulus,
            "mpm:yield_pressure": args_cli.sand_yield_pressure,
            "mpm:tensile_yield_ratio": args_cli.sand_tensile_yield_ratio,
        },
    )
    return particle_start, builder.particle_count


def build_box_sand_model() -> tuple[newton.ModelBuilder, CoupledSolverCfg, int]:
    """Build a shared Newton model and the Isaac Lab proxy-coupled solver config."""
    builder = NewtonManager.create_builder()
    SolverImplicitMPM.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = 0.5

    half_extent = 0.5 * args_cli.box_size
    collider_margin = args_cli.collider_margin
    box_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, args_cli.box_height), wp.quat_identity()),
        mass=args_cli.box_mass,
        inertia=solid_box_inertia(args_cli.box_mass, half_extent),
        lock_inertia=True,
        label=BOX_BODY_PATH,
    )
    box_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5, margin=collider_margin)
    builder.add_shape_box(
        box_body,
        hx=half_extent,
        hy=half_extent,
        hz=half_extent,
        cfg=box_cfg,
        color=(0.12, 0.34, 0.85),
    )
    builder.add_ground_plane(
        cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, margin=collider_margin),
        color=(0.46, 0.38, 0.24),
    )

    particle_start, particle_end = spawn_sand(builder)

    solver_cfg = CoupledSolverCfg(
        entries=[
            CoupledSolverEntryCfg(
                name=RIGID_ENTRY,
                solver_cfg=MJWarpSolverCfg(
                    # Match Newton's mujoco_mpm_coupled_solver: Newton collision detection feeds MuJoCo contacts.
                    use_mujoco_contacts=False,
                    njmax=100,
                ),
                body_label_patterns=[BOX_BODY_PATH],
                include_static_shapes=True,
                substeps=args_cli.rigid_substeps,
            ),
            CoupledSolverEntryCfg(
                name=SAND_ENTRY,
                solver_cfg=MPMSolverCfg(
                    voxel_size=args_cli.voxel_size,
                    grid_type="fixed",
                    grid_padding=48,
                    max_active_cell_count=1 << 15,
                    strain_basis="P0",
                    transfer_scheme="apic",
                    max_iterations=args_cli.mpm_iterations,
                    critical_fraction=0.0,
                    collider_velocity_mode="forward",
                ),
                particle_range=(particle_start, particle_end),
                in_place=True,
            ),
        ],
        # Match Newton's reference example: call model.collide() externally, then pass contacts to MuJoCo.
        use_collision_pipeline=True,
        proxy_coupling=ProxyCouplingCfg(
            proxies=[
                CoupledProxyCfg(
                    source=RIGID_ENTRY,
                    destination=SAND_ENTRY,
                    body_label_patterns=[BOX_BODY_PATH],
                    mass_scale=args_cli.proxy_mass_relaxation,
                    mode="lagged",
                )
            ],
            iterations=args_cli.proxy_iterations,
        ),
    )
    return builder, solver_cfg, box_body


def setup_kit_scene(sim: sim_utils.SimulationContext) -> None:
    """Create USD prims required for Kit/Fabric visualization."""
    if "kit" not in sim.resolve_visualizer_types():
        return

    stage = sim_utils.get_current_stage()
    if not stage.GetPrimAtPath(BOX_BODY_PATH).IsValid():
        box_cfg = sim_utils.CuboidCfg(
            size=(args_cli.box_size, args_cli.box_size, args_cli.box_size),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.34, 0.85)),
        )
        box_cfg.func(
            BOX_BODY_PATH,
            box_cfg,
            translation=(0.0, 0.0, args_cli.box_height),
            orientation=(0.0, 0.0, 0.0, 1.0),
        )

    if not stage.GetPrimAtPath("/World/Ground").IsValid():
        ground_cfg = sim_utils.GroundPlaneCfg(size=(6.0, 6.0), color=(0.46, 0.38, 0.24))
        ground_cfg.func("/World/Ground", ground_cfg)

    if not stage.GetPrimAtPath("/World/DomeLight").IsValid():
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/DomeLight", light_cfg)


class KitBoxVisual:
    """Fallback USD transform sync for the Kit box visual."""

    def __init__(self, prim_path: str, body_id: int):
        from pxr import UsdGeom  # noqa: PLC0415

        stage = sim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Kit box visual prim does not exist: {prim_path}")
        xformable = UsdGeom.Xformable(prim)
        xformable.ClearXformOpOrder()
        xformable.SetResetXformStack(True)
        self._xform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "newton_world")
        self._body_id = body_id

    def update(self, state: newton.State) -> None:
        """Write the Newton box pose into the Kit USD box Xform."""
        from pxr import Gf, Sdf  # noqa: PLC0415

        transform = state.body_q.numpy()[self._body_id]
        pos = transform[:3]
        quat = transform[3:7]
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(Gf.Quatd(float(quat[3]), Gf.Vec3d(float(quat[0]), float(quat[1]), float(quat[2]))))
        matrix.SetTranslateOnly(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
        with Sdf.ChangeBlock():
            self._xform_op.Set(matrix)


def create_kit_box_visual(sim: sim_utils.SimulationContext, box_body: int) -> KitBoxVisual | None:
    """Create a Kit-side box transform synchronizer when Kit visualization is active."""
    if "kit" not in sim.resolve_visualizer_types():
        return None
    return KitBoxVisual(BOX_BODY_PATH, box_body)


def update_kit_box_visual(box_visual: KitBoxVisual | None, state: newton.State) -> None:
    """Push the box pose into the Kit USD visual."""
    if box_visual is not None:
        box_visual.update(state)


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
                self._particle_count = particle_count
                self._widths = Vt.FloatArray(self._widths_np[:particle_count].tolist())
                self._colors = Vt.Vec3fArray([self._color] * particle_count)
                self._widths_attr.Set(self._widths)
                self._color_attr.Set(self._colors)


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


def configure_newton_viewer(sim: sim_utils.SimulationContext) -> None:
    """Enable particle rendering and picking in active Newton visualizers."""
    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is None:
            continue
        if hasattr(viewer, "show_particles"):
            viewer.show_particles = True
        if hasattr(viewer, "show_contacts"):
            viewer.show_contacts = True
        if hasattr(viewer, "picking_enabled"):
            viewer.picking_enabled = True


def apply_viewer_forces(sim: sim_utils.SimulationContext, state: newton.State) -> None:
    """Apply Newton viewer picking/wind forces before the physics step."""
    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is not None and hasattr(viewer, "apply_forces"):
            viewer.apply_forces(state)


def keep_running(sim: sim_utils.SimulationContext, count: int) -> bool:
    """Return whether the demo loop should continue."""
    if args_cli.max_steps >= 0 and count >= args_cli.max_steps:
        return False
    if not sim.visualizers:
        return True
    return any(not viz.is_closed and viz.is_running() for viz in sim.visualizers)


def read_proxy_wrench(box_body: int, dt: float) -> torch.Tensor:
    """Return the latest sand reaction wrench harvested by the proxy coupler."""
    wrenches = NewtonCoupledManager.get_proxy_body_wrenches(RIGID_ENTRY, SAND_ENTRY)
    if wrenches is not None:
        return wp.to_torch(wrenches)[box_body].detach().clone()

    try:
        mpm_solver = NewtonCoupledManager.get_entry_solver(SAND_ENTRY)
    except (KeyError, RuntimeError):
        mpm_solver = None
    if (
        mpm_solver is not None
        and hasattr(mpm_solver, "collect_collider_impulses")
        and hasattr(mpm_solver, "collider_body_index")
    ):
        state = NewtonManager.get_state_0()
        impulses, positions, collider_ids = mpm_solver.collect_collider_impulses(state)
        collider_ids_t = wp.to_torch(collider_ids).long()
        if collider_ids_t.numel() > 0:
            collider_body_index = wp.to_torch(mpm_solver.collider_body_index).long()
            valid = (collider_ids_t >= 0) & (collider_ids_t < collider_body_index.shape[0])
            body_ids = torch.full_like(collider_ids_t, -1)
            body_ids[valid] = collider_body_index[collider_ids_t[valid]]
            body_mask = body_ids == int(box_body)
            if bool(body_mask.any()):
                force_samples = wp.to_torch(impulses)[body_mask] / float(dt)
                force = force_samples.sum(dim=0)
                points = wp.to_torch(positions)[body_mask]
                body_center = wp.to_torch(state.body_q)[box_body, 0:3]
                torque = torch.cross(points - body_center, force_samples, dim=1).sum(dim=0)
                return torch.cat((force, torque)).detach().clone()
    return torch.zeros(6, device=wp.to_torch(NewtonManager.get_state_0().joint_q).device)


def log_wrench_plots(sim: sim_utils.SimulationContext, wrench: torch.Tensor) -> None:
    """Push force and torque components into the Newton viewer plot window."""
    force = wrench[0:3]
    torque = wrench[3:6]
    force_mag = float(torch.linalg.norm(force).item())
    torque_mag = float(torch.linalg.norm(torque).item())

    for visualizer in sim.visualizers:
        viewer = getattr(visualizer, "_viewer", None)
        if viewer is None or not hasattr(viewer, "log_scalar"):
            continue
        viewer.log_scalar("Sand Reaction |F| [N]", force_mag, smoothing=4)
        viewer.log_scalar("Sand Reaction Fx [N]", float(force[0].item()), smoothing=4)
        viewer.log_scalar("Sand Reaction Fy [N]", float(force[1].item()), smoothing=4)
        viewer.log_scalar("Sand Reaction Fz [N]", float(force[2].item()), smoothing=4)
        viewer.log_scalar("Sand Reaction |tau| [Nm]", torque_mag, smoothing=4)
        viewer.log_scalar("Sand Reaction tau_x [Nm]", float(torque[0].item()), smoothing=4)
        viewer.log_scalar("Sand Reaction tau_y [Nm]", float(torque[1].item()), smoothing=4)
        viewer.log_scalar("Sand Reaction tau_z [Nm]", float(torque[2].item()), smoothing=4)


def log_progress(count: int, state: newton.State, wrench: torch.Tensor) -> None:
    """Print a compact heartbeat showing motion and coupling forces."""
    if args_cli.log_interval <= 0 or count % args_cli.log_interval != 0:
        return
    body_q = wp.to_torch(state.body_q)
    particle_q = wp.to_torch(state.particle_q)
    box_pos = body_q[0, 0:3].detach().cpu().numpy()
    sand_min = particle_q.min(dim=0).values.detach().cpu().numpy()
    sand_max = particle_q.max(dim=0).values.detach().cpu().numpy()
    force_mag = float(torch.linalg.norm(wrench[0:3]).item())
    torque_mag = float(torch.linalg.norm(wrench[3:6]).item())
    print(
        "[INFO]: step "
        f"{count:06d} t={count / args_cli.fps:.2f}s "
        f"box=({box_pos[0]:.3f}, {box_pos[1]:.3f}, {box_pos[2]:.3f}) "
        f"|F_sand|={force_mag:.2f}N |tau_sand|={torque_mag:.2f}Nm "
        f"sand_z=[{sand_min[2]:.3f}, {sand_max[2]:.3f}]",
        flush=True,
    )


def run_simulator(
    sim: sim_utils.SimulationContext,
    box_body: int,
    box_visual: KitBoxVisual | None,
    sand_points: KitSandPoints | None,
) -> None:
    """Run the two-way coupled box/sand simulation loop."""
    count = 0
    while keep_running(sim, count):
        apply_viewer_forces(sim, NewtonManager.get_state_0())
        sim.step(render=False)

        state = NewtonManager.get_state_0()
        wrench = read_proxy_wrench(box_body, dt=1.0 / args_cli.fps)
        log_wrench_plots(sim, wrench)
        log_progress(count, state, wrench)

        if sim.is_rendering:
            update_kit_box_visual(box_visual, state)
            update_sand_points(sand_points, state)
            sim.render()
        count += 1


def create_launcher_sim_cfg():
    """Create the minimal config used to decide whether Kit is required."""
    from isaaclab_newton.physics import NewtonCfg as NewtonCfgClass

    import isaaclab.sim as sim_utils_module

    device = str(args_cli.device)
    if not device.startswith("cuda"):
        raise RuntimeError("Newton implicit MPM coupling requires a CUDA device.")
    return sim_utils_module.SimulationCfg(
        dt=1.0 / args_cli.fps,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfgClass(num_substeps=1, use_cuda_graph=not args_cli.disable_cuda_graph),
    )


def main() -> None:
    """Set up and run the Isaac Lab one-box MPM coupling demo."""
    sim_cfg = create_launcher_sim_cfg()

    with launch_simulation(SimpleNamespace(sim=sim_cfg), args_cli):
        import_runtime_dependencies()
        builder, solver_cfg, box_body = build_box_sand_model()
        sim_cfg.physics = NewtonCfg(
            solver_cfg=solver_cfg,
            num_substeps=1,
            use_cuda_graph=not args_cli.disable_cuda_graph,
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        try:
            sim.set_camera_view(eye=[2.7, -3.1, 1.9], target=[0.0, 0.0, 0.35])
            setup_kit_scene(sim)
            NewtonManager.set_builder(builder)
            sim.reset()
            configure_newton_viewer(sim)
            model = NewtonManager.get_model()
            state = NewtonManager.get_state_0()
            box_visual = create_kit_box_visual(sim, box_body)
            update_kit_box_visual(box_visual, state)
            sand_points = create_sand_points(sim, model)
            update_sand_points(sand_points, state)

            print("[INFO]: Isaac Lab Newton one-box MPM two-way coupling demo ready.")
            print(
                "[INFO]: Right-mouse drag the box in the Newton viewer; see the Plots window for sand reaction wrench."
            )
            run_simulator(sim, box_body, box_visual, sand_points)
        finally:
            sim.clear_instance()


if __name__ == "__main__":
    main()
