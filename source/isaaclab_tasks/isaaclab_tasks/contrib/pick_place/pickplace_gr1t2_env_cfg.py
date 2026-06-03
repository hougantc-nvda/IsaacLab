# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers.pink_ik import DampingTaskCfg, FrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from . import mdp

from isaaclab_assets.robots.fourier import GR1T2_HIGH_PD_CFG  # isort: skip
from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg  # isort: skip
from isaaclab_teleop.xr_cfg import XrCfg  # isort: skip

_GR1T2_KITLESS_KINEMATICS_URDF_PATH = Path(__file__).parent / "assets" / "GR1T2_fourier_hand_6dof_kinematics.urdf"
_KITLESS_NEWTON_GR1T2_IK_LM_DAMPING = 12.0
_KITLESS_NEWTON_GR1T2_IK_GAIN = 0.5
_KITLESS_NEWTON_GR1T2_IK_ORIENTATION_COST = 1.0
_KITLESS_NEWTON_TABLE_TOP_SIZE = (1.45, 0.36, 0.04)
_KITLESS_NEWTON_TABLE_TOP_CENTER = (-0.90, 0.45, 0.93)
_KITLESS_NEWTON_TARGET_BIN_SIZE = (0.46, 0.32, 0.12)
_KITLESS_NEWTON_TARGET_BIN_CENTER = (0.60, 0.45, 1.01)
_KITLESS_NEWTON_TARGET_BIN_WALL_THICKNESS = 0.025
_MAX_KITLESS_NEWTON_IK_TARGET_RAD = 6.5
_MAX_KITLESS_NEWTON_IK_STEP_DELTA_RAD = 0.25
_KITLESS_NEWTON_WRIST_TARGET_MAX_STEP_M = 0.010
_KITLESS_NEWTON_WRIST_TARGET_MAX_STEP_RAD = 0.06


def _as_torch_tensor(value) -> torch.Tensor:
    """Return a torch tensor from Isaac Lab or Newton proxy arrays."""
    return value.torch if hasattr(value, "torch") else value


def _teleop_action_target_issue(action: torch.Tensor) -> str | None:
    """Return a human-readable issue for unsafe GR1T2 teleop targets, if any."""
    flat = action.detach().flatten()
    if not torch.isfinite(flat).all():
        return "non-finite action target"
    if flat.numel() < 14:
        return None
    if flat.numel() > 14 and float(torch.max(torch.abs(flat[14:])).detach().cpu()) > 6.5:
        return "unreasonable hand joint target magnitude"

    wrists = flat[:14].reshape(2, 7)
    if not torch.isfinite(wrists).all():
        return "non-finite wrist target"

    positions = wrists[:, :3]
    quaternions = wrists[:, 3:7]
    quat_norms = torch.linalg.norm(quaternions, dim=1)
    if bool(torch.any(quat_norms < 0.25)) or bool(torch.any(quat_norms > 2.0)):
        return f"invalid wrist quaternion norms {quat_norms.detach().cpu().tolist()}"

    position_norms = torch.linalg.norm(positions, dim=1)
    left_identity = torch.linalg.norm(quaternions[0] - quaternions.new_tensor([0.0, 0.0, 0.0, 1.0])) < 1.0e-3
    right_identity = torch.linalg.norm(quaternions[1] - quaternions.new_tensor([0.0, 0.0, 0.0, 1.0])) < 1.0e-3
    right_z_flip = torch.linalg.norm(quaternions[1] - quaternions.new_tensor([0.0, 0.0, 1.0, 0.0])) < 1.0e-3
    if bool(torch.all(position_norms < 1.0e-4)) and bool(left_identity) and bool(right_identity or right_z_flip):
        return "identity wrist targets from inactive or invalid hand tracking"

    return None


def _format_teleop_wrist_targets(action: torch.Tensor) -> str:
    """Format the wrist target portion of a GR1T2 action tensor for diagnostics."""
    flat = action.detach().flatten()
    if flat.numel() < 14:
        return f"action_dim={flat.numel()}"
    wrists = flat[:14].reshape(2, 7).detach().cpu().numpy()
    left = np.array2string(wrists[0], precision=3, suppress_small=True)
    right = np.array2string(wrists[1], precision=3, suppress_small=True)
    if flat.numel() == 14:
        return f"left={left}, right={right}"
    hand_targets = flat[14:]
    hand_min = float(torch.min(hand_targets).detach().cpu())
    hand_max = float(torch.max(hand_targets).detach().cpu())
    return f"left={left}, right={right}, hand_range=({hand_min:.3f}, {hand_max:.3f})"


def _get_gr1t2_wrist_poses(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return the current GR1T2 wrist poses used by the Pink IK action term."""
    try:
        robot = env.scene["robot"]
        body_names = list(getattr(robot, "body_names", None) or getattr(robot.data, "body_names", []))
        target_links = env.cfg.actions.upper_body_ik.target_eef_link_names
        left_index = body_names.index(target_links["left_wrist"])
        right_index = body_names.index(target_links["right_wrist"])
        pose_w = _as_torch_tensor(robot.data.body_link_pose_w)[0]
    except (AttributeError, KeyError, ValueError):
        return None
    return (
        pose_w[left_index, :3].clone(),
        pose_w[left_index, 3:7].clone(),
        pose_w[right_index, :3].clone(),
        pose_w[right_index, 3:7].clone(),
    )


def _make_kitless_newton_idle_action(env, device: str | torch.device | None) -> torch.Tensor:
    """Build a hold-pose action for no-client kitless Newton debugging."""
    sim_device = device if device is not None else getattr(getattr(env.cfg, "sim", None), "device", "cpu")
    idle_action = getattr(env.cfg, "idle_action", None)
    if idle_action is not None:
        return torch.as_tensor(idle_action, dtype=torch.float32, device=sim_device).flatten()

    wrist_poses = _get_gr1t2_wrist_poses(env)
    if wrist_poses is None:
        action_dim = int(getattr(env.action_manager, "total_action_dim", 36))
        return torch.zeros(action_dim, dtype=torch.float32, device=sim_device)

    left_pos, left_quat, right_pos, right_quat = wrist_poses
    hand_joint_names = getattr(env.cfg.actions.upper_body_ik, "hand_joint_names", [])
    hand_targets = torch.zeros(len(hand_joint_names), dtype=torch.float32, device=sim_device)
    return torch.cat(
        (
            left_pos.to(device=sim_device, dtype=torch.float32),
            left_quat.to(device=sim_device, dtype=torch.float32),
            right_pos.to(device=sim_device, dtype=torch.float32),
            right_quat.to(device=sim_device, dtype=torch.float32),
            hand_targets,
        )
    )


class _KitlessNewtonIdleActionDevice:
    """Synthetic teleop device that holds the GR1T2 default pose without a client."""

    def __init__(self, env, device: str | torch.device | None) -> None:
        self._env = env
        self._device = device
        self._action: torch.Tensor | None = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def __str__(self) -> str:
        return "Kitless Newton idle-action debug device"

    def reset(self) -> None:
        self._action = None

    def advance(self) -> torch.Tensor:
        if self._action is None:
            self._action = _make_kitless_newton_idle_action(self._env, self._device)
            print(f"Kitless Newton debug idle action: {_format_teleop_wrist_targets(self._action)}")
        return self._action.clone()


def _step_vector_towards(current: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
    """Move a vector toward a target by at most ``max_step``."""
    delta = target - current
    distance = torch.linalg.norm(delta)
    if float(distance.detach().cpu()) <= max_step:
        return target.clone()
    return current + delta * (max_step / torch.clamp(distance, min=1.0e-6))


def _step_quaternion_towards(current: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
    """Move a quaternion toward a target by at most ``max_step`` radians."""
    current = torch.nn.functional.normalize(current, dim=0)
    target = torch.nn.functional.normalize(target, dim=0)
    if float(torch.sum(current * target).detach().cpu()) < 0.0:
        target = -target
    dot = torch.clamp(torch.abs(torch.sum(current * target)), 0.0, 1.0)
    angle = 2.0 * torch.acos(dot)
    if float(angle.detach().cpu()) <= max_step:
        return target.clone()
    alpha = max_step / torch.clamp(angle, min=1.0e-6)
    return torch.nn.functional.normalize((1.0 - alpha) * current + alpha * target, dim=0)


def _condition_kitless_newton_gr1t2_action(env, action: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Ramp GR1T2 wrist targets from the current robot wrists for kitless Newton."""
    flat = action.detach().flatten()
    if flat.numel() < 14:
        state.clear()
        return action

    conditioned = action.clone()
    conditioned_flat = conditioned.flatten()
    current_wrist_poses = _get_gr1t2_wrist_poses(env)
    if current_wrist_poses is None:
        state.clear()
        return conditioned

    if "left_pos" not in state:
        left_pos, left_quat, right_pos, right_quat = current_wrist_poses
        state["left_pos"] = left_pos.to(device=conditioned_flat.device, dtype=conditioned_flat.dtype)
        state["left_quat"] = left_quat.to(device=conditioned_flat.device, dtype=conditioned_flat.dtype)
        state["right_pos"] = right_pos.to(device=conditioned_flat.device, dtype=conditioned_flat.dtype)
        state["right_quat"] = right_quat.to(device=conditioned_flat.device, dtype=conditioned_flat.dtype)

    state["left_pos"] = _step_vector_towards(
        state["left_pos"],
        flat[0:3].to(device=state["left_pos"].device, dtype=state["left_pos"].dtype),
        _KITLESS_NEWTON_WRIST_TARGET_MAX_STEP_M,
    )
    state["left_quat"] = _step_quaternion_towards(
        state["left_quat"],
        flat[3:7].to(device=state["left_quat"].device, dtype=state["left_quat"].dtype),
        _KITLESS_NEWTON_WRIST_TARGET_MAX_STEP_RAD,
    )
    state["right_pos"] = _step_vector_towards(
        state["right_pos"],
        flat[7:10].to(device=state["right_pos"].device, dtype=state["right_pos"].dtype),
        _KITLESS_NEWTON_WRIST_TARGET_MAX_STEP_M,
    )
    state["right_quat"] = _step_quaternion_towards(
        state["right_quat"],
        flat[10:14].to(device=state["right_quat"].device, dtype=state["right_quat"].dtype),
        _KITLESS_NEWTON_WRIST_TARGET_MAX_STEP_RAD,
    )
    conditioned_flat[0:3] = state["left_pos"]
    conditioned_flat[3:7] = state["left_quat"]
    conditioned_flat[7:10] = state["right_pos"]
    conditioned_flat[10:14] = state["right_quat"]
    return conditioned


def _format_largest_named_values(names: list[str], values: torch.Tensor, *, limit: int = 5) -> str:
    """Format the largest absolute joint values with their names."""
    if values.numel() == 0:
        return "<none>"
    finite_values = torch.nan_to_num(values.detach().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
    count = min(limit, finite_values.numel())
    indices = torch.topk(torch.abs(finite_values), count).indices.tolist()
    entries = []
    for index in indices:
        name = names[index] if index < len(names) else f"joint_{index}"
        entries.append(f"{name}={float(finite_values[index]):.3f}")
    return ", ".join(entries)


def _format_gr1t2_pose_diagnostics(env) -> str | None:
    """Format root, head, and foot/ankle poses for GR1T2 teleop debugging."""
    try:
        robot = env.scene["robot"]
        body_names = list(getattr(robot, "body_names", None) or getattr(robot.data, "body_names", []))
        pose_w = _as_torch_tensor(robot.data.body_link_pose_w)[0].detach().cpu().numpy()
        selected = []
        for index, name in enumerate(body_names):
            lower_name = name.lower()
            is_root_body = lower_name in {"base_link", "pelvis", "pelvis_link"}
            is_limb_body = any(token in lower_name for token in ("head", "foot", "ankle", "sole"))
            if is_root_body or is_limb_body:
                pose = np.array2string(pose_w[index], precision=3, suppress_small=True)
                selected.append(f"{name}={pose}")

        joint_names = list(getattr(robot, "joint_names", None) or getattr(robot.data, "joint_names", []))
        joint_ids = [
            index
            for index, name in enumerate(joint_names)
            if any(
                token in name
                for token in ("head_", "_hip_", "_knee_", "_ankle_", "waist_", "shoulder_", "elbow_", "wrist_")
            )
        ]
        joint_summary = ""
        if joint_ids:
            joint_pos = _as_torch_tensor(robot.data.joint_pos)[0, joint_ids].detach().cpu()
            joint_vel = _as_torch_tensor(robot.data.joint_vel)[0, joint_ids].detach().cpu()
            selected_joint_names = [joint_names[index] for index in joint_ids]
            joint_pos_target = getattr(robot.data, "joint_pos_target", None)
            target_summary = ""
            if joint_pos_target is not None:
                joint_target = _as_torch_tensor(joint_pos_target)[0, joint_ids].detach().cpu()
                joint_target_error = joint_target - joint_pos
                target_summary = (
                    f"; limb_joint_target_range=({float(torch.min(joint_target)):.3f}, "
                    f"{float(torch.max(joint_target)):.3f})"
                    f"; largest_joint_target_error="
                    f"{_format_largest_named_values(selected_joint_names, joint_target_error)}"
                )
            joint_summary = (
                f"; limb_joint_pos_range=({float(torch.min(joint_pos)):.3f}, {float(torch.max(joint_pos)):.3f})"
                f"; limb_joint_vel_range=({float(torch.min(joint_vel)):.3f}, {float(torch.max(joint_vel)):.3f})"
                f"; largest_joint_pos={_format_largest_named_values(selected_joint_names, joint_pos)}"
                f"; largest_joint_vel={_format_largest_named_values(selected_joint_names, joint_vel)}"
                f"{target_summary}"
            )

        if not selected and not joint_summary:
            return None
        return "; ".join(selected) + joint_summary
    except Exception:
        return None


def _log_gr1t2_pose_diagnostics(env, label: str) -> None:
    """Print GR1T2 transform diagnostics when body state is available."""
    diagnostics = _format_gr1t2_pose_diagnostics(env)
    if diagnostics is not None:
        print(f"{label}: {diagnostics}")


def _install_kitless_newton_pink_ik_guard(env) -> None:
    """Clamp impossible Pink IK targets before they destabilize kitless Newton."""
    try:
        action_term = env.action_manager.get_term("upper_body_ik")
    except (AttributeError, KeyError):
        return
    if getattr(action_term, "_kitless_newton_guard_installed", False):
        return
    if not hasattr(action_term, "_compute_ik_solutions"):
        return

    original_compute = action_term._compute_ik_solutions
    controlled_joint_ids = list(getattr(action_term, "_isaaclab_controlled_joint_ids", []))
    controlled_joint_names = list(getattr(action_term, "_isaaclab_controlled_joint_names", []))
    if not controlled_joint_ids:
        return

    log_count = 0

    def _guarded_compute_ik_solutions():
        nonlocal log_count

        targets = original_compute()
        current = _as_torch_tensor(action_term._asset.data.joint_pos)[:, controlled_joint_ids].to(
            device=targets.device, dtype=targets.dtype
        )
        if not torch.isfinite(targets).all():
            issue = "non-finite IK target"
            safe_targets = current.detach().clone()
            delta = torch.zeros_like(current)
        else:
            delta = targets - current
            max_abs_target = float(torch.max(torch.abs(targets)).detach().cpu()) if targets.numel() else 0.0
            max_abs_delta = float(torch.max(torch.abs(delta)).detach().cpu()) if delta.numel() else 0.0
            issue = None
            if max_abs_target > _MAX_KITLESS_NEWTON_IK_TARGET_RAD:
                issue = f"target magnitude {max_abs_target:.3f} rad"
            if max_abs_delta > _MAX_KITLESS_NEWTON_IK_STEP_DELTA_RAD:
                delta_issue = f"step delta {max_abs_delta:.3f} rad"
                issue = f"{issue}, {delta_issue}" if issue is not None else delta_issue

            if issue is None:
                safe_targets = targets
            else:
                clipped_delta = torch.clamp(
                    delta,
                    -_MAX_KITLESS_NEWTON_IK_STEP_DELTA_RAD,
                    _MAX_KITLESS_NEWTON_IK_STEP_DELTA_RAD,
                )
                safe_targets = torch.clamp(
                    current + clipped_delta,
                    -_MAX_KITLESS_NEWTON_IK_TARGET_RAD,
                    _MAX_KITLESS_NEWTON_IK_TARGET_RAD,
                )

        if issue is not None and log_count < 24:
            target_range = (float(torch.min(targets).detach().cpu()), float(torch.max(targets).detach().cpu()))
            delta_range = (float(torch.min(delta).detach().cpu()), float(torch.max(delta).detach().cpu()))
            largest = _format_largest_named_values(controlled_joint_names, targets[0])
            print(
                "Clamping unsafe Pink IK targets for kitless Newton: "
                f"{issue}; target_range=({target_range[0]:.3f}, {target_range[1]:.3f}); "
                f"delta_range=({delta_range[0]:.3f}, {delta_range[1]:.3f}); largest={largest}"
            )
            log_count += 1
        elif issue is None and log_count < 4:
            target_range = (float(torch.min(targets).detach().cpu()), float(torch.max(targets).detach().cpu()))
            delta_range = (float(torch.min(delta).detach().cpu()), float(torch.max(delta).detach().cpu()))
            largest = _format_largest_named_values(controlled_joint_names, targets[0])
            print(
                "Pink IK targets for kitless Newton: "
                f"target_range=({target_range[0]:.3f}, {target_range[1]:.3f}); "
                f"delta_range=({delta_range[0]:.3f}, {delta_range[1]:.3f}); largest={largest}"
            )
            log_count += 1

        return safe_targets

    action_term._compute_ik_solutions = _guarded_compute_ik_solutions
    action_term._kitless_newton_guard_installed = True
    print("Installed kitless Newton Pink IK target guard.")


@configclass
class PickPlaceGR1T2PhysicsCfg(PresetCfg):
    """Physics backend presets for GR1T2 pick-place."""

    default: PhysxCfg = PhysxCfg()
    physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()


_KITLESS_NEWTON_GR1T2_ACTUATOR_GAINS = {
    "trunk": (500.0, 50.0, 0.10, 250.0),
    "legs": (300.0, 30.0, 0.10, 250.0),
    "head": (80.0, 8.0, 0.03, 80.0),
    "right-arm": (1200.0, 60.0, 0.30, 800.0),
    "left-arm": (1200.0, 60.0, 0.30, 800.0),
    "right-hand": (5.0, 0.5, 0.001, 20.0),
    "left-hand": (5.0, 0.5, 0.001, 20.0),
}


def _build_gr1t2_pickplace_pipeline():
    """Build an IsaacTeleop retargeting pipeline for GR1T2 pick-place teleoperation.

    Creates two Se3AbsRetargeters for left and right wrist pose tracking and
    two DexHandRetargeters for left and right dexterous hand finger control
    from hand tracking data. All outputs are flattened into a single action
    tensor via TensorReorderer.
    """
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # Create input sources (trackers are auto-discovered from pipeline)
    controllers = ControllersSource(name="controllers")
    hands = HandsSource(name="hands")

    # External input: world-to-anchor 4x4 transform matrix provided by IsaacTeleopDevice
    transform_input = ValueInput("world_T_anchor", TransformMatrix())

    # Apply the coordinate-frame transform to controller poses so that
    # downstream retargeters receive data in the simulation world frame.
    _transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # -------------------------------------------------------------------------
    # SE3 Absolute Pose Retargeters (left and right wrists)
    # -------------------------------------------------------------------------
    # Left wrist: identity rotation offset (passes through as-is in original retargeter)
    left_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {
            HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT),
        }
    )

    # Right wrist: 180-degree Z rotation offset
    # From GR1T2Retargeter._retarget_abs: the USD control frame is 180 degrees
    # rotated around the Z axis w.r.t. the OpenXR frame.
    right_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=180.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {
            HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # DexHand Retargeters (left and right hands)
    # -------------------------------------------------------------------------
    # Resolve dex-retargeting YAML config paths from IsaacLab's retargeter data directory
    import isaaclab_teleop.isaac_teleop_cfg as _teleop_cfg_mod

    _teleop_cfg_file = _teleop_cfg_mod.__file__
    if _teleop_cfg_file is None:
        raise RuntimeError("Could not resolve isaaclab_teleop package path for dex-retargeting configs.")
    _teleop_pkg_dir = os.path.dirname(_teleop_cfg_file)
    _data_dir = os.path.join(
        _teleop_pkg_dir,
        "deprecated",
        "openxr",
        "retargeters",
        "humanoid",
        "fourier",
        "data",
    )
    _config_dir = os.path.join(_data_dir, "configs", "dex-retargeting")
    left_yaml_path = os.path.join(_config_dir, "fourier_hand_left_dexpilot.yml")
    right_yaml_path = os.path.join(_config_dir, "fourier_hand_right_dexpilot.yml")

    # Resolve URDF paths (downloads from Omniverse if needed)
    local_left_urdf = retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_left_hand.urdf")
    local_right_urdf = retrieve_file_path(f"{ISAACLAB_NUCLEUS_DIR}/Mimic/GR1T2_assets/GR1_T2_right_hand.urdf")

    # Hand-tracking to base-link frame transform (OPERATOR2MANO matrix)
    # From gr1_t2_dex_retargeting_utils: [[0,-1,0],[-1,0,0],[0,0,-1]]
    operator2mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)

    # Joint names for each hand (11 DOF per hand)
    left_hand_joint_names = [
        "L_index_proximal_joint",
        "L_index_intermediate_joint",
        "L_middle_proximal_joint",
        "L_middle_intermediate_joint",
        "L_pinky_proximal_joint",
        "L_pinky_intermediate_joint",
        "L_ring_proximal_joint",
        "L_ring_intermediate_joint",
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        "L_thumb_distal_joint",
    ]

    right_hand_joint_names = [
        "R_index_proximal_joint",
        "R_index_intermediate_joint",
        "R_middle_proximal_joint",
        "R_middle_intermediate_joint",
        "R_pinky_proximal_joint",
        "R_pinky_intermediate_joint",
        "R_ring_proximal_joint",
        "R_ring_intermediate_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        "R_thumb_distal_joint",
    ]

    left_dex_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=left_yaml_path,
        hand_urdf=local_left_urdf,
        hand_joint_names=left_hand_joint_names,
        hand_side="left",
        handtracking_to_baselink_frame_transform=operator2mano,
    )
    left_dex = DexHandRetargeter(left_dex_cfg, name="left_hand")
    connected_left_dex = left_dex.connect(
        {
            HandsSource.LEFT: hands.output(HandsSource.LEFT),
        }
    )

    right_dex_cfg = DexHandRetargeterConfig(
        hand_retargeting_config=right_yaml_path,
        hand_urdf=local_right_urdf,
        hand_joint_names=right_hand_joint_names,
        hand_side="right",
        handtracking_to_baselink_frame_transform=operator2mano,
    )
    right_dex = DexHandRetargeter(right_dex_cfg, name="right_hand")
    connected_right_dex = right_dex.connect(
        {
            HandsSource.RIGHT: hands.output(HandsSource.RIGHT),
        }
    )

    # -------------------------------------------------------------------------
    # TensorReorderer: flatten into a 36D action tensor
    # -------------------------------------------------------------------------
    # Se3AbsRetargeter outputs 7D arrays: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]

    # Output order must match the PinkInverseKinematicsAction resolved hand joint order:
    #   [left_wrist(7), right_wrist(7), left_hand_joints(11), right_hand_joints(11)].
    # PinkInverseKinematicsAction preserves this configured hand order, so keep each hand retargeter output
    # in the same order.
    output_order = tuple(left_ee_elements + right_ee_elements + left_hand_joint_names + right_hand_joint_names)

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_hand_joints": left_hand_joint_names,
            "right_hand_joints": right_hand_joint_names,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_hand_joints": connected_left_dex.output("hand_joints"),
            "right_hand_joints": connected_right_dex.output("hand_joints"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline, [left_dex, right_dex]


def _configure_gr1t2_newton_usd_imports() -> None:
    """Register task-specific Newton USD import options for kitless visualization."""
    from isaaclab_newton.physics import NewtonManager

    NewtonManager.register_usd_import_options(
        "PackingTable",
        load_static_visual_shapes=True,
        load_xform_collision_shapes=True,
        floating=False,
    )
    NewtonManager.register_usd_import_options(
        "Object",
        load_collision_visual_shapes=True,
        hide_collision_shapes=True,
    )


def _install_kitless_newton_table_fallback(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Use Newton-native primitive table/bin geometry for kitless teleoperation."""
    scene_cfg = getattr(env_cfg, "scene", None)
    if scene_cfg is None or not hasattr(scene_cfg, "packing_table"):
        return

    def _cuboid_cfg(
        prim_name: str,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        color: tuple[float, float, float],
    ) -> RigidObjectCfg:
        return RigidObjectCfg(
            prim_path=f"/World/envs/env_.*/{prim_name}",
            init_state=RigidObjectCfg.InitialStateCfg(pos=center, rot=(0.0, 0.0, 0.0, 1.0)),
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.8),
            ),
        )

    table_color = (0.32, 0.34, 0.34)
    bin_color = (0.05, 0.06, 0.07)
    table_center = _KITLESS_NEWTON_TABLE_TOP_CENTER
    table_size = _KITLESS_NEWTON_TABLE_TOP_SIZE
    bin_center = _KITLESS_NEWTON_TARGET_BIN_CENTER
    bin_size = _KITLESS_NEWTON_TARGET_BIN_SIZE
    wall_thickness = _KITLESS_NEWTON_TARGET_BIN_WALL_THICKNESS

    scene_cfg.packing_table = _cuboid_cfg("PackingTable", table_center, table_size, table_color)
    scene_cfg.packing_table_target_bin_floor = _cuboid_cfg(
        "PackingTableTargetBinFloor",
        (bin_center[0], bin_center[1], table_center[2]),
        (bin_size[0], bin_size[1], table_size[2]),
        table_color,
    )
    scene_cfg.packing_table_target_bin_left = _cuboid_cfg(
        "PackingTableTargetBinLeft",
        (bin_center[0] - bin_size[0] * 0.5, bin_center[1], bin_center[2]),
        (wall_thickness, bin_size[1] + 2.0 * wall_thickness, bin_size[2]),
        bin_color,
    )
    scene_cfg.packing_table_target_bin_right = _cuboid_cfg(
        "PackingTableTargetBinRight",
        (bin_center[0] + bin_size[0] * 0.5, bin_center[1], bin_center[2]),
        (wall_thickness, bin_size[1] + 2.0 * wall_thickness, bin_size[2]),
        bin_color,
    )
    scene_cfg.packing_table_target_bin_front = _cuboid_cfg(
        "PackingTableTargetBinFront",
        (bin_center[0], bin_center[1] - bin_size[1] * 0.5, bin_center[2]),
        (bin_size[0], wall_thickness, bin_size[2]),
        bin_color,
    )
    scene_cfg.packing_table_target_bin_back = _cuboid_cfg(
        "PackingTableTargetBinBack",
        (bin_center[0], bin_center[1] + bin_size[1] * 0.5, bin_center[2]),
        (bin_size[0], wall_thickness, bin_size[2]),
        bin_color,
    )
    print("Installed kitless Newton primitive packing table and target bin.")


def _gr1t2_kitless_kinematics_candidate(env_cfg: ManagerBasedRLEnvCfg) -> tuple[Path, Path | None, bool] | None:
    """Return the configured GR1T2 Pink IK URDF asset for kitless teleoperation."""
    urdf_path_value = getattr(env_cfg, "kitless_kinematics_urdf_path", None)
    if not urdf_path_value:
        return None

    urdf_path = Path(urdf_path_value).expanduser()
    mesh_path_value = getattr(env_cfg, "kitless_kinematics_mesh_path", None)
    mesh_root = Path(mesh_path_value).expanduser() if mesh_path_value else None
    unprefixed_urdf = bool(getattr(env_cfg, "kitless_kinematics_unprefixed_urdf", True))
    return urdf_path, mesh_root, unprefixed_urdf


def _rewrite_gr1t2_kitless_task_frames(controller_cfg, *, unprefixed_urdf: bool) -> None:
    """Match GR1T2 Pink task frame names to the selected URDF asset."""
    if not unprefixed_urdf:
        return

    frame_map = {
        "GR1T2_fourier_hand_6dof_left_hand_pitch_link": "left_hand_pitch_link",
        "GR1T2_fourier_hand_6dof_right_hand_pitch_link": "right_hand_pitch_link",
    }
    for task in tuple(getattr(controller_cfg, "variable_input_tasks", [])) + tuple(
        getattr(controller_cfg, "fixed_input_tasks", [])
    ):
        frame = getattr(task, "frame", None)
        if frame in frame_map:
            task.frame = frame_map[frame]
        task_class_type = str(getattr(task, "class_type", ""))
        if task_class_type.endswith(":FrameTask"):
            task.class_type = "isaaclab.controllers.pink_ik.pink_tasks:LocalFrameTask"
            task.base_link_frame_name = getattr(controller_cfg, "base_link_name", "base_link")

        controlled_frames = getattr(task, "controlled_frames", None)
        if isinstance(controlled_frames, list):
            task.controlled_frames = [
                frame_map.get(controlled_frame, controlled_frame) for controlled_frame in controlled_frames
            ]


def _configure_gr1t2_kitless_kinematics_asset(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Use a pre-generated GR1T2 URDF for Pink IK so kitless teleoperation never requires Isaac Sim conversion."""
    action_cfg = getattr(getattr(env_cfg, "actions", None), "upper_body_ik", None)
    controller_cfg = getattr(action_cfg, "controller", None)
    if controller_cfg is None or getattr(controller_cfg, "urdf_path", None):
        return

    task_frames = [
        getattr(task, "frame", "")
        for task in tuple(getattr(controller_cfg, "variable_input_tasks", []))
        + tuple(getattr(controller_cfg, "fixed_input_tasks", []))
    ]
    if not any("GR1T2_fourier_hand_6dof" in str(frame) for frame in task_frames):
        return

    candidate = _gr1t2_kitless_kinematics_candidate(env_cfg)
    if candidate is None:
        raise RuntimeError(
            "Kitless GR1T2 teleop requires a pre-generated Pink IK URDF. Set "
            "PickPlaceGR1T2EnvCfg.kitless_kinematics_urdf_path and, if needed, "
            "PickPlaceGR1T2EnvCfg.kitless_kinematics_mesh_path."
        )

    urdf_path, mesh_root, unprefixed_urdf = candidate
    if not urdf_path.is_file() or (mesh_root is not None and not mesh_root.is_dir()):
        raise RuntimeError(
            "Kitless GR1T2 teleop requires an existing Pink IK URDF and optional mesh root. "
            f"Got urdf_path={urdf_path} and mesh_path={mesh_root}."
        )

    controller_cfg.urdf_path = str(urdf_path)
    controller_cfg.mesh_path = str(mesh_root) if mesh_root is not None else None
    controller_cfg.usd_path = None
    _rewrite_gr1t2_kitless_task_frames(controller_cfg, unprefixed_urdf=unprefixed_urdf)
    print(f"Using kitless GR1T2 Pink IK URDF: {urdf_path}")


def _stabilize_gr1t2_actuators_for_kitless_newton(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Use Newton-friendly GR1T2 PD gains for kitless teleoperation."""
    robot_cfg = getattr(getattr(env_cfg, "scene", None), "robot", None)
    actuators = getattr(robot_cfg, "actuators", None)
    if not isinstance(actuators, dict):
        return

    tuned_actuators: list[str] = []
    for actuator_name, gains in _KITLESS_NEWTON_GR1T2_ACTUATOR_GAINS.items():
        actuator = actuators.get(actuator_name)
        if actuator is None:
            continue
        stiffness, damping, armature, effort_limit_sim = gains
        actuator.stiffness = stiffness
        actuator.damping = damping
        actuator.armature = armature
        actuator.effort_limit_sim = effort_limit_sim
        tuned_actuators.append(actuator_name)

    if tuned_actuators:
        print(
            "Applied kitless Newton GR1T2 actuator gains: "
            + ", ".join(f"{name}={_KITLESS_NEWTON_GR1T2_ACTUATOR_GAINS[name]}" for name in tuned_actuators)
        )


def _stabilize_gr1t2_ik_for_kitless_newton(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Use conservative Pink IK gains for GR1T2 kitless Newton teleoperation."""
    action_cfg = getattr(getattr(env_cfg, "actions", None), "upper_body_ik", None)
    controller_cfg = getattr(action_cfg, "controller", None)
    if controller_cfg is None:
        return

    tuned_any = False
    for task in tuple(getattr(controller_cfg, "variable_input_tasks", [])) + tuple(
        getattr(controller_cfg, "fixed_input_tasks", [])
    ):
        if hasattr(task, "lm_damping"):
            task.lm_damping = max(float(task.lm_damping), _KITLESS_NEWTON_GR1T2_IK_LM_DAMPING)
            tuned_any = True
        if hasattr(task, "gain"):
            task.gain = min(float(task.gain), _KITLESS_NEWTON_GR1T2_IK_GAIN)
            tuned_any = True
        if hasattr(task, "orientation_cost"):
            task.orientation_cost = min(float(task.orientation_cost), _KITLESS_NEWTON_GR1T2_IK_ORIENTATION_COST)
            tuned_any = True

    if tuned_any:
        print(
            "Applied kitless Newton GR1T2 Pink IK damping: "
            f"lm_damping>={_KITLESS_NEWTON_GR1T2_IK_LM_DAMPING}, "
            f"gain<={_KITLESS_NEWTON_GR1T2_IK_GAIN}, "
            f"orientation_cost<={_KITLESS_NEWTON_GR1T2_IK_ORIENTATION_COST}."
        )


##
# Scene definition
##
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the GR1T2 Pick Place Base Scene."""

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 0.9996], rot=[0.0, 0.0, 0.0, 1.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # Humanoid robot configured for pick-place manipulation tasks
    robot: ArticulationCfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.0, 0.0, 0.7071, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        hand_joint_names=[
            "L_index_proximal_joint",
            "L_index_intermediate_joint",
            "L_middle_proximal_joint",
            "L_middle_intermediate_joint",
            "L_pinky_proximal_joint",
            "L_pinky_intermediate_joint",
            "L_ring_proximal_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_yaw_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_distal_joint",
            "R_index_proximal_joint",
            "R_index_intermediate_joint",
            "R_middle_proximal_joint",
            "R_middle_intermediate_joint",
            "R_pinky_proximal_joint",
            "R_pinky_intermediate_joint",
            "R_ring_proximal_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_yaw_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_hand_pitch_link",
            "right_wrist": "right_hand_pitch_link",
        },
        # the robot in the sim scene we are controlling
        asset_name="robot",
        # Configuration for the IK controller
        # The frames names are the ones present in the URDF file
        # The urdf has to be generated from the USD that is being used in the scene
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=22,
            show_ik_warnings=False,
            # Determines whether Pink IK solver will fail due to a joint limit violation
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame="GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=12,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                FrameTaskCfg(
                    frame="GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    position_cost=8.0,  # [cost] / [m]
                    orientation_cost=1.0,  # [cost] / [rad]
                    lm_damping=12,  # dampening for solver for step jumps
                    gain=0.5,
                ),
                DampingTaskCfg(
                    cost=0.5,  # [cost] * [s] / [rad]
                ),
                NullSpacePostureTaskCfg(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                        "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                ),
            ],
            fixed_input_tasks=[],
        ),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_hand_roll_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_hand_roll_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"})

        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})
        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]},
        )

        object = ObsTerm(
            func=mdp.object_obs,
            params={"left_eef_link_name": "left_hand_roll_link", "right_eef_link_name": "right_hand_roll_link"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done_pick_place, params={"task_link_name": "right_hand_roll_link"})


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class PickPlaceGR1T2EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    # Pre-generated Pink IK URDF used by kitless teleoperation.
    kitless_kinematics_urdf_path: str | None = str(_GR1T2_KITLESS_KINEMATICS_URDF_PATH)

    # Optional mesh root for ``kitless_kinematics_urdf_path``.
    kitless_kinematics_mesh_path: str | None = None

    # Whether the kitless URDF uses unprefixed GR1T2 link names.
    kitless_kinematics_unprefixed_urdf: bool = False

    # Idle action to hold robot in default pose
    # Action format: [left arm pos (3), left arm quat (4), right arm pos (3), right arm quat (4),
    #                 left hand joint pos (11), right hand joint pos (11)]
    idle_action = [
        -0.22878,
        0.2536,
        1.0953,
        0.5,
        -0.5,
        0.5,
        0.5,
        0.22878,
        0.2536,
        1.0953,
        0.5,
        -0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]

    def apply_kitless_newton_overrides(self) -> None:
        """Apply GR1T2-specific settings for kitless Newton teleoperation."""
        _configure_gr1t2_newton_usd_imports()
        _configure_gr1t2_kitless_kinematics_asset(self)
        _stabilize_gr1t2_actuators_for_kitless_newton(self)
        _stabilize_gr1t2_ik_for_kitless_newton(self)

    def apply_kitless_teleop_overrides(self) -> None:
        """Apply GR1T2-specific non-physics settings for kitless teleoperation."""
        _configure_gr1t2_newton_usd_imports()
        _configure_gr1t2_kitless_kinematics_asset(self)

    def create_kitless_debug_idle_device(self, env, device: str | torch.device | None):
        """Create a synthetic hold-pose teleop device for no-client kitless debugging."""
        return _KitlessNewtonIdleActionDevice(env, device)

    def install_kitless_teleop_runtime_hooks(self, env) -> None:
        """Install runtime safety hooks for GR1T2 kitless teleoperation."""
        _install_kitless_newton_pink_ik_guard(env)

    def condition_kitless_teleop_action(
        self, env, action: torch.Tensor, state: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Condition GR1T2 teleop actions before stepping kitless Newton physics."""
        return _condition_kitless_newton_gr1t2_action(env, action, state)

    def teleop_action_target_issue(self, action: torch.Tensor) -> str | None:
        """Return a human-readable issue for unsafe GR1T2 teleop targets, if any."""
        return _teleop_action_target_issue(action)

    def format_kitless_teleop_action(self, action: torch.Tensor) -> str:
        """Format a GR1T2 teleop action for concise diagnostics."""
        return _format_teleop_wrist_targets(action)

    def log_kitless_teleop_diagnostics(self, env, label: str) -> None:
        """Print GR1T2 kitless teleop diagnostics when body state is available."""
        _log_gr1t2_pose_diagnostics(env, label)

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2
        self.sim.physics = PickPlaceGR1T2PhysicsCfg()
        self.scene.robot.actuators.setdefault(
            "head",
            ImplicitActuatorCfg(
                joint_names_expr=["head_.*"],
                effort_limit=None,
                velocity_limit=None,
                stiffness=4400.0,
                damping=40.0,
                armature=0.01,
            ),
        )
        self.scene.robot.actuators.setdefault(
            "legs",
            ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
                effort_limit=None,
                velocity_limit=None,
                stiffness=4400.0,
                damping=40.0,
                armature=0.01,
            ),
        )

        # Defer USD→URDF conversion to controller initialization (requires Isaac Sim at runtime).
        self.actions.upper_body_ik.controller.usd_path = self.scene.robot.spawn.usd_path
        self.actions.upper_body_ik.controller.urdf_output_dir = self.temp_urdf_dir

        # IsaacTeleop-based teleoperation pipeline.
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.0),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
        )
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda: _build_gr1t2_pickplace_pipeline()[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )
