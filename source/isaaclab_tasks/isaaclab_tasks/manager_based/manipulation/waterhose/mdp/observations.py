# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .. import waterhose_core as core

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ensure_metadata(env: ManagerBasedRLEnv) -> None:
    if hasattr(env, "waterhose_right_ee_body_ids"):
        return
    scene_builder = env.waterhose_scene_builder
    labels = core.NewtonManager.get_model().body_label

    def resolve_short(short_name: str) -> list[int]:
        suffix = "/" + short_name
        matches = [idx for idx, label in enumerate(labels) if label == short_name or label.endswith(suffix)]
        if len(matches) < env.num_envs:
            raise RuntimeError(f"Expected at least {env.num_envs} bodies named {short_name!r}, found {matches}.")
        return matches[: env.num_envs]

    def resolve_label(label: str) -> list[int]:
        matches = [idx for idx, candidate in enumerate(labels) if candidate == label]
        if len(matches) < env.num_envs:
            if env.num_envs == 1 and label in labels:
                return [labels.index(label)]
            raise RuntimeError(f"Expected {env.num_envs} bodies matching {label!r}, found {matches}.")
        return matches[: env.num_envs]

    env.waterhose_right_ee_body_ids = resolve_short(core.RIGHT_EE)
    env.waterhose_tip_body_ids = resolve_label(labels[scene_builder.tip_body_id])
    env.waterhose_plug_body_ids = resolve_label(labels[scene_builder.plug_body_id])
    socket_pos = np.array([float(scene_builder.socket_pos[i]) for i in range(3)], dtype=np.float64)
    socket_rot = np.array([float(scene_builder.socket_rot[i]) for i in range(4)], dtype=np.float64)
    env.waterhose_socket_pose = np.concatenate((socket_pos, socket_rot), axis=0)


def _body_pose_tensor(env: ManagerBasedRLEnv, body_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    _ensure_metadata(env)
    body_q = core.NewtonManager.get_state_0().body_q.numpy()
    data = torch.as_tensor(body_q[body_ids], device=env.device, dtype=torch.float32)
    return data[:, :3], data[:, 3:7]


def eef_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Right end-effector position [m]."""
    _ensure_metadata(env)
    return _body_pose_tensor(env, env.waterhose_right_ee_body_ids)[0]


def eef_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Right end-effector orientation quaternion ``xyzw``."""
    _ensure_metadata(env)
    return _body_pose_tensor(env, env.waterhose_right_ee_body_ids)[1]


def plug_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Waterhose plug position [m]."""
    _ensure_metadata(env)
    return _body_pose_tensor(env, env.waterhose_plug_body_ids)[0]


def plug_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Waterhose plug orientation quaternion ``xyzw``."""
    _ensure_metadata(env)
    return _body_pose_tensor(env, env.waterhose_plug_body_ids)[1]


def tip_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Waterhose tip position [m]."""
    _ensure_metadata(env)
    return _body_pose_tensor(env, env.waterhose_tip_body_ids)[0]


def tip_quat(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Waterhose tip orientation quaternion ``xyzw``."""
    _ensure_metadata(env)
    return _body_pose_tensor(env, env.waterhose_tip_body_ids)[1]


def socket_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Socket pose as position [m] and quaternion ``xyzw``."""
    _ensure_metadata(env)
    socket = torch.as_tensor(env.waterhose_socket_pose, device=env.device, dtype=torch.float32)
    return socket.unsqueeze(0).repeat(env.num_envs, 1)


def alignment(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Tip lateral error [m], insertion depth [m], and axis alignment cosine."""
    _ensure_metadata(env)
    body_q = core.NewtonManager.get_state_0().body_q.numpy()
    socket_pos = env.waterhose_socket_pose[:3]
    socket_quat = env.waterhose_socket_pose[3:]
    insertion_dir = core._np_quat_rotate(socket_quat, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    insertion_dir /= max(np.linalg.norm(insertion_dir), 1.0e-12)
    values = []
    for tip_id in env.waterhose_tip_body_ids:
        tip_q = body_q[tip_id]
        delta = tip_q[:3] - (socket_pos + float(env.cfg.insert_start_depth) * insertion_dir)
        lateral = delta - np.dot(delta, insertion_dir) * insertion_dir
        axis = core._np_quat_rotate(tip_q[3:], np.array([0.0, 0.0, 1.0], dtype=np.float64))
        axis /= max(np.linalg.norm(axis), 1.0e-12)
        values.append(
            [
                float(np.linalg.norm(lateral)),
                float(np.dot(tip_q[:3] - socket_pos, insertion_dir)),
                float(np.dot(axis, insertion_dir)),
            ]
        )
    return torch.as_tensor(values, device=env.device, dtype=torch.float32)


def proxy_wrench_norm(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Norm of proxy coupling forces [N]."""
    _ensure_metadata(env)
    wrenches = core.NewtonCoupledManager.get_proxy_body_wrenches(core.ROBOT_ENTRY, core.HOSE_ENTRY)
    result = torch.zeros(env.num_envs, 1, device=env.device)
    if wrenches is None:
        return result
    wrench_np = wrenches.numpy()
    proxy_ids = [body_id for body_id in env.waterhose_scene_builder.proxy_body_ids if body_id < wrench_np.shape[0]]
    if not proxy_ids:
        return result
    # The proxy ids are grouped by env because the builder appends robots env-by-env.
    chunks = np.array_split(np.asarray(proxy_ids, dtype=np.int32), env.num_envs)
    for env_id, ids in enumerate(chunks):
        if ids.size:
            result[env_id, 0] = float(np.linalg.norm(wrench_np[ids, :3]))
    return result


def joint_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """RBY1 joint positions [m or rad, depending on joint type]."""
    _ensure_metadata(env)
    joint_q = core.NewtonManager.get_state_0().joint_q.numpy()
    values = [joint_q[ids] for ids in env.waterhose_scene_builder.robot_joint_coord_ids_by_env]
    return torch.as_tensor(np.asarray(values), device=env.device, dtype=torch.float32)


def gripper_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Right gripper joint positions [m]."""
    q = joint_pos(env)
    dofs = env.waterhose_scene_builder.right_gripper_dofs[:2]
    if not dofs:
        return torch.zeros(env.num_envs, 1, device=env.device)
    return q[:, dofs]


def waterhose_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compact task observation vector."""
    return torch.cat(
        (
            eef_pos(env),
            eef_quat(env),
            plug_pos(env),
            plug_quat(env),
            tip_pos(env),
            alignment(env),
            proxy_wrench_norm(env),
        ),
        dim=-1,
    )


def approach_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the TCP is near the hose plug."""
    return torch.linalg.norm(eef_pos(env) - plug_pos(env), dim=-1) < 0.035


def grasp_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the hose is likely grasped by the right gripper."""
    return torch.logical_and(approach_done(env), proxy_wrench_norm(env).squeeze(-1) > 1.0)


def align_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the hose tip is aligned to the socket."""
    errors = alignment(env)
    return torch.logical_and(
        errors[:, 0] < env.cfg.success_lateral_threshold,
        errors[:, 2] < env.cfg.success_axis_cosine,
    )


def insert_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Whether the hose tip is inserted into the socket."""
    errors = alignment(env)
    return torch.logical_and(align_done(env), errors[:, 1] > env.cfg.success_insert_depth)
