# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility hooks for proxy-coupled Newton implicit MPM."""

from __future__ import annotations

import warp as wp

from .constants import BODY_FLAG_PROXY


def apply_mpm_proxy_compat() -> None:
    """Install PR-2848 MPM proxy hooks when normal Newton does not provide them."""
    from newton.solvers import SolverImplicitMPM  # noqa: PLC0415

    if not hasattr(SolverImplicitMPM, "coupling_rewind_proxy_body_velocity"):
        SolverImplicitMPM.coupling_rewind_proxy_body_velocity = _coupling_rewind_proxy_body_velocity
    if not hasattr(SolverImplicitMPM, "coupling_harvest_proxy_wrenches"):
        SolverImplicitMPM.coupling_harvest_proxy_wrenches = _coupling_harvest_proxy_wrenches


def _coupling_rewind_proxy_body_velocity(
    self,
    body_local_to_proxy_global: wp.array[int],
    state,
    coupling_forces: wp.array[wp.spatial_vector],
    dt: float,
) -> None:
    """Remove lagged proxy wrenches from MPM collider body velocities."""
    if state.body_q is None or state.body_qd is None or body_local_to_proxy_global.shape[0] == 0:
        return

    wp.launch(
        _rewind_mpm_proxy_bodies_kernel,
        dim=body_local_to_proxy_global.shape[0],
        inputs=[
            float(dt),
            body_local_to_proxy_global,
            coupling_forces,
            state.body_q,
            self.model.body_inv_inertia,
            self.model.body_inv_mass,
            state.body_qd,
        ],
        device=self.model.device,
    )


def _coupling_harvest_proxy_wrenches(
    self,
    body_local_to_proxy_global: wp.array[int],
    out_body_f: wp.array[wp.spatial_vector],
    *,
    state=None,
    state_out=None,
    contacts=None,
    dt: float = 0.0,
) -> None:
    """Convert MPM collider grid impulses to proxy-body wrenches."""
    del state_out, contacts
    if dt <= 0.0:
        raise ValueError("MPM proxy wrench harvesting requires a positive dt")

    impulses, positions, collider_ids = self.collect_collider_impulses(state)
    if collider_ids.shape[0] == 0:
        return
    body_q = state.body_q if state is not None and state.body_q is not None else self.model.body_q

    wp.launch(
        _harvest_mpm_proxy_wrenches_kernel,
        dim=collider_ids.shape[0],
        inputs=[
            float(dt),
            collider_ids,
            impulses,
            positions,
            self.collider_body_index,
            body_local_to_proxy_global,
            BODY_FLAG_PROXY,
            self.model.body_flags,
            self.model.body_com,
            body_q,
            out_body_f,
        ],
        device=self.model.device,
    )


@wp.kernel(enable_backward=False)
def _rewind_mpm_proxy_bodies_kernel(
    dt: float,
    body_local_to_proxy_global: wp.array[int],
    coupling_forces: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_inv_inertia: wp.array[wp.mat33],
    body_inv_mass: wp.array[float],
    body_qd: wp.array[wp.spatial_vector],
):
    local_body = wp.tid()
    proxy_global = body_local_to_proxy_global[local_body]
    if proxy_global < 0:
        return

    f = coupling_forces[proxy_global]
    delta_v = dt * body_inv_mass[local_body] * wp.spatial_top(f)
    rot = wp.transform_get_rotation(body_q[local_body])
    delta_w = dt * wp.quat_rotate(
        rot,
        body_inv_inertia[local_body] * wp.quat_rotate_inv(rot, wp.spatial_bottom(f)),
    )

    body_qd[local_body] = body_qd[local_body] - wp.spatial_vector(delta_v, delta_w)


@wp.kernel(enable_backward=False)
def _harvest_mpm_proxy_wrenches_kernel(
    dt: float,
    collider_ids: wp.array[int],
    collider_impulses: wp.array[wp.vec3],
    collider_impulse_pos: wp.array[wp.vec3],
    collider_body_ids: wp.array[int],
    body_local_to_proxy_global: wp.array[int],
    proxy_flag: int,
    body_flags: wp.array[wp.int32],
    body_com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    out_body_f: wp.array[wp.spatial_vector],
):
    i = wp.tid()
    cid = collider_ids[i]

    if cid < 0 or cid >= collider_body_ids.shape[0]:
        return

    local_body = collider_body_ids[cid]
    if local_body < 0 or local_body >= body_local_to_proxy_global.shape[0]:
        return

    proxy_global = body_local_to_proxy_global[local_body]
    if proxy_global < 0 or proxy_global >= out_body_f.shape[0] or (body_flags[local_body] & proxy_flag) == 0:
        return

    f_world = collider_impulses[i] / dt
    center = wp.transform_point(body_q[local_body], body_com[local_body])
    r = collider_impulse_pos[i] - center
    wp.atomic_add(out_body_f, proxy_global, wp.spatial_vector(f_world, wp.cross(r, f_world)))
