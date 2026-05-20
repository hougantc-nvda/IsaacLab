# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Coupled Newton multi-solver manager."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from newton import Model
from newton.solvers.coupled_experimental import SolverAdmmCoupled, SolverCoupled, SolverProxyCoupled

from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsManager

from .coupled_manager_cfg import (
    AdmmContactPairCfg,
    AdmmCouplingCfg,
    CoupledProxyCfg,
    CoupledSolverCfg,
    CoupledSolverEntryCfg,
)
from .mjwarp_manager import apply_mujoco_warp_model_overrides
from .newton_manager import NewtonManager
from .solver_factory import (
    resolve_class_or_callable,
    resolve_newton_solver_class_and_kwargs,
    solver_cfg_needs_external_contacts,
)

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg


class NewtonCoupledManager(NewtonManager):
    """:class:`NewtonManager` specialization for Newton coupled solvers.

    The manager is intentionally thin: Isaac Lab owns lifecycle, state buffers,
    collision-pipeline refresh, and visualization, while Newton's coupled
    solvers own per-solver ``ModelView`` construction and
    cross-entry force or constraint exchange.
    """

    @classmethod
    def get_entry_solver(cls, name: str):
        """Return a named sub-solver from the active coupled solver."""
        solver = NewtonManager._solver
        if solver is None:
            raise RuntimeError("Newton coupled solver is not initialized.")
        return solver.solver(name)

    @classmethod
    def get_entry_view(cls, name: str):
        """Return a named sub-solver model view from the active coupled solver."""
        solver = NewtonManager._solver
        if solver is None:
            raise RuntimeError("Newton coupled solver is not initialized.")
        return solver.view(name)

    @classmethod
    def get_proxy_body_wrenches(cls, source: str, destination: str):
        """Return proxy body feedback wrenches when the active Newton solver exposes them."""
        solver = NewtonManager._solver
        if solver is None:
            return None
        for mapping in getattr(solver, "_proxy_mappings", ()):
            if mapping.src_name == source and mapping.dst_name == destination:
                return mapping.coupling_forces
        return None

    @classmethod
    def _build_solver(cls, model: Model, solver_cfg: CoupledSolverCfg) -> None:
        """Construct a Newton coupled solver and populate the base-class slots."""
        solver_cfg = cls._resolve_solver_cfg(model, solver_cfg)
        cls._validate_solver_cfg(solver_cfg)

        entries = [cls._build_entry(entry_cfg) for entry_cfg in solver_cfg.entries]
        if solver_cfg.coupling_type == "base":
            NewtonManager._solver = SolverCoupled(model=model, entries=entries)
        elif solver_cfg.coupling_type == "proxy":
            NewtonManager._solver = SolverProxyCoupled(
                model=model,
                entries=entries,
                coupling=SolverProxyCoupled.Config(
                    proxies=[cls._build_proxy(proxy_cfg) for proxy_cfg in solver_cfg.proxy_coupling.proxies],
                    iterations=solver_cfg.proxy_coupling.iterations,
                ),
            )
        elif solver_cfg.coupling_type == "admm":
            NewtonManager._solver = SolverAdmmCoupled(
                model=model,
                entries=entries,
                coupling=cls._build_admm(solver_cfg.admm_coupling, entries),
            )
        else:
            raise ValueError(f"Unsupported Newton coupling_type {solver_cfg.coupling_type!r}.")

        cls._apply_entry_solver_overrides(solver_cfg.entries)
        cls._configure_fk_articulation_filter(model, solver_cfg.entries)
        if hasattr(NewtonManager._solver, "prepare_graph_capture"):
            NewtonManager._solver.prepare_graph_capture()
        NewtonManager._use_single_state = False
        NewtonManager._needs_collision_pipeline = cls._needs_external_collision_pipeline(solver_cfg)

    @classmethod
    def _resolve_solver_cfg(cls, model: Model, solver_cfg: CoupledSolverCfg) -> CoupledSolverCfg:
        """Return a shallow copy of ``solver_cfg`` with selector fields resolved to ids."""
        scene_cfg = cls._resolve_scene_cfg(solver_cfg)
        resolved_cfg = copy.copy(solver_cfg)
        resolved_cfg.entries = [
            cls._resolve_entry_cfg(model, entry_cfg, scene_cfg) for entry_cfg in solver_cfg.entries
        ]
        resolved_proxy_coupling = copy.copy(solver_cfg.proxy_coupling)
        resolved_proxy_coupling.proxies = [
            cls._resolve_proxy_cfg(model, proxy_cfg, scene_cfg) for proxy_cfg in solver_cfg.proxy_coupling.proxies
        ]
        resolved_cfg.proxy_coupling = resolved_proxy_coupling
        return resolved_cfg

    @staticmethod
    def _resolve_scene_cfg(solver_cfg: CoupledSolverCfg):
        """Resolve the scene cfg used by ``SceneEntityCfg`` selectors."""
        if solver_cfg.scene_cfg is not None:
            return solver_cfg.scene_cfg
        return getattr(PhysicsManager._cfg, "scene_cfg", None)

    @classmethod
    def _resolve_entry_cfg(
        cls, model: Model, entry_cfg: CoupledSolverEntryCfg, scene_cfg: InteractiveSceneCfg | None
    ) -> CoupledSolverEntryCfg:
        """Resolve one entry's front-end selectors into raw Newton index lists."""
        resolved = copy.copy(entry_cfg)
        body_selector_used = cls._uses_body_selectors(entry_cfg)
        selected_bodies = cls._resolve_body_selectors(model, entry_cfg, scene_cfg, f"entry {entry_cfg.name!r}")
        bodies = cls._unique_ints([*entry_cfg.bodies, *selected_bodies])
        joints = list(entry_cfg.joints)
        shapes = list(entry_cfg.shapes)
        if body_selector_used:
            if entry_cfg.include_child_joints:
                joints.extend(cls._child_joints_for_bodies(model, bodies))
            if entry_cfg.include_body_shapes or entry_cfg.include_static_shapes:
                shapes.extend(
                    cls._shapes_for_bodies(
                        model,
                        bodies,
                        include_body_shapes=entry_cfg.include_body_shapes,
                        include_static_shapes=entry_cfg.include_static_shapes,
                    )
                )

        resolved.bodies = bodies
        resolved.joints = cls._unique_ints(joints)
        resolved.shapes = cls._unique_ints(shapes)
        resolved.particles = cls._resolve_particles(
            model,
            explicit=entry_cfg.particles,
            particle_range=entry_cfg.particle_range,
            all_particles=entry_cfg.all_particles,
            field=f"CoupledSolverEntryCfg {entry_cfg.name!r}",
        )
        return resolved

    @classmethod
    def _resolve_proxy_cfg(
        cls, model: Model, proxy_cfg: CoupledProxyCfg, scene_cfg: InteractiveSceneCfg | None
    ) -> CoupledProxyCfg:
        """Resolve one proxy cfg's selectors into raw Newton index lists."""
        resolved = copy.copy(proxy_cfg)
        selected_bodies = cls._resolve_body_selectors(
            model,
            proxy_cfg,
            scene_cfg,
            f"proxy {proxy_cfg.source!r}->{proxy_cfg.destination!r}",
        )
        resolved.bodies = cls._unique_ints([*proxy_cfg.bodies, *selected_bodies])
        resolved.particles = cls._resolve_particles(
            model,
            explicit=proxy_cfg.particles,
            particle_range=proxy_cfg.particle_range,
            all_particles=proxy_cfg.all_particles,
            field=f"CoupledProxyCfg {proxy_cfg.source!r}->{proxy_cfg.destination!r}",
        )
        return resolved

    @staticmethod
    def _uses_body_selectors(cfg: CoupledSolverEntryCfg | CoupledProxyCfg) -> bool:
        return bool(cfg.body_entities or cfg.body_label_patterns or cfg.body_name_patterns)

    @classmethod
    def _resolve_body_selectors(
        cls,
        model: Model,
        cfg: CoupledSolverEntryCfg | CoupledProxyCfg,
        scene_cfg: InteractiveSceneCfg | None,
        field: str,
    ) -> list[int]:
        body_ids: list[int] = []
        if cfg.body_entities:
            if scene_cfg is None:
                raise ValueError(
                    f"{type(cfg).__name__} {field} uses body_entities, but CoupledSolverCfg.scene_cfg is not set. "
                    "Set scene_cfg=self.scene in the coupled solver cfg or use body_label_patterns/body_name_patterns."
                )
            for entity_cfg in cfg.body_entities:
                body_ids.extend(cls._resolve_entity_to_body_ids(model, entity_cfg, scene_cfg, field))
        body_ids.extend(cls._resolve_body_label_patterns(model, cfg.body_label_patterns, field))
        body_ids.extend(cls._resolve_body_name_patterns(model, cfg.body_name_patterns, field))
        return cls._unique_ints(body_ids)

    @classmethod
    def _resolve_entity_to_body_ids(
        cls,
        model: Model,
        entity_cfg: SceneEntityCfg,
        scene_cfg: InteractiveSceneCfg,
        field: str,
    ) -> list[int]:
        """Resolve one ``SceneEntityCfg`` to Newton body ids."""
        asset_cfg = getattr(scene_cfg, entity_cfg.name, None)
        if asset_cfg is None or not hasattr(asset_cfg, "prim_path"):
            raise ValueError(
                f"CoupledSolverCfg {field} references scene entity {entity_cfg.name!r}, "
                "which is not present on scene_cfg or lacks prim_path."
            )

        asset_pattern = str(asset_cfg.prim_path).replace("{ENV_REGEX_NS}", r"/World/envs/env_.*")
        asset_regex = re.compile(rf"^{asset_pattern}(/|$)")
        labels = cls._body_labels(model)
        candidate_ids = [body_id for body_id, label in enumerate(labels) if asset_regex.match(label)]
        patterns = entity_cfg.body_names
        if isinstance(patterns, str):
            patterns = [patterns]
        if patterns is None:
            body_ids = cls._select_entity_body_ids(candidate_ids, entity_cfg.body_ids, field, entity_cfg.name)
            if not candidate_ids:
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {entity_cfg.name!r} matched no Newton bodies "
                    f"under prim_path regex {asset_pattern!r}."
                )
            if not body_ids:
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {entity_cfg.name!r} body_ids selected no bodies "
                    f"from {len(candidate_ids)} candidate Newton bodies."
                )
            return body_ids
        if not cls._is_all_slice(entity_cfg.body_ids):
            raise ValueError(
                f"CoupledSolverCfg {field}: scene entity {entity_cfg.name!r} sets both body_names and body_ids. "
                "Use only one selector to avoid ambiguous Newton body ownership."
            )

        compiled = [re.compile(pattern) for pattern in patterns]
        matched = [False] * len(compiled)
        body_ids: list[int] = []
        if entity_cfg.preserve_order:
            for index, pattern in enumerate(compiled):
                matches = [
                    body_id for body_id in candidate_ids if pattern.fullmatch(labels[body_id].rsplit("/", 1)[-1])
                ]
                if matches:
                    matched[index] = True
                    body_ids.extend(matches)
        else:
            for body_id in candidate_ids:
                short_name = labels[body_id].rsplit("/", 1)[-1]
                hit = next((index for index, pattern in enumerate(compiled) if pattern.fullmatch(short_name)), None)
                if hit is None:
                    continue
                matched[hit] = True
                body_ids.append(body_id)

        unmatched = [pattern for pattern, ok in zip(patterns, matched) if not ok]
        if unmatched:
            raise ValueError(
                f"CoupledSolverCfg {field}: scene entity {entity_cfg.name!r} has no Newton bodies matching "
                f"{unmatched}. Check the regexes against body short names."
            )
        return cls._unique_ints(body_ids)

    @staticmethod
    def _is_all_slice(value) -> bool:
        return isinstance(value, slice) and value.start is None and value.stop is None and value.step is None

    @classmethod
    def _select_entity_body_ids(cls, candidate_ids: list[int], selector, field: str, entity_name: str) -> list[int]:
        """Apply an entity-local ``body_ids`` selector to candidate Newton body ids."""
        if cls._is_all_slice(selector):
            return candidate_ids
        if isinstance(selector, int):
            selector = [selector]
        if isinstance(selector, slice):
            return candidate_ids[selector]

        body_ids: list[int] = []
        for raw_index in selector:
            local_index = int(raw_index)
            if local_index < 0:
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {entity_name!r} body_ids index {local_index} is "
                    "negative. Use non-negative entity-local body ids."
                )
            try:
                body_ids.append(candidate_ids[local_index])
            except IndexError as exc:
                raise ValueError(
                    f"CoupledSolverCfg {field}: scene entity {entity_name!r} body_ids index {local_index} is "
                    f"outside the matched Newton body range [0, {len(candidate_ids)})."
                ) from exc
        return body_ids

    @classmethod
    def _resolve_body_label_patterns(cls, model: Model, patterns: list[str], field: str) -> list[int]:
        """Resolve full-body-label regexes to body ids."""
        labels = cls._body_labels(model)
        return cls._resolve_body_patterns(labels, patterns, field, "body_label_patterns")

    @classmethod
    def _resolve_body_name_patterns(cls, model: Model, patterns: list[str], field: str) -> list[int]:
        """Resolve short-body-name regexes to body ids."""
        labels = cls._body_labels(model)
        short_names = [label.rsplit("/", 1)[-1] for label in labels]
        return cls._resolve_body_patterns(short_names, patterns, field, "body_name_patterns")

    @staticmethod
    def _resolve_body_patterns(
        match_values: list[str], patterns: list[str], field: str, selector_name: str
    ) -> list[int]:
        body_ids: list[int] = []
        for pattern in patterns:
            regex = re.compile(pattern)
            matches = [body_id for body_id, value in enumerate(match_values) if regex.fullmatch(value)]
            if not matches:
                raise ValueError(f"CoupledSolverCfg {field}: {selector_name} pattern {pattern!r} matched no bodies.")
            body_ids.extend(matches)
        return body_ids

    @staticmethod
    def _body_labels(model: Model) -> list[str]:
        labels = getattr(model, "body_label", None) or getattr(model, "body_key", None)
        if labels is None:
            raise ValueError("Newton model does not expose body_label/body_key; body selectors cannot be resolved.")
        return [str(label) for label in labels]

    @classmethod
    def _child_joints_for_bodies(cls, model: Model, body_ids: list[int]) -> list[int]:
        """Return joints whose child body is in ``body_ids``."""
        if int(getattr(model, "joint_count", 0)) <= 0 or getattr(model, "joint_child", None) is None:
            return []
        owned = set(body_ids)
        return [joint_id for joint_id, child in enumerate(model.joint_child.numpy()) if int(child) in owned]

    @classmethod
    def _shapes_for_bodies(
        cls,
        model: Model,
        body_ids: list[int],
        *,
        include_body_shapes: bool,
        include_static_shapes: bool,
    ) -> list[int]:
        """Return shapes attached to selected bodies and optionally static shapes."""
        if int(getattr(model, "shape_count", 0)) <= 0 or getattr(model, "shape_body", None) is None:
            return []
        owned = set(body_ids)
        shape_ids: list[int] = []
        for shape_id, body_id_raw in enumerate(model.shape_body.numpy()):
            body_id = int(body_id_raw)
            if (include_body_shapes and body_id in owned) or (include_static_shapes and body_id < 0):
                shape_ids.append(shape_id)
        return shape_ids

    @classmethod
    def _resolve_particles(
        cls,
        model: Model,
        *,
        explicit: list[int],
        particle_range: tuple[int | None, int | None] | None,
        all_particles: bool,
        field: str,
    ) -> list[int]:
        particle_count = int(getattr(model, "particle_count", 0))
        particles = list(explicit)
        if all_particles:
            particles.extend(range(particle_count))
        if particle_range is not None:
            start_raw, end_raw = particle_range
            start = 0 if start_raw is None else int(start_raw)
            end = particle_count if end_raw is None else int(end_raw)
            if start < 0 or end < start or end > particle_count:
                raise ValueError(
                    f"{field}.particle_range must satisfy 0 <= start <= end <= particle_count "
                    f"({particle_count}), got ({start}, {end})."
                )
            particles.extend(range(start, end))
        return cls._unique_ints(particles)

    @staticmethod
    def _unique_ints(values) -> list[int]:
        seen: set[int] = set()
        result: list[int] = []
        for value in values:
            index = int(value)
            if index in seen:
                continue
            seen.add(index)
            result.append(index)
        return result

    @classmethod
    def _apply_entry_solver_overrides(cls, entries: list[CoupledSolverEntryCfg]) -> None:
        """Apply post-construction solver cfg overrides for coupled sub-solvers."""
        for entry_cfg in entries:
            if getattr(entry_cfg.solver_cfg, "solver_type", None) != "mujoco_warp":
                continue
            apply_mujoco_warp_model_overrides(NewtonManager._solver.solver(entry_cfg.name), entry_cfg.solver_cfg)

    @classmethod
    def _configure_fk_articulation_filter(cls, model: Model, entries: list[CoupledSolverEntryCfg]) -> None:
        """Exclude solver-owned VBD articulations from NewtonManager's generic FK path."""
        if model.articulation_count <= 0 or getattr(model, "joint_articulation", None) is None:
            NewtonManager._set_fk_articulation_filter(None)
            return

        fk_mask = np.ones(int(model.articulation_count), dtype=bool)
        joint_articulation = model.joint_articulation.numpy()
        disabled_any = False
        for entry_cfg in entries:
            solver_class, _ = resolve_newton_solver_class_and_kwargs(
                entry_cfg.solver_cfg,
                entry_cfg.solver_class,
                entry_cfg.solver_kwargs,
            )
            if getattr(solver_class, "__name__", "") != "SolverVBD":
                continue
            for joint_id in entry_cfg.joints:
                joint_index = int(joint_id)
                if joint_index < 0 or joint_index >= joint_articulation.shape[0]:
                    continue
                articulation_id = int(joint_articulation[joint_index])
                if articulation_id < 0:
                    continue
                fk_mask[articulation_id] = False
                disabled_any = True

        NewtonManager._set_fk_articulation_filter(fk_mask if disabled_any else None)

    @classmethod
    def _build_entry(cls, entry_cfg: CoupledSolverEntryCfg) -> SolverCoupled.Entry:
        """Build a Newton ``SolverCoupled.Entry`` from an Isaac Lab entry cfg."""
        solver_class, solver_kwargs = resolve_newton_solver_class_and_kwargs(
            entry_cfg.solver_cfg,
            entry_cfg.solver_class,
            entry_cfg.solver_kwargs,
        )
        configure_view = (
            None if entry_cfg.configure_view is None else resolve_class_or_callable(entry_cfg.configure_view)
        )

        entry_kwargs = dict(
            name=entry_cfg.name,
            solver=cls._make_entry_solver_factory(solver_class, solver_kwargs),
            bodies=list(entry_cfg.bodies),
            particles=list(entry_cfg.particles),
            joints=list(entry_cfg.joints),
            shapes=list(entry_cfg.shapes),
            configure_view=configure_view,
            substeps=entry_cfg.substeps,
            in_place=entry_cfg.in_place,
        )

        return SolverCoupled.Entry(**entry_kwargs)

    @staticmethod
    def _make_entry_solver_factory(solver_class: Callable, solver_kwargs: dict) -> Callable:
        """Bind constructor kwargs into a Newton coupled entry solver factory."""

        def _factory(model_view):
            return solver_class(model_view, **solver_kwargs)

        _factory.__name__ = getattr(solver_class, "__name__", type(solver_class).__name__)
        return _factory

    @classmethod
    def _build_proxy(cls, proxy_cfg: CoupledProxyCfg) -> SolverProxyCoupled.Proxy:
        """Build a Newton proxy mapping from an Isaac Lab proxy cfg."""
        if not proxy_cfg.source or not proxy_cfg.destination:
            raise ValueError("CoupledProxyCfg source and destination must be non-empty.")
        if not proxy_cfg.bodies and not proxy_cfg.particles:
            raise ValueError("CoupledProxyCfg must map at least one body or particle.")

        return SolverProxyCoupled.Proxy(
            source=proxy_cfg.source,
            destination=proxy_cfg.destination,
            bodies=list(proxy_cfg.bodies),
            proxy_bodies=None if proxy_cfg.proxy_bodies is None else list(proxy_cfg.proxy_bodies),
            mass_scale=proxy_cfg.mass_scale,
            mode=cls._build_proxy_mode(proxy_cfg.mode),
            particles=list(proxy_cfg.particles),
            proxy_particles=None if proxy_cfg.proxy_particles is None else list(proxy_cfg.proxy_particles),
            collision_pipeline=proxy_cfg.collision_pipeline_factory,
            collide_interval=proxy_cfg.collide_interval,
        )

    @staticmethod
    def _build_proxy_mode(mode: str | int) -> str:
        """Return the Newton proxy mode string for an Isaac Lab proxy cfg mode."""
        if isinstance(mode, str):
            return mode
        if mode == 0:
            return "lagged"
        if mode == 1:
            return "staggered"
        raise ValueError(f"Unsupported CoupledProxyCfg mode {mode!r}; expected 'lagged', 'staggered', 0, or 1.")

    @classmethod
    def _build_admm(
        cls, admm_cfg: AdmmCouplingCfg, entries: list[SolverCoupled.Entry] | None = None
    ) -> SolverAdmmCoupled.Config:
        """Build a Newton ADMM coupling config from an Isaac Lab cfg."""
        contact_pairs = [cls._build_admm_contact_pair(pair_cfg) for pair_cfg in admm_cfg.contact_pairs]
        if admm_cfg.auto_contact_pairs:
            if entries is None:
                raise ValueError("AdmmCouplingCfg.auto_contact_pairs requires coupled solver entries.")
            contact_pairs.extend(
                SolverAdmmCoupled.auto_detect_contact_pairs(
                    entries,
                    contact_distance=admm_cfg.auto_contact_distance,
                    detection_margin=admm_cfg.auto_detection_margin,
                )
            )

        return SolverAdmmCoupled.Config(
            iterations=admm_cfg.iterations,
            rho=admm_cfg.rho,
            gamma=admm_cfg.gamma,
            baumgarte=admm_cfg.baumgarte,
            joint_stiffness=admm_cfg.joint_stiffness,
            joint_damping=admm_cfg.joint_damping,
            joint_angular_stiffness=admm_cfg.joint_angular_stiffness,
            joint_angular_damping=admm_cfg.joint_angular_damping,
            contact_pairs=contact_pairs,
        )

    @staticmethod
    def _build_admm_contact_pair(pair_cfg: AdmmContactPairCfg) -> SolverAdmmCoupled.ContactPair:
        """Build a Newton ADMM contact-pair config from an Isaac Lab cfg."""
        return SolverAdmmCoupled.ContactPair(
            source=pair_cfg.source,
            destination=pair_cfg.destination,
            contact_distance=pair_cfg.contact_distance,
            detection_margin=pair_cfg.detection_margin,
        )

    @classmethod
    def _validate_solver_cfg(cls, solver_cfg: CoupledSolverCfg) -> None:
        """Validate coupled-solver config before constructing Newton objects."""
        if solver_cfg.coupling_type not in ("base", "proxy", "admm"):
            raise ValueError(f"Unsupported Newton coupling_type {solver_cfg.coupling_type!r}.")
        if len(solver_cfg.entries) < 2:
            raise ValueError("Newton coupled solver requires at least two solver entries.")
        cls._validate_entries(solver_cfg.entries)
        if solver_cfg.coupling_type == "base":
            return
        if solver_cfg.coupling_type == "proxy":
            cls._validate_proxy_coupling(solver_cfg)
        else:
            cls._validate_admm_coupling(solver_cfg.admm_coupling)

    @classmethod
    def _validate_entries(cls, entries: list[CoupledSolverEntryCfg]) -> None:
        names: set[str] = set()
        for entry in entries:
            if not entry.name:
                raise ValueError("CoupledSolverEntryCfg.name must be non-empty.")
            if entry.name in names:
                raise ValueError(f"Duplicate CoupledSolverEntryCfg name {entry.name!r}.")
            names.add(entry.name)
            if entry.substeps < 1:
                raise ValueError(f"CoupledSolverEntryCfg {entry.name!r} substeps must be >= 1.")
            if entry.in_place and entry.substeps != 1:
                raise ValueError(f"CoupledSolverEntryCfg {entry.name!r} in_place requires substeps=1.")
        for field_name in ("bodies", "particles", "joints", "shapes"):
            cls._validate_unique_entry_ownership(entries, field_name)

    @staticmethod
    def _validate_unique_entry_ownership(entries: list[CoupledSolverEntryCfg], field_name: str) -> None:
        owners: dict[int, str] = {}
        for entry in entries:
            for raw_index in getattr(entry, field_name):
                index = int(raw_index)
                owner = owners.get(index)
                if owner is not None:
                    raise ValueError(
                        f"CoupledSolverEntryCfg {field_name} index {index} is owned by both "
                        f"{owner!r} and {entry.name!r}."
                    )
                owners[index] = entry.name

    @classmethod
    def _validate_proxy_coupling(cls, solver_cfg: CoupledSolverCfg) -> None:
        if len(solver_cfg.entries) > 2:
            raise ValueError("Newton proxy coupling currently supports at most two solver entries.")
        if not solver_cfg.proxy_coupling.proxies:
            raise ValueError("Newton proxy coupling requires at least one proxy mapping.")
        if solver_cfg.proxy_coupling.iterations < 1:
            raise ValueError("ProxyCouplingCfg.iterations must be >= 1.")
        entry_names = {entry.name for entry in solver_cfg.entries}
        for proxy in solver_cfg.proxy_coupling.proxies:
            if proxy.source not in entry_names:
                raise ValueError(f"CoupledProxyCfg source {proxy.source!r} does not match a coupled entry.")
            if proxy.destination not in entry_names:
                raise ValueError(f"CoupledProxyCfg destination {proxy.destination!r} does not match a coupled entry.")
            if proxy.source == proxy.destination:
                raise ValueError("CoupledProxyCfg source and destination must be different entries.")
            if not proxy.bodies and not proxy.particles:
                raise ValueError("CoupledProxyCfg must map at least one body or particle.")
            if proxy.proxy_bodies is not None and len(proxy.proxy_bodies) != len(proxy.bodies):
                raise ValueError("CoupledProxyCfg proxy_bodies must match bodies length.")
            if proxy.proxy_particles is not None and len(proxy.proxy_particles) != len(proxy.particles):
                raise ValueError("CoupledProxyCfg proxy_particles must match particles length.")
            if proxy.mass_scale <= 0.0:
                raise ValueError("CoupledProxyCfg mass_scale must be > 0.")
            if proxy.collide_interval is not None and proxy.collide_interval < 1:
                raise ValueError("CoupledProxyCfg collide_interval must be >= 1.")
            cls._build_proxy_mode(proxy.mode)

    @staticmethod
    def _validate_admm_coupling(admm_cfg: AdmmCouplingCfg) -> None:
        if admm_cfg.iterations < 1:
            raise ValueError("AdmmCouplingCfg.iterations must be >= 1.")
        if admm_cfg.rho <= 0.0:
            raise ValueError("AdmmCouplingCfg.rho must be > 0.")
        if admm_cfg.gamma < 0.0:
            raise ValueError("AdmmCouplingCfg.gamma must be >= 0.")
        if admm_cfg.auto_contact_distance is not None and admm_cfg.auto_contact_distance < 0.0:
            raise ValueError("AdmmCouplingCfg.auto_contact_distance must be >= 0.")
        if admm_cfg.auto_detection_margin is not None and admm_cfg.auto_detection_margin < 0.0:
            raise ValueError("AdmmCouplingCfg.auto_detection_margin must be >= 0.")
        for pair in admm_cfg.contact_pairs:
            if pair.source == pair.destination:
                raise ValueError("AdmmContactPairCfg source and destination must be different.")
            if pair.contact_distance is not None and pair.contact_distance < 0.0:
                raise ValueError("AdmmContactPairCfg.contact_distance must be >= 0.")
            if pair.detection_margin is not None and pair.detection_margin < 0.0:
                raise ValueError("AdmmContactPairCfg.detection_margin must be >= 0.")

    @classmethod
    def _needs_external_collision_pipeline(cls, solver_cfg: CoupledSolverCfg) -> bool:
        """Return whether the coupled solver should receive external contacts."""
        if solver_cfg.use_collision_pipeline is not None:
            return solver_cfg.use_collision_pipeline
        return any(solver_cfg_needs_external_contacts(entry.solver_cfg) for entry in solver_cfg.entries)
