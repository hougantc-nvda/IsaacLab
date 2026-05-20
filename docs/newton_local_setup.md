# Isaac Lab Newton Local Setup

Last verified: 2026-05-20.

This workspace uses sibling checkouts under `/home/maximiliank/Work`.

## Active Branches

- Isaac Lab feature reference: `/home/maximiliank/Work/IsaacLab` on `feat/newton-implicit-mpm`.
- Isaac Lab MPM base: `/home/maximiliank/Work/IsaacLab-mpm` on `max/newton-mpm-manager`.
- Isaac Lab coupling worktree: `/home/maximiliank/Work/IsaacLab-coupling` on `max/newton-coupling-manager`.
- Newton PR 2848 reference: `/home/maximiliank/Work/newton` on `pr-2848-coupled-solver-framework-fresh`.

The verified Newton PR 2848 commit is:

```text
e9851d3e11ad35e879e818c789570eb4fa5b0264
```

`origin/pr-2848-head` was refreshed directly from `refs/pull/2848/head` and points at that same commit.

## Runtime Wiring

Install the local Newton checkout editable into the Isaac Lab Python environment before running coupled solver demos or tasks. This worktree currently does not have its own `_isaac_sim/` or `env_isaaclab/`, so validation used the existing Isaac Lab environment from the sibling `IsaacLab-ik` checkout:

```bash
/home/maximiliank/Work/IsaacLab-ik/env_isaaclab/bin/python -m pip install -e /home/maximiliank/Work/newton
```

The coupling manager expects Newton PR 2848 APIs, especially:

- `newton.solvers.coupled_experimental.SolverCoupled`
- `newton.solvers.coupled_experimental.SolverProxyCoupled`
- `newton.solvers.coupled_experimental.SolverAdmmCoupled`
- `SolverCoupled.Entry(..., configure_view=..., substeps=..., in_place=...)`
- `SolverProxyCoupled.Proxy(..., bodies=..., particles=..., collision_pipeline=..., collide_interval=...)`
- `SolverAdmmCoupled.ContactPair(...)`
- `SolverAdmmCoupled.add_body_particle_attachment(...)`

## Worktree Rule

Do not push to upstream IsaacLab. The active writable target for this task is the fork branch in `/home/maximiliank/Work/IsaacLab-coupling`.
