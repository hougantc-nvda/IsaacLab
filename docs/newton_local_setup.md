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

The coupling manager no longer requires installing Newton PR 2848. Isaac Lab imports coupled solver classes through `isaaclab_newton.physics.coupled_solvers`:

- If normal Newton provides `newton.solvers.coupled_experimental`, Isaac Lab uses the upstream Newton implementation.
- Otherwise Isaac Lab falls back to the vendored compatibility copy in `isaaclab_newton.physics._coupled_solvers`.

The fallback currently carries the Newton PR 2848 coupled solver package plus a small implicit-MPM proxy-body hook shim. The package dependency is back on the normal Newton `v1.2.0` pin used by the other Newton extras.

When PR 2848 lands in the Newton release Isaac Lab targets, remove `isaaclab_newton.physics._coupled_solvers` and keep `isaaclab_newton.physics.coupled_solvers` as the single import location, or inline its upstream imports if no fallback is needed.

## Worktree Rule

Do not push to upstream IsaacLab. The active writable target for this task is the fork branch in `/home/maximiliank/Work/IsaacLab-coupling`.
