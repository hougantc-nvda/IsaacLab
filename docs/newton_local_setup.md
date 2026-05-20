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

The coupling manager uses Newton PR 2848 directly. Isaac Lab imports coupled solver classes from `newton.solvers.coupled_experimental`, and `source/isaaclab_newton/setup.py` pins `newton[sim]` to the verified PR commit above.

There is no vendored coupled solver fallback in this worktree. If PR 2848 changes, update the Newton pin and the local `/home/maximiliank/Work/newton` checkout together.

## Worktree Rule

Do not push to upstream IsaacLab. The active writable target for this task is the fork branch in `/home/maximiliank/Work/IsaacLab-coupling`.
