# Waterhose Demo

## Assets

The waterhose demo expects one unpacked asset bundle at:

```bash
source/isaaclab_assets/data/WaterhoseDemo
```

Create or restore it from the local tarball:

```bash
mkdir -p source/isaaclab_assets/data
tar -xzf waterhose_demo_assets.tar.gz -C source/isaaclab_assets/data
```

To keep the bundle outside the repo, set:

```bash
export ISAACLAB_WATERHOSE_ASSET_ROOT=/absolute/path/to/WaterhoseDemo
```

The bundle contains the authored `Waterhose/Cable008` USD scene, cable curves,
plug attachments, textures, and the `RBY1DF` URDF meshes. These assets are
intentionally ignored by git.

## Run

From the repo root, run the Newton viewer teleop demo:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-RBY1DF-IK-Rel-Play-v0 \
  --visualizer newton \
  --teleop_device spacemouse
```

For Kit plus Newton visualization:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Waterhose-RBY1DF-IK-Rel-Play-v0 \
  --visualizer kit,newton \
  --teleop_device spacemouse
```

The task does not set `DISPLAY`; configure it in the shell when your machine
requires one.
