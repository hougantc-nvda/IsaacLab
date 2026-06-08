# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers).

This script supports two teleoperation stacks:
1. Native Isaac Lab teleop stack (via teleop_devices in env_cfg)
2. IsaacTeleop-based stack (via isaac_teleop in env_cfg)

The script automatically detects which stack to use based on the environment config.
"""

import argparse
import contextlib
import logging
import sys
from collections.abc import Callable

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.preset_cli import fold_preset_tokens, setup_preset_cli

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default=None,
    help=(
        "Legacy teleop device name. When omitted, the IsaacTeleop pipeline is used if configured in the env,"
        " otherwise keyboard is used as fallback. When explicitly provided, the script uses the legacy"
        " teleop_devices path and looks up this name in env_cfg.teleop_devices.devices."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help=(
        "Path to a CloudXR .env file, or a shorthand: 'cloudxrjs' (Quest/Pico, default), 'avp' "
        "(Apple Vision Pro), or 'newton' (Newton OpenXR CloudXR). Set to 'none' to disable CloudXR "
        "auto-launch entirely."
    ),
)
parser.add_argument(
    "--auto_launch_cloudxr",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Auto-launch the CloudXR runtime when --cloudxr_env is set. Use --no-auto_launch_cloudxr to disable.",
)
# append launcher cli args and preserve Hydra/preset overrides
add_launcher_args(parser)
_ORIGINAL_ARGV = sys.argv[:]
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


import gymnasium as gym
import torch
from isaaclab_teleop.openxr_runtime import KitlessTeleopLauncher
from isaaclab_teleop.xr_cfg import remove_camera_configs

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

logger = logging.getLogger(__name__)
_KITLESS_TELEOP_LAUNCHER = KitlessTeleopLauncher(args_cli, _ORIGINAL_ARGV, logger=logger)


def _create_builtin_device(device_name: str, sensitivity: float) -> object | None:
    """Create a built-in teleop device by name, or return None if unrecognized."""
    name = device_name.lower()
    if name == "keyboard":
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

        return Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity))
    elif name == "spacemouse":
        from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg

        return Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity))
    elif name == "gamepad":
        from isaaclab.devices import Se3Gamepad, Se3GamepadCfg

        return Se3Gamepad(Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity))
    return None


def main() -> None:
    """Run teleoperation with an Isaac Lab manipulation environment."""
    env_cfg, _ = _KITLESS_TELEOP_LAUNCHER.resolve_env_cfg(args_cli.task, resolve_task_config, sys.argv)
    env_cfg.env_name = args_cli.task
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )

    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        from isaaclab_tasks.manager_based.manipulation.lift import mdp

        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    # When --teleop_device is explicitly provided, use the legacy teleop_devices path
    # even if isaac_teleop is configured. Otherwise prefer isaac_teleop when available.
    teleop_device_explicitly_set = args_cli.teleop_device is not None
    use_isaac_teleop = (
        not teleop_device_explicitly_set and hasattr(env_cfg, "isaac_teleop") and env_cfg.isaac_teleop is not None
    )

    if use_isaac_teleop:
        env_cfg.isaac_teleop.sim_device = env_cfg.sim.device

    _KITLESS_TELEOP_LAUNCHER.configure_env_cfg(env_cfg, sys.argv)
    kitless_debug_idle = _KITLESS_TELEOP_LAUNCHER.debug_idle_for_env_cfg(env_cfg)
    if kitless_debug_idle:
        print(
            f"Kitless {_KITLESS_TELEOP_LAUNCHER.display_name} idle-action debug mode enabled; "
            "CloudXR/OpenXR client input is bypassed."
        )
    if use_isaac_teleop or args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    launcher_args = _KITLESS_TELEOP_LAUNCHER.simulation_launcher_args()
    cloudxr_env_path = _KITLESS_TELEOP_LAUNCHER.resolve_cloudxr_env(args_cli.cloudxr_env)
    cloudxr_launcher = None
    try:
        with launch_simulation(env_cfg, launcher_args):
            try:
                env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
                # check environment name (for reach , we don't allow the gripper)
                if "Reach" in args_cli.task:
                    logger.warning(
                        f"The environment '{args_cli.task}' does not support gripper control. The device command "
                        "will be ignored."
                    )
            except Exception:
                logger.exception("Failed to create environment.")
                return

            try:
                cloudxr_launcher = _KITLESS_TELEOP_LAUNCHER.launch_cloudxr(
                    use_isaac_teleop=use_isaac_teleop,
                    cloudxr_env_path=cloudxr_env_path,
                    auto_launch=args_cli.auto_launch_cloudxr,
                )
                renderer_openxr_session = _KITLESS_TELEOP_LAUNCHER.configure_openxr_teleop(
                    env,
                    env_cfg,
                    enabled=use_isaac_teleop and _KITLESS_TELEOP_LAUNCHER.enabled and not kitless_debug_idle,
                )
                _run_teleoperation(
                    env,
                    env_cfg,
                    use_isaac_teleop,
                    teleop_device_explicitly_set,
                    renderer_openxr_session,
                    _KITLESS_TELEOP_LAUNCHER,
                    cloudxr_env_path,
                    cloudxr_launcher,
                )
            finally:
                env.close()
                print("Environment closed")
    finally:
        _KITLESS_TELEOP_LAUNCHER.stop_cloudxr(cloudxr_launcher)


def _run_teleoperation(  # noqa: C901
    env,
    env_cfg,
    use_isaac_teleop: bool,
    teleop_device_explicitly_set: bool,
    renderer_openxr_session: object | None,
    kitless_teleop_launcher: KitlessTeleopLauncher,
    cloudxr_env_path: str | None,
    cloudxr_launcher: object | None,
) -> None:
    """Create the teleop device and run the simulation loop."""
    should_reset_recording_instance = False
    teleoperation_active = True
    runtime_name = kitless_teleop_launcher.display_name
    runtime_physics_active = kitless_teleop_launcher.env_uses_runtime_physics(env)
    kitless_debug_idle = kitless_teleop_launcher.debug_idle_for_env(env)
    kitless_debug_step_limit = kitless_teleop_launcher.debug_step_limit()

    def reset_recording_instance() -> None:
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    if use_isaac_teleop or args_cli.xr:
        teleoperation_active = env_cfg.isaac_teleop.teleoperation_active_default if use_isaac_teleop else False
    else:
        teleoperation_active = True
    if kitless_debug_idle:
        teleoperation_active = True
        print(f"Kitless {runtime_name} idle-action debug mode auto-started teleoperation.")

    teleop_interface = None

    try:
        if use_isaac_teleop and kitless_debug_idle:
            debug_device_factory = getattr(env_cfg, "create_kitless_debug_idle_device", None)
            if not callable(debug_device_factory):
                logger.error("Kitless debug idle mode requires env_cfg.create_kitless_debug_idle_device().")
                return
            teleop_interface = debug_device_factory(env, args_cli.device)

        elif use_isaac_teleop:
            from isaaclab_teleop import create_isaac_teleop_device, poll_control_events

            teleop_interface = create_isaac_teleop_device(
                env_cfg.isaac_teleop,
                sim_device=args_cli.device,
                callbacks=teleoperation_callbacks,
                cloudxr_env_file=None if cloudxr_launcher is not None else cloudxr_env_path,
                auto_launch_cloudxr=args_cli.auto_launch_cloudxr and cloudxr_launcher is None,
            )

        elif teleop_device_explicitly_set:
            device_name = args_cli.teleop_device
            if hasattr(env_cfg, "teleop_devices") and device_name in env_cfg.teleop_devices.devices:
                from isaaclab.devices.teleop_device_factory import create_teleop_device

                teleop_interface = create_teleop_device(
                    device_name, env_cfg.teleop_devices.devices, teleoperation_callbacks
                )
            else:
                teleop_interface = _create_builtin_device(device_name, args_cli.sensitivity)
                if teleop_interface is None:
                    logger.error(
                        "--teleop_device=%s was passed but no matching entry exists in env_cfg.teleop_devices and "
                        "it is not a built-in device name. Either remove --teleop_device to use the IsaacTeleop "
                        "pipeline, or add a '%s' entry under teleop_devices in the environment config. Built-in "
                        "devices: keyboard, spacemouse, gamepad.",
                        device_name,
                        device_name,
                    )
                    return
                for key, callback in teleoperation_callbacks.items():
                    try:
                        teleop_interface.add_callback(key, callback)
                    except (ValueError, TypeError) as e:
                        logger.warning("Failed to add callback for key %s: %s", key, e)
        else:
            # No --teleop_device and no isaac_teleop: fall back to keyboard
            sensitivity = args_cli.sensitivity
            from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

            teleop_interface = Se3Keyboard(
                Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
            )
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning("Failed to add callback for key %s: %s", key, e)
    except Exception:
        logger.exception("Failed to create teleop device.")
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        return

    print(f"Using teleop device: {teleop_interface}")

    if use_isaac_teleop:
        kitless_teleop_launcher.attach_anchor_provider(teleop_interface, renderer_openxr_session)
        runtime_hooks = getattr(env_cfg, "install_kitless_teleop_runtime_hooks", None)
        if runtime_physics_active and callable(runtime_hooks):
            runtime_hooks(env)

    action_conditioner = getattr(env_cfg, "condition_kitless_teleop_action", None)
    action_issue_checker = getattr(env_cfg, "teleop_action_target_issue", None)
    action_formatter = getattr(env_cfg, "format_kitless_teleop_action", None)
    diagnostics_logger = getattr(env_cfg, "log_kitless_teleop_diagnostics", None)
    skipped_action_log_count = 0
    kitless_action_state: dict[str, torch.Tensor] = {}

    def run_loop():
        """Inner function to run the teleop loop with access to nonlocal variables."""
        nonlocal should_reset_recording_instance, skipped_action_log_count, teleoperation_active

        env.reset()
        teleop_interface.reset()
        kitless_action_state.clear()
        if use_isaac_teleop and runtime_physics_active and kitless_debug_idle:
            if callable(diagnostics_logger):
                diagnostics_logger(env, "GR1T2 body transforms after reset")

        stack_name = "IsaacTeleop" if use_isaac_teleop else "native"
        print(f"{stack_name} teleoperation started. Press 'R' to reset the environment.")
        debug_step_count = 0

        while kitless_teleop_launcher.is_loop_running(env):
            try:
                with torch.inference_mode():
                    action = teleop_interface.advance()

                    if use_isaac_teleop and not kitless_debug_idle:
                        ctrl = poll_control_events(teleop_interface)
                        if ctrl.is_active is not None:
                            teleoperation_active = ctrl.is_active
                        if ctrl.should_reset:
                            should_reset_recording_instance = True

                    if (
                        action is not None
                        and use_isaac_teleop
                        and teleoperation_active
                        and runtime_physics_active
                        and not kitless_debug_idle
                        and callable(action_conditioner)
                    ):
                        action = action_conditioner(env, action, kitless_action_state)
                    elif action is None or not teleoperation_active:
                        kitless_action_state.clear()

                    action_target_issue = (
                        action_issue_checker(action)
                        if callable(action_issue_checker)
                        and use_isaac_teleop
                        and runtime_physics_active
                        and action is not None
                        else None
                    )

                    # action is None when IsaacTeleop session hasn't started yet
                    # (e.g. waiting for renderer-owned OpenXR handles)
                    if action is None:
                        env.sim.render()
                    elif teleoperation_active and action_target_issue is None:
                        actions = action.repeat(env.num_envs, 1)
                        env.step(actions)
                        if kitless_debug_idle:
                            debug_step_count += 1
                        if kitless_debug_idle and debug_step_count > 0 and debug_step_count % 60 == 0:
                            if callable(diagnostics_logger):
                                diagnostics_logger(env, f"GR1T2 body transforms after active step {debug_step_count}")
                        if (
                            kitless_debug_idle
                            and kitless_debug_step_limit is not None
                            and debug_step_count >= kitless_debug_step_limit
                        ):
                            print(
                                f"Kitless {runtime_name} idle-action debug completed after "
                                f"{debug_step_count} simulated steps."
                            )
                            break
                    elif teleoperation_active:
                        if skipped_action_log_count < 20:
                            action_summary = (
                                action_formatter(action)
                                if callable(action_formatter)
                                else f"action_shape={tuple(action.shape)}"
                            )
                            print(f"Skipping unsafe IsaacTeleop action: {action_target_issue}; {action_summary}")
                            skipped_action_log_count += 1
                        env.sim.render()
                    else:
                        env.sim.render()

                    if should_reset_recording_instance:
                        env.reset()
                        teleop_interface.reset()
                        kitless_action_state.clear()
                        if use_isaac_teleop and runtime_physics_active and kitless_debug_idle:
                            if callable(diagnostics_logger):
                                diagnostics_logger(env, "GR1T2 body transforms after reset")
                        should_reset_recording_instance = False
                        print("Environment reset complete")
            except Exception as e:
                logger.error("Error during simulation step: %s", e)
                break

    if use_isaac_teleop:
        with teleop_interface:
            run_loop()
    else:
        run_loop()


if __name__ == "__main__":
    main()
