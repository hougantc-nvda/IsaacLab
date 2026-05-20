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

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

from isaaclab_tasks.manager_based.manipulation.waterhose.launch import (
    get_visualizer_types,
    prepare_waterhose_launch,
)

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
    "--spacemouse_mode",
    choices=("auto", "simple", "full"),
    default="auto",
    help=(
        "SpaceMouse control mode. 'simple' maps cap translation to gripper XYZ and cap twist to yaw only; "
        "'full' keeps all 6-DoF axes. 'auto' uses simple mode for standalone Newton waterhose teleop."
    ),
)
parser.add_argument(
    "--spacemouse_pos_sensitivity",
    type=float,
    default=None,
    help="Override SpaceMouse translation sensitivity. Defaults to 0.05 * --sensitivity.",
)
parser.add_argument(
    "--spacemouse_rot_sensitivity",
    type=float,
    default=None,
    help=(
        "Override SpaceMouse rotation sensitivity. Defaults to 0.15 * --sensitivity in simple mode and "
        "0.05 * --sensitivity in full mode."
    ),
)
parser.add_argument(
    "--spacemouse_simple_x_sign",
    type=float,
    choices=(-1.0, 1.0),
    default=-1.0,
    help="Simple SpaceMouse sign for right EEF x translation.",
)
parser.add_argument(
    "--spacemouse_simple_y_sign",
    type=float,
    choices=(-1.0, 1.0),
    default=-1.0,
    help="Simple SpaceMouse sign for right EEF y translation.",
)
parser.add_argument(
    "--spacemouse_simple_z_sign",
    type=float,
    choices=(-1.0, 1.0),
    default=1.0,
    help="Simple SpaceMouse sign for right EEF z translation.",
)
parser.add_argument(
    "--spacemouse_simple_yaw_sign",
    type=float,
    choices=(-1.0, 1.0),
    default=-1.0,
    help="Simple SpaceMouse sign for right EEF yaw rotation.",
)
parser.add_argument(
    "--spacemouse_simple_deadzone",
    type=float,
    default=1.0e-3,
    help="Simple SpaceMouse deadzone applied to each command axis after scaling.",
)
parser.add_argument(
    "--spacemouse_simple_yaw_translation_lock",
    action=argparse.BooleanOptionalAction,
    default=False,
    help=(
        "Zero translation while simple SpaceMouse yaw is active. Disabled by default because the SpaceMouse driver "
        "stores the last rotation sample separately from translation samples."
    ),
)
parser.add_argument(
    "--debug_teleop",
    action="store_true",
    help="Print periodic teleop command diagnostics.",
)
parser.add_argument(
    "--cloudxr_env",
    type=str,
    default="cloudxrjs",
    help=(
        "Path to a CloudXR .env file, or a shorthand: 'cloudxrjs' (Quest/Pico, default) or 'avp' (Apple Vision Pro)."
        " Set to 'none' to disable CloudXR auto-launch entirely."
    ),
)
parser.add_argument(
    "--auto_launch_cloudxr",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Auto-launch the CloudXR runtime when --cloudxr_env is set. Use --no-auto_launch_cloudxr to disable.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


def _prepare_native_teleop_visualizers() -> None:
    """Ensure native Omniverse teleop devices have a Kit app window."""
    visualizer_types = get_visualizer_types(args_cli)
    uses_waterhose_task = args_cli.task is not None and "waterhose" in args_cli.task.lower()
    uses_newton_only_waterhose = uses_waterhose_task and "newton" in visualizer_types and "kit" not in visualizer_types
    if uses_newton_only_waterhose:
        return

    device_name = args_cli.teleop_device.lower() if args_cli.teleop_device is not None else "keyboard"
    if device_name not in {"keyboard", "spacemouse", "gamepad"}:
        return
    if device_name == "spacemouse":
        return
    if getattr(args_cli, "headless_explicit", False) and args_cli.headless:
        parser.error(
            "Native teleop devices require a Kit app window; remove --headless or use an XR/IsaacTeleop device."
        )

    if "none" in visualizer_types:
        parser.error("Native teleop devices require a Kit app window; use --visualizer kit or --visualizer kit,newton.")
    if "kit" not in visualizer_types:
        args_cli.visualizer = ["kit", *visualizer_types]
        args_cli.visualizer_explicit = True


_prepare_native_teleop_visualizers()
waterhose_launch = prepare_waterhose_launch(
    args_cli,
    parser=parser,
    default_standalone_spacemouse=True,
    require_standalone_spacemouse=True,
    standalone_spacemouse_error=(
        "Waterhose teleoperation with the Newton viewer requires --teleop_device spacemouse. "
        "Keyboard and gamepad teleoperation use Omniverse input and require --visualizer kit."
    ),
)
uses_standalone_waterhose_spacemouse = (
    waterhose_launch.uses_kitless_waterhose
    and args_cli.teleop_device is not None
    and args_cli.teleop_device.lower() == "spacemouse"
    and not args_cli.xr
)

app_launcher_args = vars(args_cli)

# launch omniverse app
if uses_standalone_waterhose_spacemouse:
    app_launcher = None
    simulation_app = None
else:
    app_launcher = AppLauncher(app_launcher_args)
    simulation_app = app_launcher.app

"""Rest everything follows."""


import logging

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import launch_simulation, parse_env_cfg

logger = logging.getLogger(__name__)

_CLOUDXR_ENV_SHORTHANDS: dict[str, str] = {}


def _resolve_cloudxr_env(value: str | None) -> str | None:
    """Resolve ``--cloudxr_env`` shorthands to absolute ``.env`` file paths.

    Accepts ``"cloudxrjs"`` (Quest/Pico), ``"avp"`` (Apple Vision Pro),
    ``"none"`` / ``None`` (disable), or an arbitrary file path.
    """
    if value is None or value.strip() == "" or value.lower() == "none":
        return None
    if not _CLOUDXR_ENV_SHORTHANDS:
        from isaaclab_teleop import CLOUDXR_AVP_ENV, CLOUDXR_JS_ENV

        _CLOUDXR_ENV_SHORTHANDS["cloudxrjs"] = CLOUDXR_JS_ENV
        _CLOUDXR_ENV_SHORTHANDS["avp"] = CLOUDXR_AVP_ENV
    return _CLOUDXR_ENV_SHORTHANDS.get(value.lower(), value)


class _SimpleSpaceMouse:
    """Restrict SpaceMouse commands to translation and yaw for easier teleoperation."""

    def __init__(
        self,
        device,
        translation_signs: tuple[float, float, float],
        yaw_sign: float,
        deadzone: float,
        yaw_translation_lock: bool,
    ):
        self._device = device
        self._translation_signs = translation_signs
        self._yaw_sign = yaw_sign
        self._deadzone = max(0.0, float(deadzone))
        self._yaw_translation_lock = yaw_translation_lock

    def __str__(self) -> str:
        return f"{self._device} (simple XYZ+gripper-spin mode)"

    def reset(self) -> None:
        self._device.reset()

    def add_callback(self, key: str, func: Callable[[], None]) -> None:
        self._device.add_callback(key, func)

    def advance(self):
        command = self._device.advance().clone()
        if self._deadzone > 0.0:
            command[torch.abs(command) < self._deadzone] = 0.0
        for axis, sign in enumerate(self._translation_signs):
            command[axis] *= sign
        command[3:5] = 0.0
        command[5] *= self._yaw_sign
        translation_norm = torch.linalg.vector_norm(command[:3])
        yaw_abs = torch.abs(command[5])
        if translation_norm > self._deadzone:
            command[5] = 0.0
        elif yaw_abs > self._deadzone:
            command[:3] = 0.0
        if self._yaw_translation_lock and abs(float(command[5].detach().cpu())) > self._deadzone:
            command[:3] = 0.0
        return command


def _create_builtin_device(device_name: str, sensitivity: float) -> object | None:
    """Create a built-in teleop device by name, or return None if unrecognized."""
    name = device_name.lower()
    if name == "keyboard":
        from isaaclab.devices.keyboard.se3_keyboard import Se3Keyboard
        from isaaclab.devices.keyboard.se3_keyboard_cfg import Se3KeyboardCfg

        return Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity))
    elif name == "spacemouse":
        from isaaclab.devices.spacemouse.se3_spacemouse import Se3SpaceMouse
        from isaaclab.devices.spacemouse.se3_spacemouse_cfg import Se3SpaceMouseCfg

        mode = args_cli.spacemouse_mode
        if mode == "auto":
            mode = "simple" if uses_standalone_waterhose_spacemouse else "full"
        pos_sensitivity = (
            float(args_cli.spacemouse_pos_sensitivity)
            if args_cli.spacemouse_pos_sensitivity is not None
            else 0.05 * sensitivity
        )
        rot_sensitivity = (
            float(args_cli.spacemouse_rot_sensitivity)
            if args_cli.spacemouse_rot_sensitivity is not None
            else (0.15 if mode == "simple" else 0.05) * sensitivity
        )
        device = Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=pos_sensitivity, rot_sensitivity=rot_sensitivity))
        if mode == "simple":
            return _SimpleSpaceMouse(
                device,
                (
                    float(args_cli.spacemouse_simple_x_sign),
                    float(args_cli.spacemouse_simple_y_sign),
                    float(args_cli.spacemouse_simple_z_sign),
                ),
                float(args_cli.spacemouse_simple_yaw_sign),
                float(args_cli.spacemouse_simple_deadzone),
                bool(args_cli.spacemouse_simple_yaw_translation_lock),
            )
        return device
    elif name == "gamepad":
        from isaaclab.devices.gamepad.se3_gamepad import Se3Gamepad
        from isaaclab.devices.gamepad.se3_gamepad_cfg import Se3GamepadCfg

        return Se3Gamepad(Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity))
    return None


def _close_simulation_app() -> None:
    if simulation_app is not None:
        simulation_app.close()


def _simulation_is_running(env) -> bool:
    if simulation_app is not None:
        return simulation_app.is_running()
    if env.sim.visualizers:
        return any(visualizer.is_running() and not visualizer.is_closed for visualizer in env.sim.visualizers)
    return True


def _create_teleop_interface(
    env_cfg,
    teleop_device_explicitly_set: bool,
    teleoperation_callbacks: dict[str, Callable[[], None]],
    use_isaac_teleop: bool,
):
    if use_isaac_teleop:
        from isaaclab_teleop import create_isaac_teleop_device, poll_control_events

        teleop_interface = create_isaac_teleop_device(
            env_cfg.isaac_teleop,
            sim_device=args_cli.device,
            callbacks=teleoperation_callbacks,
            cloudxr_env_file=_resolve_cloudxr_env(args_cli.cloudxr_env),
            auto_launch_cloudxr=args_cli.auto_launch_cloudxr,
        )
        return teleop_interface, poll_control_events

    if teleop_device_explicitly_set:
        device_name = args_cli.teleop_device
        if hasattr(env_cfg, "teleop_devices") and device_name in env_cfg.teleop_devices.devices:
            from isaaclab.devices.teleop_device_factory import create_teleop_device

            return create_teleop_device(device_name, env_cfg.teleop_devices.devices, teleoperation_callbacks), None
        teleop_interface = _create_builtin_device(device_name, args_cli.sensitivity)
        if teleop_interface is None:
            raise ValueError(
                f"--teleop_device={device_name} was passed but no matching entry exists in env_cfg.teleop_devices "
                "and it is not a built-in device name. Either remove --teleop_device to use the IsaacTeleop "
                "pipeline, or add an entry under teleop_devices in the environment config. Built-in devices: "
                "keyboard, spacemouse, gamepad."
            )
    else:
        teleop_interface = _create_builtin_device("keyboard", args_cli.sensitivity)

    for key, callback in teleoperation_callbacks.items():
        try:
            teleop_interface.add_callback(key, callback)
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to add callback for key {key}: {e}")

    return teleop_interface, None


def main() -> None:
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
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

    if use_isaac_teleop or args_cli.xr:
        from isaaclab.devices.openxr import remove_camera_configs

        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    launch_context = None
    if simulation_app is None:
        launch_context = launch_simulation(env_cfg, args_cli)
        launch_context.__enter__()

    def close_launch_context() -> None:
        if launch_context is not None:
            launch_context.__exit__(None, None, None)

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        close_launch_context()
        _close_simulation_app()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # For XR devices (hand tracking or IsaacTeleop), default to inactive
    if use_isaac_teleop or args_cli.xr:
        teleoperation_active = env_cfg.isaac_teleop.teleoperation_active_default if use_isaac_teleop else False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device based on configuration
    teleop_interface = None
    poll_control_events = None

    try:
        teleop_interface, poll_control_events = _create_teleop_interface(
            env_cfg,
            teleop_device_explicitly_set,
            teleoperation_callbacks,
            use_isaac_teleop,
        )
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        close_launch_context()
        _close_simulation_app()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        close_launch_context()
        _close_simulation_app()
        return

    print(f"Using teleop device: {teleop_interface}")

    def run_loop():
        """Inner function to run the teleop loop with access to nonlocal variables."""
        nonlocal should_reset_recording_instance, teleoperation_active

        # reset environment
        env.reset()
        teleop_interface.reset()

        stack_name = "IsaacTeleop" if use_isaac_teleop else "native"
        print(f"{stack_name} teleoperation started. Press 'R' to reset the environment.")

        step_count = 0

        # simulate environment
        while _simulation_is_running(env):
            try:
                # run everything in inference mode
                with torch.inference_mode():
                    # get device command
                    action = teleop_interface.advance()
                    step_count += 1

                    if use_isaac_teleop:
                        ctrl = poll_control_events(teleop_interface)
                        if ctrl.is_active is not None:
                            teleoperation_active = ctrl.is_active
                        if ctrl.should_reset:
                            should_reset_recording_instance = True

                    # action is None when IsaacTeleop session hasn't started yet
                    # (e.g. waiting for user to click "Start AR")
                    if action is None:
                        env.sim.render()
                    elif teleoperation_active:
                        if args_cli.debug_teleop and step_count % 15 == 0:
                            action_cpu = action.detach().cpu()
                            print(
                                "teleop action:",
                                " ".join(f"{value:+.3f}" for value in action_cpu.tolist()),
                                flush=True,
                            )
                        # process actions
                        actions = action.repeat(env.num_envs, 1)
                        # apply actions
                        env.step(actions)
                    else:
                        env.sim.render()

                    if should_reset_recording_instance:
                        env.reset()
                        teleop_interface.reset()
                        should_reset_recording_instance = False
                        print("Environment reset complete")
            except Exception as e:
                logger.error(f"Error during simulation step: {e}")
                break

    # Run the teleoperation loop
    # IsaacTeleop requires a context manager, native devices don't
    if use_isaac_teleop:
        with teleop_interface:
            run_loop()
    else:
        run_loop()

    # close the simulator
    env.close()
    close_launch_context()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # env.close() already closes the USD stage via sim.clear_instance().
    # Pump the event loop so the viewport processes closure, then close the app.
    if simulation_app is not None:
        simulation_app.update()
        simulation_app.close()
