# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Script to record demonstrations with Isaac Lab environments using human teleoperation.

This script allows users to record demonstrations operated by human teleoperation for a specified task.
The recorded demonstrations are stored as episodes in a hdf5 file. Users can specify the task, teleoperation
device, dataset directory, and environment stepping rate through command-line arguments.

This script supports two teleoperation stacks:
1. Native Isaac Lab teleop stack (via teleop_devices in env_cfg)
2. IsaacTeleop-based stack (via isaac_teleop in env_cfg)

The script automatically detects which stack to use based on the environment config.

required arguments:
    --task                    Name of the task.

optional arguments:
    -h, --help                Show this help message and exit
    --teleop_device           Legacy teleop device name. When omitted, IsaacTeleop is used if
                              configured, otherwise keyboard. When set, forces the legacy path.
    --dataset_file            File path to export recorded demos. (default: "./datasets/dataset.hdf5")
    --step_hz                 Environment stepping rate in Hz. (default: 30)
    --num_demos               Number of demonstrations to record. (default: 0)
    --num_success_steps       Number of continuous steps with task success for concluding a demo as successful.
                              (default: 10)
"""

# Standard library imports
import argparse
import contextlib
import logging
import os
import sys
import time
from collections.abc import Callable

from isaaclab_teleop.openxr_runtime import KitlessTeleopLauncher

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import resolve_task_config
from isaaclab_tasks.utils.preset_cli import fold_preset_tokens, setup_preset_cli

# add argparse arguments
parser = argparse.ArgumentParser(description="Record demonstrations for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--step_hz", type=int, default=30, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help="Number of continuous steps with task success for concluding a demo as successful. Default is 10.",
)
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
parser.add_argument(
    "--mcap_record_path",
    type=str,
    default=None,
    help=(
        "Debug-only: write the live IsaacTeleop session to this MCAP file (one continuous file for the whole run)."
        " Intended for pairing with teleop_replay_agent.py in CI -- NOT a data-generation format. MCAPs produced"
        " here lack per-episode segmentation, world-frame anchor state, env reset state, and have no public Python"
        " decoder. For data-gen workflows use the HDF5 dataset path (default). Ignored when the IsaacTeleop stack"
        " is not in use."
    ),
)

# append launcher cli args and preserve Hydra/preset overrides
add_launcher_args(parser)
_ORIGINAL_ARGV = sys.argv[:]
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

# Validate required arguments
if args_cli.task is None:
    parser.error("--task is required")

logger = logging.getLogger(__name__)
_KITLESS_TELEOP_LAUNCHER = KitlessTeleopLauncher(args_cli, _ORIGINAL_ARGV, logger=logger)


# Third-party imports
import gymnasium as gym
import torch
from isaaclab_teleop.xr_cfg import remove_camera_configs

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import isaaclab_mimic.envs  # noqa: F401
from isaaclab_mimic.ui.instruction_display import InstructionDisplay, show_subtask_instructions


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        """Initialize a RateLimiter with specified frequency.

        Args:
            hz: Frequency to enforce in Hertz.
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.033, self.sleep_duration)

    def sleep(self, env: gym.Env):
        """Attempt to sleep at the specified rate in hz.

        Args:
            env: Environment to render during sleep periods.
        """
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def setup_output_directories() -> tuple[str, str]:
    """Set up output directories for saving demonstrations.

    Creates the output directory if it doesn't exist and extracts the file name
    from the dataset file path.

    Returns:
        tuple[str, str]: A tuple containing:
            - output_dir: The directory path where the dataset will be saved
            - output_file_name: The filename (without extension) for the dataset
    """
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    return output_dir, output_file_name


def create_environment_config(
    output_dir: str, output_file_name: str
) -> tuple[ManagerBasedRLEnvCfg | DirectRLEnvCfg, object | None, bool]:
    """Create and configure the environment configuration.

    Parses the environment configuration and makes necessary adjustments for demo recording.
    Extracts the success termination function and configures the recorder manager.

    Args:
        output_dir: Directory where recorded demonstrations will be saved
        output_file_name: Name of the file to store the demonstrations

    Returns:
        tuple[isaaclab_tasks.utils.parse_cfg.EnvCfg, Optional[object], bool]: A tuple containing:
            - env_cfg: The configured environment configuration
            - success_term: The success termination object or None if not available
            - use_isaac_teleop: Whether IsaacTeleop stack should be used

    Raises:
        Exception: If parsing the environment configuration fails
    """
    # parse configuration
    try:
        env_cfg, _ = _KITLESS_TELEOP_LAUNCHER.resolve_env_cfg(args_cli.task, resolve_task_config, sys.argv)
        env_cfg.env_name = args_cli.task.split(":")[-1]
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if hasattr(env_cfg, "scene"):
            env_cfg.scene.num_envs = 1
    except Exception as e:
        logger.error(f"Failed to parse environment configuration: {e}")
        exit(1)

    # When --teleop_device is explicitly provided, use the legacy teleop_devices path
    # even if isaac_teleop is configured. Otherwise prefer isaac_teleop when available.
    teleop_device_explicitly_set = args_cli.teleop_device is not None
    use_isaac_teleop = (
        not teleop_device_explicitly_set and hasattr(env_cfg, "isaac_teleop") and env_cfg.isaac_teleop is not None
    )

    if use_isaac_teleop:
        env_cfg.isaac_teleop.sim_device = env_cfg.sim.device

    _KITLESS_TELEOP_LAUNCHER.configure_env_cfg(env_cfg, sys.argv)

    # extract success checking function to invoke in the main loop
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success
        env_cfg.terminations.success = None
    else:
        logger.warning(
            "No success termination term was found in the environment."
            " Will not be able to mark recorded demos as successful."
        )

    if use_isaac_teleop or args_cli.xr:
        # If cameras are not enabled and XR is enabled, remove camera configs
        if not args_cli.enable_cameras:
            env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    # modify configuration such that the environment runs indefinitely until
    # the goal is reached or other termination conditions are met
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False

    env_cfg.recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return env_cfg, success_term, use_isaac_teleop


def create_environment(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> gym.Env:
    """Create the environment from the configuration.

    Args:
        env_cfg: The environment configuration object that defines the environment properties.
            This should be an instance of EnvCfg created by parse_env_cfg().

    Returns:
        gym.Env: A Gymnasium environment instance for the specified task.

    Raises:
        Exception: If environment creation fails for any reason.
    """
    try:
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        return env
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        exit(1)


def _create_builtin_device(device_name: str) -> object | None:
    """Create a built-in teleop device by name, or return None if unrecognized."""
    name = device_name.lower()
    if name == "keyboard":
        return Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    elif name == "spacemouse":
        return Se3SpaceMouse(Se3SpaceMouseCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
    return None


def setup_teleop_device(
    callbacks: dict[str, Callable],
    use_isaac_teleop: bool = False,
    renderer_openxr_session: object | None = None,
    cloudxr_env_path: str | None = None,
    cloudxr_launcher: object | None = None,
) -> object:
    """Set up the teleoperation device based on configuration.

    Attempts to create a teleoperation device based on the environment configuration.
    Falls back to default devices if the specified device is not found in the configuration.

    Args:
        callbacks: Dictionary mapping callback keys to functions that will be
                   attached to the teleop device
        use_isaac_teleop: Whether to use IsaacTeleop stack instead of native stack

    Returns:
        object: The configured teleoperation device interface

    Raises:
        Exception: If teleop device creation fails
    """
    teleop_device_explicitly_set = args_cli.teleop_device is not None
    teleop_interface = None
    try:
        if use_isaac_teleop:
            from isaaclab_teleop import create_isaac_teleop_device

            teleop_interface = create_isaac_teleop_device(
                env_cfg.isaac_teleop,
                sim_device=args_cli.device,
                callbacks=callbacks,
                cloudxr_env_file=None if cloudxr_launcher is not None else cloudxr_env_path,
                auto_launch_cloudxr=args_cli.auto_launch_cloudxr and cloudxr_launcher is None,
                mcap_record_path=args_cli.mcap_record_path,
            )
            _KITLESS_TELEOP_LAUNCHER.attach_anchor_provider(teleop_interface, renderer_openxr_session)
            if args_cli.mcap_record_path is not None:
                logger.info("Recording live IsaacTeleop session to MCAP (debug-only): %s", args_cli.mcap_record_path)

        elif teleop_device_explicitly_set:
            device_name = args_cli.teleop_device
            if hasattr(env_cfg, "teleop_devices") and device_name in env_cfg.teleop_devices.devices:
                teleop_interface = create_teleop_device(device_name, env_cfg.teleop_devices.devices, callbacks)
            else:
                teleop_interface = _create_builtin_device(device_name)
                if teleop_interface is None:
                    logger.error(
                        f"--teleop_device={device_name} was passed but no matching entry exists in"
                        " env_cfg.teleop_devices and it is not a built-in device name. Either remove"
                        " --teleop_device to use the IsaacTeleop pipeline, or add a"
                        f" '{device_name}' entry under teleop_devices in the environment config."
                        " Built-in devices: keyboard, spacemouse."
                    )
                    exit(1)
                for key, callback in callbacks.items():
                    teleop_interface.add_callback(key, callback)
        else:
            # No --teleop_device and no isaac_teleop: fall back to keyboard
            teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.2, rot_sensitivity=0.5))
            for key, callback in callbacks.items():
                teleop_interface.add_callback(key, callback)
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        exit(1)

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        exit(1)

    return teleop_interface


class _NullInstructionDisplay:
    """Instruction-display stub for kitless renderers without Kit UI widgets."""

    def set_labels(self, subtask_label, demo_label) -> None:
        """Accept non-XR labels for API compatibility."""

    def show_subtask(self, text: str) -> None:
        """Skip subtask UI text for kitless renderers."""

    def show_demo(self, text: str) -> None:
        """Print demo status for kitless renderers."""
        print(text)


def setup_ui(label_text: str, env: gym.Env) -> object:
    """Set up the user interface elements.

    Creates instruction display and UI window with labels for showing information
    to the user during demonstration recording.

    Args:
        label_text: Text to display showing current recording status
        env: The environment instance for which UI is being created

    Returns:
        InstructionDisplay: The configured instruction display object
    """
    if _KITLESS_TELEOP_LAUNCHER.enabled:
        return _NullInstructionDisplay()

    instruction_display = InstructionDisplay(args_cli.xr)
    if not args_cli.xr:
        import omni.ui as ui

        from isaaclab.envs.ui import EmptyWindow

        window = EmptyWindow(env, "Instruction")
        with window.ui_window_elements["main_vstack"]:
            demo_label = ui.Label(label_text)
            subtask_label = ui.Label("")
            instruction_display.set_labels(subtask_label, demo_label)

    return instruction_display


def process_success_condition(env: gym.Env, success_term: object | None, success_step_count: int) -> tuple[int, bool]:
    """Process the success condition for the current step.

    Checks if the environment has met the success condition for the required
    number of consecutive steps. Marks the episode as successful if criteria are met.

    Args:
        env: The environment instance to check
        success_term: The success termination object or None if not available
        success_step_count: Current count of consecutive successful steps

    Returns:
        tuple[int, bool]: A tuple containing:
            - updated success_step_count: The updated count of consecutive successful steps
            - success_reset_needed: Boolean indicating if reset is needed due to success
    """
    if success_term is None:
        return success_step_count, False

    if bool(success_term.func(env, **success_term.params)[0]):
        success_step_count += 1
        if success_step_count >= args_cli.num_success_steps:
            env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
            env.recorder_manager.set_success_to_episodes(
                [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([0])
            print("Success condition met! Recording completed.")
            return success_step_count, True
    else:
        success_step_count = 0

    return success_step_count, False


def handle_reset(
    env: gym.Env,
    success_step_count: int,
    instruction_display: InstructionDisplay,
    label_text: str,
    teleop_interface: object | None = None,
) -> int:
    """Handle resetting the environment.

    Resets the environment, recorder manager, teleop device, and related
    state variables.  Updates the instruction display with current status.

    Args:
        env: The environment instance to reset.
        success_step_count: Current count of consecutive successful steps.
        instruction_display: The display object to update.
        label_text: Text to display showing current recording status.
        teleop_interface: Optional teleop device to reset (resets XR anchor
            and retargeter cross-step state).

    Returns:
        Reset success step count (0).
    """
    print("Resetting environment...")
    env.sim.reset()
    env.recorder_manager.reset()
    env.reset()
    if teleop_interface is not None and hasattr(teleop_interface, "reset"):
        teleop_interface.reset()
    success_step_count = 0
    instruction_display.show_demo(label_text)
    return success_step_count


def run_simulation_loop(
    env: gym.Env,
    teleop_interface: object | None,
    success_term: object | None,
    rate_limiter: RateLimiter | None,
    use_isaac_teleop: bool = False,
    renderer_openxr_session: object | None = None,
    cloudxr_env_path: str | None = None,
    cloudxr_launcher: object | None = None,
) -> int:
    """Run the main simulation loop for collecting demonstrations.

    Sets up callback functions for teleop device, initializes the UI,
    and runs the main loop that processes user inputs and environment steps.
    Records demonstrations when success conditions are met.

    Args:
        env: The environment instance
        teleop_interface: Optional teleop interface (will be created if None)
        success_term: The success termination object or None if not available
        rate_limiter: Optional rate limiter to control simulation speed
        use_isaac_teleop: Whether to use IsaacTeleop stack

    Returns:
        int: Number of successful demonstrations recorded
    """
    current_recorded_demo_count = 0
    success_step_count = 0
    should_reset_recording_instance = False
    # For IsaacTeleop or XR, default to inactive until START is triggered
    running_recording_instance = not (args_cli.xr or use_isaac_teleop)

    # Callback closures for the teleop device
    def reset_recording_instance():
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Recording instance reset requested")

    def start_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = True
        print("Recording started")

    def stop_recording_instance():
        nonlocal running_recording_instance
        running_recording_instance = False
        print("Recording paused")

    # Set up teleoperation callbacks.  For IsaacTeleop the primary control
    # path is poll_control_events(); these callbacks are bridged automatically
    # and also serve native (keyboard / spacemouse) devices.
    teleoperation_callbacks = {
        "R": reset_recording_instance,
        "START": start_recording_instance,
        "STOP": stop_recording_instance,
        "RESET": reset_recording_instance,
    }

    teleop_interface = setup_teleop_device(
        teleoperation_callbacks,
        use_isaac_teleop,
        renderer_openxr_session=renderer_openxr_session,
        cloudxr_env_path=cloudxr_env_path,
        cloudxr_launcher=cloudxr_launcher,
    )

    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
    instruction_display = setup_ui(label_text, env)

    runtime_physics_active = _KITLESS_TELEOP_LAUNCHER.env_uses_runtime_physics(env)
    if use_isaac_teleop:
        runtime_hooks = getattr(env_cfg, "install_kitless_teleop_runtime_hooks", None)
        if runtime_physics_active and callable(runtime_hooks):
            runtime_hooks(env)

    action_conditioner = getattr(env_cfg, "condition_kitless_teleop_action", None)
    action_issue_checker = getattr(env_cfg, "teleop_action_target_issue", None)
    action_formatter = getattr(env_cfg, "format_kitless_teleop_action", None)
    skipped_action_log_count = 0
    kitless_action_state: dict[str, torch.Tensor] = {}

    def inner_loop():
        """Inner loop function with access to nonlocal variables."""
        nonlocal current_recorded_demo_count, success_step_count, should_reset_recording_instance
        nonlocal running_recording_instance, label_text, skipped_action_log_count

        # Reset before starting
        env.sim.reset()
        env.reset()
        teleop_interface.reset()
        kitless_action_state.clear()

        subtasks = {}
        stack_name = "IsaacTeleop" if use_isaac_teleop else "native"
        print(f"{stack_name} recording started.")

        if use_isaac_teleop:
            from isaaclab_teleop import poll_control_events

        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while _KITLESS_TELEOP_LAUNCHER.is_loop_running(env):
                # Get teleop command (may be None while waiting for session start)
                action = teleop_interface.advance()

                if use_isaac_teleop:
                    ctrl = poll_control_events(teleop_interface)
                    if ctrl.is_active is not None:
                        running_recording_instance = ctrl.is_active
                    if ctrl.should_reset:
                        should_reset_recording_instance = True

                if (
                    action is not None
                    and use_isaac_teleop
                    and running_recording_instance
                    and runtime_physics_active
                    and callable(action_conditioner)
                ):
                    action = action_conditioner(env, action, kitless_action_state)
                elif action is None or not running_recording_instance:
                    kitless_action_state.clear()

                action_target_issue = (
                    action_issue_checker(action)
                    if callable(action_issue_checker)
                    and use_isaac_teleop
                    and runtime_physics_active
                    and action is not None
                    else None
                )

                if action is None:
                    env.sim.render()
                    continue
                if action_target_issue is not None:
                    if skipped_action_log_count < 20:
                        action_summary = (
                            action_formatter(action)
                            if callable(action_formatter)
                            else f"action_shape={tuple(action.shape)}"
                        )
                        print(f"Skipping unsafe IsaacTeleop action: {action_target_issue}; {action_summary}")
                        skipped_action_log_count += 1
                    env.sim.render()
                    continue
                # Expand to batch dimension
                actions = action.repeat(env.num_envs, 1)

                # Perform action on environment
                if running_recording_instance:
                    # Compute actions based on environment
                    obv = env.step(actions)
                    if subtasks is not None:
                        if subtasks == {}:
                            subtasks = obv[0].get("subtask_terms")
                        elif subtasks:
                            show_subtask_instructions(instruction_display, subtasks, obv, env.cfg)
                else:
                    env.sim.render()

                # Check for success condition
                success_step_count_new, success_reset_needed = process_success_condition(
                    env, success_term, success_step_count
                )
                success_step_count = success_step_count_new
                if success_reset_needed:
                    should_reset_recording_instance = True

                # Update demo count if it has changed
                if env.recorder_manager.exported_successful_episode_count > current_recorded_demo_count:
                    current_recorded_demo_count = env.recorder_manager.exported_successful_episode_count
                    label_text = f"Recorded {current_recorded_demo_count} successful demonstrations."
                    print(label_text)

                # Check if we've reached the desired number of demos
                if (
                    args_cli.num_demos > 0
                    and env.recorder_manager.exported_successful_episode_count >= args_cli.num_demos
                ):
                    label_text = f"All {current_recorded_demo_count} demonstrations recorded.\nExiting the app."
                    instruction_display.show_demo(label_text)
                    print(label_text)
                    target_time = time.time() + 0.8
                    while time.time() < target_time:
                        if rate_limiter:
                            rate_limiter.sleep(env)
                        else:
                            env.sim.render()
                    break

                # Handle reset if requested
                if should_reset_recording_instance:
                    success_step_count = handle_reset(
                        env, success_step_count, instruction_display, label_text, teleop_interface
                    )
                    kitless_action_state.clear()
                    should_reset_recording_instance = False

                # Check if simulation is stopped
                if env.sim.is_stopped():
                    break

                # Rate limiting
                if rate_limiter:
                    rate_limiter.sleep(env)

    # Run the loop with or without context manager based on stack
    if use_isaac_teleop:
        with teleop_interface:
            inner_loop()
    else:
        inner_loop()

    return current_recorded_demo_count


def main() -> None:
    """Collect demonstrations from the environment using teleop interfaces.

    Main function that orchestrates the entire process:
    1. Sets up rate limiting based on configuration
    2. Creates output directories for saving demonstrations
    3. Configures the environment
    4. Runs the simulation loop to collect demonstrations
    5. Cleans up resources when done

    Raises:
        Exception: Propagates exceptions from any of the called functions
    """
    # Set up output directories
    output_dir, output_file_name = setup_output_directories()

    # Create and configure environment
    global env_cfg  # Make env_cfg available to setup_teleop_device
    env_cfg, success_term, use_isaac_teleop = create_environment_config(output_dir, output_file_name)

    # if handtracking or IsaacTeleop is selected, rate limiting is achieved via OpenXR
    if args_cli.xr or use_isaac_teleop:
        rate_limiter = None
        if not _KITLESS_TELEOP_LAUNCHER.enabled:
            from isaaclab.ui.xr_widgets import TeleopVisualizationManager, XRVisualization

            # Assign the teleop visualization manager to the visualization system
            XRVisualization.assign_manager(TeleopVisualizationManager)
    else:
        rate_limiter = RateLimiter(args_cli.step_hz)

    launcher_args = _KITLESS_TELEOP_LAUNCHER.simulation_launcher_args()
    cloudxr_env_path = _KITLESS_TELEOP_LAUNCHER.resolve_cloudxr_env(args_cli.cloudxr_env)
    cloudxr_launcher = _KITLESS_TELEOP_LAUNCHER.launch_cloudxr(
        use_isaac_teleop=use_isaac_teleop,
        cloudxr_env_path=cloudxr_env_path,
        auto_launch=args_cli.auto_launch_cloudxr,
    )
    env = None
    current_recorded_demo_count = 0
    try:
        with launch_simulation(env_cfg, launcher_args):
            # Create environment
            env = create_environment(env_cfg)
            renderer_openxr_session = _KITLESS_TELEOP_LAUNCHER.configure_openxr_teleop(
                env,
                env_cfg,
                enabled=use_isaac_teleop and _KITLESS_TELEOP_LAUNCHER.enabled,
            )

            # Run simulation loop
            current_recorded_demo_count = run_simulation_loop(
                env,
                None,
                success_term,
                rate_limiter,
                use_isaac_teleop,
                renderer_openxr_session=renderer_openxr_session,
                cloudxr_env_path=cloudxr_env_path,
                cloudxr_launcher=cloudxr_launcher,
            )
    finally:
        if env is not None:
            env.close()
        _KITLESS_TELEOP_LAUNCHER.stop_cloudxr(cloudxr_launcher)

    print(f"Recording session completed with {current_recorded_demo_count} successful demonstrations")
    print(f"Demonstrations saved to: {args_cli.dataset_file}")


if __name__ == "__main__":
    # run the main function
    main()
