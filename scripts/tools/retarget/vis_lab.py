"""
Visualization tool for Legged Lab motion data.

This script loads multiple Legged Lab format motion files and visualizes them simultaneously
in the Isaac Lab simulator. Each motion is replayed in a separate environment.

Behavior:
 - Reads all .pkl files from the input directory (sorted).
 - For each file, loads the Legged Lab pickle data using extract_lab_data.
 - Runs the simulator with all motions (num_envs = number of motions).
 - Visualizes key body positions with markers.
 - Loops playback continuously until simulation is stopped.

Usage example:
    python scripts/tools/retarget/vis_lab.py \
        --robot g1 \
        --input_dir data/lab/ \
        --config_file scripts/tools/retarget/config/g1_29dof.yaml

Note: This is a visualization tool, not a conversion tool. It expects Legged Lab format files.
"""

import argparse
import os
import pickle
import yaml
from pathlib import Path

from isaaclab.app import AppLauncher

# append AppLauncher cli args
parser = argparse.ArgumentParser(
    description="Batch retarget GMR -> Legged Lab (multiple files)."
)
parser.add_argument(
    "--robot",
    type=str,
    default="g1",
    help="Robot name to use (default: g1)",
)
parser.add_argument(
    "--input_dir",
    type=str,
    required=True,
    help="Directory containing input GMR .pkl files",
)
parser.add_argument(
    "--config_file",
    type=str,
    required=True,
    help="Path to YAML config containing gmr_dof_names, lab_dof_names, lab_key_body_names",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

"""Launch Omniverse App"""
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import numpy as np
import sys
import torch
import warnings

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

# load robot cfg as single_retarget does
if args_cli.robot == "g1":
    from legged_lab.assets.unitree import UNITREE_G1_29DOF_CFG as ROBOT_CFG
elif args_cli.robot == "g1_23dof":
    from legged_lab.assets.unitree import UNITREE_G1_23DOF_CFG as ROBOT_CFG

else:
    raise ValueError(f"Robot {args_cli.robot} not supported.")

import enum

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


class LoopMode(enum.Enum):
    CLAMP = 0
    WRAP = 1


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = None


def extract_lab_data(lab_file_path: str):
    with open(lab_file_path, "rb") as f:
        lab_data = pickle.load(f)

    # Extract data from GMR format
    fps = lab_data["fps"]
    root_pos = lab_data["root_pos"]  # Shape: (num_frames, 3)
    root_rot_quat = lab_data["root_rot"]  # Shape: (num_frames, 4), quaternion format
    dof_pos = lab_data["dof_pos"]  # Shape: (num_frames, num_dofs)
    loop_mode_value = lab_data["loop_mode"]  # Shape: (num_frames, 1)
    key_body_pos = lab_data["key_body_pos"]  # Shape: (num_frames, num_key_body_names)

    if loop_mode_value == 0:
        loop_mode = LoopMode.CLAMP
    elif loop_mode_value == 1:
        loop_mode = LoopMode.WRAP

    # Log the type and shape of each extracted term
    print("\n" + "=" * 60)
    print("📥 LOADED GMR DATA")
    print("=" * 60)
    print(f"⏱️  FPS:           type={type(fps).__name__}, value={fps}")
    print(f"📍 Root Position: type={type(root_pos).__name__}, shape={root_pos.shape}")
    print(
        f"🔄 Root Rotation: type={type(root_rot_quat).__name__}, shape={root_rot_quat.shape}"
    )
    print(f"🦴 DOF Position:  type={type(dof_pos).__name__}, shape={dof_pos.shape}")
    print(
        f"🦴 Key body pos:  type={type(key_body_pos).__name__}, shape={key_body_pos.shape}"
    )
    print(f"Loop Mode:       {loop_mode.name} ({loop_mode.value})")
    print("=" * 60 + "\n")

    output_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot_quat,
        "dof_pos": dof_pos,
        "loop_mode": loop_mode,
        "key_body_pos": key_body_pos,
    }

    return output_data


def run_vis_simulator(
    simulation_app,
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    motion_data_dicts: list[dict[str, np.ndarray]],
    key_body_names: list[str],
    marker: VisualizationMarkers,
):

    robot: Articulation = scene["robot"]

    # get the motion data
    num_motions = len(motion_data_dicts)
    assert (
        num_motions == scene.num_envs
    ), "Number of motions must match number of environments."
    fps = motion_data_dicts[0]["fps"]
    root_pos_w_list = []
    root_quat_list = []
    dof_pos_list = []
    num_frames_list = []

    for motion_data in motion_data_dicts:
        # assert motion_data['fps'] == fps, "All motions must have the same fps."
        root_pos_w_list.append(
            torch.from_numpy(motion_data["root_pos"]).to(scene.device).float()
        )

        root_quat_list.append(
            torch.from_numpy(motion_data["root_rot"]).to(scene.device).float()
        )

        dof_pos_list.append(
            torch.from_numpy(motion_data["dof_pos"]).to(scene.device).float()
        )
        num_frames_list.append(motion_data["dof_pos"].shape[0])

    max_num_frames = max(num_frames_list)

    lab_body_names = robot.data.body_names
    key_body_indices = []

    for name in key_body_names:
        if name in lab_body_names:
            key_body_indices.append(lab_body_names.index(name))
        else:
            raise ValueError(
                f"Key body name '{name}' not found in Legged Lab body names."
            )
    key_body_pos_w_list = [
        torch.zeros((num_frames, len(key_body_indices), 3), device=scene.device)
        for num_frames in num_frames_list
    ]

    count = 0
    sim_time = 0.0
    dt = sim.cfg.dt

    while simulation_app.is_running():
        root_states = robot.data.default_root_state.clone()
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = torch.zeros_like(robot.data.default_joint_vel)

        for motion_idx in range(num_motions):
            num_frames = num_frames_list[motion_idx]
            frame_idx = count if count < num_frames else num_frames - 1

            # set root state
            root_states[motion_idx, :3] = root_pos_w_list[motion_idx][frame_idx, :]
            root_states[motion_idx, :3] += scene.env_origins[motion_idx, :3]
            root_states[motion_idx, 3:7] = root_quat_list[motion_idx][frame_idx, :]
            root_states[motion_idx, 7:10] = 0.0  # zero linear velocity
            root_states[motion_idx, 10:13] = 0.0  # zero angular velocity

            # set joint state
            joint_pos[motion_idx, :] = dof_pos_list[motion_idx][frame_idx, :]

        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # step without physics
        sim.render()
        scene.update(dt)

        vis_key_body_pos_w = robot.data.body_pos_w[:, key_body_indices, :]
        marker.visualize(translations=vis_key_body_pos_w.reshape(-1, 3))

        count += 1
        sim_time += dt
        if count >= max_num_frames:
            break


def list_input_files(input_dir: str):
    p = Path(input_dir)
    if not p.exists() or not p.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    files = sorted([f for f in p.iterdir() if f.is_file() and f.suffix == ".pkl"])
    return files


def main():
    # read config
    with open(args_cli.config_file, "r") as f:
        config = yaml.safe_load(f)

    gmr_dof_names = config["gmr_dof_names"]
    lab_dof_names = config["lab_dof_names"]
    lab_key_body_names = config["lab_key_body_names"]

    # load data from lab format
    input_files = list_input_files(args_cli.input_dir)
    if len(input_files) == 0:
        print(f"No .pkl files found in input directory: {args_cli.input_dir}")
        return

    # load and convert all gmr files (entire motion)
    motion_data_dicts = []
    input_names = []
    fps_values = []

    print(f"Found {len(input_files)} files to convert.")
    for p in input_files:
        print(f"Loading and converting: {p.name}")
        motion = extract_lab_data(
            lab_file_path=str(p),
        )
        motion_data_dicts.append(motion)
        input_names.append(p.name)
        fps_values.append(motion["fps"])
        print(
            f"  - FPS: {motion['fps']}, Frames: {motion['dof_pos'].shape[0]}, Loop Mode: {motion['loop_mode'].name}"
        )

    # check fps consistency
    if not all(f == fps_values[0] for f in fps_values):
        print(fps_values)
        warnings.warn("Motions have different fps. Using fps from first motion.")

    # check loop_mode consistency
    loop_modes = [motion["loop_mode"] for motion in motion_data_dicts]
    if not all(mode == loop_modes[0] for mode in loop_modes):
        warnings.warn(
            "Motions have different loop modes. Using loop mode from first motion."
        )

    fps = fps_values[0]
    dt = 1.0 / fps

    # start simulation context
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=dt, device=args_cli.device)
    )
    scene_cfg = ReplayMotionsSceneCfg(
        num_envs=len(motion_data_dicts),
        env_spacing=3.0,
        robot=ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    )
    scene = InteractiveScene(scene_cfg)

    sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

    sim.reset()
    print("Simulation starting ...")

    # marker
    marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/FrameVisualizerFromScript",
        markers={
            "red_sphere": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
        },
    )

    marker: VisualizationMarkers = VisualizationMarkers(marker_cfg)

    # run simulator with all motions
    count = 1
    while True:
        print("Play:", count)
        run_vis_simulator(
            simulation_app, sim, scene, motion_data_dicts, lab_key_body_names, marker
        )
        count = count + 1
        sim.reset()

    print("Closing simulation app...")
    simulation_app.close()
    print("Done.")


if __name__ == "__main__":
    main()
