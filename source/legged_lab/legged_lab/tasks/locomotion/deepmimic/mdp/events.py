from __future__ import annotations

import math
import re
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.version import compare_versions

if TYPE_CHECKING:
    from legged_lab.envs import ManagerBasedAnimationEnv
    from legged_lab.managers import AnimationTerm
    
def reset_from_ref(
    env: ManagerBasedAnimationEnv, 
    env_ids: torch.Tensor, 
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.1,
):
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    
    offset = torch.tensor([0.0, 0.0, height_offset], device=env.device, dtype=torch.float32).unsqueeze(0)  # (1, 3)
    position = animation_term.get_root_pos_w(env_ids)[:, 0, :] + env.scene.env_origins[env_ids, :] + offset
    orientation = animation_term.get_root_quat(env_ids)[:, 0, :]
    lin_vel = animation_term.get_root_vel_w(env_ids)[:, 0, :]
    ang_vel = animation_term.get_root_ang_vel_w(env_ids)[:, 0, :]
    
    pos = torch.cat([position, orientation], dim=-1)
    vel = torch.cat([lin_vel, ang_vel], dim=-1)
    
    robot.write_root_pose_to_sim(pos, env_ids=env_ids)
    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)
    
    dof_pos = animation_term.get_dof_pos(env_ids)[:, 0, :]
    dof_vel = animation_term.get_dof_vel(env_ids)[:, 0, :]
    robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
    

def reset_from_ref_random(
    env: ManagerBasedAnimationEnv, 
    env_ids: torch.Tensor, 
    animation: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_offset: float = 0.1,
    pose_offset_range: Optional[dict[str, tuple[float, float]]] = None,
    velocity_offset_range: Optional[dict[str, tuple[float, float]]] = None,
    joint_position_offset_range: Optional[tuple[float, float]] = None,
    joint_velocity_offset_range: Optional[tuple[float, float]] = None,
):
    """Reset the robot state from reference animation with random offsets.
    
    This function resets the robot to a reference animation state with optional random offsets
    applied to the root pose, root velocity, joint positions, and joint velocities.
    
    Args:
        env: The animation environment.
        env_ids: Environment IDs to reset.
        animation: Name of the animation to use as reference.
        asset_cfg: Configuration for the robot asset.
        height_offset: Base height offset to apply.
        pose_offset_range: Dictionary of pose offset ranges for each axis (x, y, z, roll, pitch, yaw).
            Each key maps to a tuple (min, max) offset range.
        velocity_offset_range: Dictionary of velocity offset ranges for each axis (x, y, z, roll, pitch, yaw).
            Each key maps to a tuple (min, max) offset range.
        joint_position_offset_range: Tuple (min, max) for joint position offsets.
        joint_velocity_offset_range: Tuple (min, max) for joint velocity offsets.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    animation_term: AnimationTerm = env.animation_manager.get_term(animation)
    
    # 基础高度偏移
    base_offset = torch.tensor([0.0, 0.0, height_offset], device=env.device, dtype=torch.float32).unsqueeze(0)  # (1, 3)
    
    # 获取参考动画的根状态
    ref_position = animation_term.get_root_pos_w(env_ids)[:, 0, :] + env.scene.env_origins[env_ids, :] + base_offset
    ref_orientation = animation_term.get_root_quat(env_ids)[:, 0, :]
    ref_lin_vel = animation_term.get_root_vel_w(env_ids)[:, 0, :]
    ref_ang_vel = animation_term.get_root_ang_vel_w(env_ids)[:, 0, :]
    
    # 应用根位姿随机偏移
    if pose_offset_range is not None:
        # 位置偏移
        pos_keys = ["x", "y", "z"]
        pos_ranges = [pose_offset_range.get(key, (0.0, 0.0)) for key in pos_keys]
        pos_ranges_tensor = torch.tensor(pos_ranges, device=env.device)
        
        if torch.any(pos_ranges_tensor[:, 0] != pos_ranges_tensor[:, 1]):
            pos_offsets = math_utils.sample_uniform(
                pos_ranges_tensor[:, 0], 
                pos_ranges_tensor[:, 1], 
                (len(env_ids), 3), 
                device=env.device
            )
            ref_position = ref_position + pos_offsets
        
        # 方向偏移
        rot_keys = ["roll", "pitch", "yaw"]
        rot_ranges = [pose_offset_range.get(key, (0.0, 0.0)) for key in rot_keys]
        rot_ranges_tensor = torch.tensor(rot_ranges, device=env.device)
        
        if torch.any(rot_ranges_tensor[:, 0] != rot_ranges_tensor[:, 1]):
            rot_offsets = math_utils.sample_uniform(
                rot_ranges_tensor[:, 0], 
                rot_ranges_tensor[:, 1], 
                (len(env_ids), 3), 
                device=env.device
            )
            orientation_delta = math_utils.quat_from_euler_xyz(
                rot_offsets[:, 0], rot_offsets[:, 1], rot_offsets[:, 2]
            )
            ref_orientation = math_utils.quat_mul(ref_orientation, orientation_delta)
    
    # 应用根速度随机偏移
    if velocity_offset_range is not None:
        vel_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        vel_ranges = [velocity_offset_range.get(key, (0.0, 0.0)) for key in vel_keys]
        vel_ranges_tensor = torch.tensor(vel_ranges, device=env.device)
        
        if torch.any(vel_ranges_tensor[:, 0] != vel_ranges_tensor[:, 1]):
            vel_offsets = math_utils.sample_uniform(
                vel_ranges_tensor[:, 0], 
                vel_ranges_tensor[:, 1], 
                (len(env_ids), 6), 
                device=env.device
            )
            ref_lin_vel = ref_lin_vel + vel_offsets[:, 0:3]
            ref_ang_vel = ref_ang_vel + vel_offsets[:, 3:6]
    
    # 设置根状态
    pos = torch.cat([ref_position, ref_orientation], dim=-1)
    vel = torch.cat([ref_lin_vel, ref_ang_vel], dim=-1)
    
    robot.write_root_pose_to_sim(pos, env_ids=env_ids)
    robot.write_root_velocity_to_sim(vel, env_ids=env_ids)
    
    # 获取关节状态
    dof_pos = animation_term.get_dof_pos(env_ids)[:, 0, :]
    dof_vel = animation_term.get_dof_vel(env_ids)[:, 0, :]
    
    # 应用关节位置随机偏移
    if joint_position_offset_range is not None:
        if joint_position_offset_range[0] != joint_position_offset_range[1]:
            joint_pos_offsets = math_utils.sample_uniform(
                joint_position_offset_range[0], 
                joint_position_offset_range[1], 
                dof_pos.shape, 
                device=env.device
            )
            dof_pos = dof_pos + joint_pos_offsets
    
    # 应用关节速度随机偏移
    if joint_velocity_offset_range is not None:
        if joint_velocity_offset_range[0] != joint_velocity_offset_range[1]:
            joint_vel_offsets = math_utils.sample_uniform(
                joint_velocity_offset_range[0], 
                joint_velocity_offset_range[1], 
                dof_vel.shape, 
                device=env.device
            )
            dof_vel = dof_vel + joint_vel_offsets
    
    # 设置关节状态
    robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

    
    
    
