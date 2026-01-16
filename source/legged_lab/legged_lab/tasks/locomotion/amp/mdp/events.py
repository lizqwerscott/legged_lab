
from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal
import random

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from legged_lab.envs import ManagerBasedAmpEnv


def ref_state_init_root(
    env: ManagerBasedAmpEnv, 
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_rsi: bool = True,
):
    """Reference State Initialization (RSI) for the root of the robot.
    Sample from the motion loader and set the root position and orientation.
    Refer to the paper of Adversarial Motion Priors (AMP) for more details.

    Args:
        env (AmpEnv): The manager-based env.
        env_ids (torch.Tensor): The env IDs to reset.
        asset_cfg (SceneEntityCfg, optional): The asset configuration. Defaults to SceneEntityCfg("robot").
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    num_envs = env_ids.shape[0]
    dt = env.cfg.sim.dt * env.cfg.decimation

    if motion_dataset is None:
        # select one dataset randomly by weights
        term_weights = env.motion_data_manager.get_term_weights()
        motion_dataset = random.choices(list(term_weights.keys()), weights=list(term_weights.values()))[0]
    else:
        if motion_dataset not in env.motion_data_manager.active_terms():
            raise ValueError(f"Motion dataset '{motion_dataset}' not found in the active terms.")
    motion_loader = env.motion_data_manager.get_term(motion_dataset)
    motion_ids = motion_loader.sample_motions(num_envs)
    motion_times = motion_loader.sample_times(motion_ids, truncate_time=dt)
    motion_state_dict = motion_loader.get_motion_state(motion_ids, motion_times)
    
    lift_a_little = 0.05
    # lift the root position a little bit to avoid collision with the ground
    motion_state_dict["root_pos_w"][:, 2] += lift_a_little
    
    if not pos_rsi:
        motion_state_dict["root_pos_w"][:, :2] = 0.0    # no offset in x and y
    ref_root_pos_w = motion_state_dict["root_pos_w"] + env.scene.env_origins[env_ids]
    ref_root_quat = motion_state_dict["root_quat"]
    ref_root_vel_w = motion_state_dict["root_vel_w"]
    ref_root_ang_vel_w = motion_state_dict["root_ang_vel_w"]
    
    asset.write_root_pose_to_sim(
        torch.cat([ref_root_pos_w, ref_root_quat], dim=-1),
        env_ids=env_ids,
    )
    asset.write_root_velocity_to_sim(
        torch.cat([ref_root_vel_w, ref_root_ang_vel_w], dim=-1),
        env_ids=env_ids,
    )
    

def ref_state_init_dof(
    env: ManagerBasedAmpEnv, 
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_dataset: str | None = None,
):
    """Reference State Initialization (RSI) for the joints (DoF) of the robot.
    Sample from the motion loader and set the joint positions and velocities.
    Refer to the paper of Adversarial Motion Priors (AMP) for more details.

    Args:
        env (AmpEnv): The manager-based env.
        env_ids (torch.Tensor): The env IDs to reset.
        asset_cfg (SceneEntityCfg, optional): The asset configuration. Defaults to SceneEntityCfg("robot").
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    num_envs = env_ids.shape[0]
    dt = env.cfg.sim.dt * env.cfg.decimation
    
    if motion_dataset is None:
        # select one dataset randomly by weights
        term_weights = env.motion_data_manager.get_term_weights()
        motion_dataset = random.choices(list(term_weights.keys()), weights=list(term_weights.values()))[0]
    else:
        if motion_dataset not in env.motion_data_manager.active_terms():
            raise ValueError(f"Motion dataset '{motion_dataset}' not found in the active terms.")
    motion_loader = env.motion_data_manager.get_term(motion_dataset)
    motion_ids = motion_loader.sample_motions(num_envs)
    motion_times = motion_loader.sample_times(motion_ids, truncate_time=dt)
    motion_state_dict = motion_loader.get_motion_state(motion_ids, motion_times)

    joint_pos = motion_state_dict["dof_pos"]
    joint_vel = motion_state_dict["dof_vel"]
    
    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization, or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)

"""
Internal helper functions.
"""

    
def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data

