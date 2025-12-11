"""Functions to specify the symmetry in the observation and action space for Unitree G1 23dof."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying
    four symmetrical transformations: original, left-right, front-back, and diagonal. The symmetry
    transformations are beneficial for reinforcement learning tasks by providing additional
    diverse data without requiring additional data collection.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """
    if obs is not None:
        batch_size = obs.batch_size[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        obs_aug = obs.repeat(2)

        # policy observation group
        # -- original
        obs_aug["policy"][:batch_size] = obs["policy"][:]
        # -- left-right
        obs_aug["policy"][batch_size : 2 * batch_size] = (
            _transform_policy_obs_left_right(env.unwrapped, obs["policy"][:])
        )
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        # since we have 2 different symmetries, we need to augment the batch size by 2
        actions_aug = torch.zeros(
            batch_size * 2, actions.shape[1], device=actions.device
        )
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(
            actions
        )
    else:
        actions_aug = None

    return obs_aug, actions_aug


"""
Symmetry functions for observations.
"""


def _transform_policy_obs_left_right(
    env: ManagerBasedRLEnv, obs: torch.Tensor
) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the observation tensor.

    This function modifies the given observation tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    negating certain components of the linear and angular velocities, projected gravity,
    velocity commands, and flipping the joint positions, joint velocities, and last actions
    for the ANYmal robot. Additionally, if height-scan data is present, it is flipped
    along the relevant dimension.

    Args:
        env: The environment instance from which the observation is obtained.
        obs: The observation tensor to be transformed.

    Returns:
        The transformed observation tensor with left-right symmetry applied.
    """
    # copy observation tensor
    obs = obs.clone()
    device = obs.device
    joint_num = 23  # G1 23dof
    key_body_num = 6

    # policy_obs_term_dim = env.observation_manager.group_obs_term_dim["policy"]
    # [(15,), (15,), (15,), (145,), (145,), (145,)]
    HISTORY_LEN = 5
    ANG_VEL_DIM = 3
    ROT_TAN_NORM = 6
    VEL_CMD_DIM = 3
    JOINT_POS_DIM = joint_num
    JOINT_VEL_DIM = joint_num
    LAST_ACTIONS_DIM = joint_num
    KEY_BODY_POS_DIM = key_body_num * 3

    end_idx = 0
    # ang vel
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ANG_VEL_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor(
            [-1, 1, -1], device=device
        )
    # root rot tan norm
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + ROT_TAN_NORM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor(
            [1, -1, 1, 1, -1, 1], device=device
        )
    # velocity command
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + VEL_CMD_DIM
        obs[:, start_idx:end_idx] = obs[:, start_idx:end_idx] * torch.tensor(
            [1, -1, -1], device=device
        )
    # joint pos
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_POS_DIM
        obs[:, start_idx:end_idx] = _switch_g1_23dof_joints_left_right(
            obs[:, start_idx:end_idx]
        )
    # joint vel
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + JOINT_VEL_DIM
        obs[:, start_idx:end_idx] = _switch_g1_23dof_joints_left_right(
            obs[:, start_idx:end_idx]
        )
    # last actions
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + LAST_ACTIONS_DIM
        obs[:, start_idx:end_idx] = _switch_g1_23dof_joints_left_right(
            obs[:, start_idx:end_idx]
        )
    # key body pos
    for h in range(HISTORY_LEN):
        start_idx = end_idx
        end_idx = start_idx + KEY_BODY_POS_DIM
        obs[:, start_idx:end_idx] = _switch_g1_23dof_key_body_pos_left_right(
            obs[:, start_idx:end_idx]
        )

    return obs


"""
Symmetry functions for actions.
"""


def _transform_actions_left_right(actions: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the actions tensor.

    This function modifies the given actions tensor by applying transformations
    that represent a symmetry with respect to the left-right axis. This includes
    flipping the joint positions, joint velocities, and last actions for the
    ANYmal robot.

    Args:
        actions: The actions tensor to be transformed.

    Returns:
        The transformed actions tensor with left-right symmetry applied.
    """
    actions = actions.clone()
    actions[:] = _switch_g1_23dof_joints_left_right(actions[:])
    return actions


"""
Lab joint names:
 0 - left_hip_pitch_joint
 1 - right_hip_pitch_joint
 2 - waist_yaw_joint
 3 - left_hip_roll_joint
 4 - right_hip_roll_joint
 5 - left_hip_yaw_joint
 6 - right_hip_yaw_joint
 7 - left_knee_joint
 8 - right_knee_joint
 9 - left_shoulder_pitch_joint
10 - right_shoulder_pitch_joint
11 - left_ankle_pitch_joint
12 - right_ankle_pitch_joint
13 - left_shoulder_roll_joint
14 - right_shoulder_roll_joint
15 - left_ankle_roll_joint
16 - right_ankle_roll_joint
17 - left_shoulder_yaw_joint
18 - right_shoulder_yaw_joint
19 - left_elbow_joint
20 - right_elbow_joint
21 - left_wrist_roll_joint
22 - right_wrist_roll_joint
"""


def _switch_g1_23dof_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    joint_data_switched = torch.zeros_like(joint_data)

    # Indices for left and right joints
    left_indices = [0, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    right_indices = [1, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    # Indices for roll and yaw joints that need sign flipping
    roll_indices = [3, 4, 15, 16, 17]
    yaw_indices = [5, 6, 17, 18]

    # Copy non-symmetric joints first (waist joints)
    joint_data_switched[..., [2]] = joint_data[..., [2]]

    # Swap left and right joints
    joint_data_switched[..., left_indices] = joint_data[..., right_indices]
    joint_data_switched[..., right_indices] = joint_data[..., left_indices]

    # Flip the sign of roll and yaw joints
    joint_data_switched[..., roll_indices] *= -1.0
    joint_data_switched[..., yaw_indices] *= -1.0

    # Flip the sign of waist_yaw, waist_roll
    joint_data_switched[..., [2]] *= -1.0

    return joint_data_switched


def _switch_g1_23dof_key_body_pos_left_right(
    key_body_pos: torch.Tensor,
) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the key body positions tensor."""

    # We assume that the key body are in pair, for example:
    # "left_ankle_roll_link",
    # "right_ankle_roll_link",
    # "left_wrist_roll_rubber_hand",
    # "right_wrist_roll_rubber_hand",
    # "left_shoulder_roll_link",
    # "right_shoulder_roll_link",

    key_body_pos_switched = key_body_pos.clone()
    num_key_bodies = key_body_pos.shape[-1] // 3

    for i in range(num_key_bodies // 2):
        left_idx = i * 2
        right_idx = i * 2 + 1

        # Swap left and right key body positions
        key_body_pos_switched[..., left_idx * 3 : left_idx * 3 + 3] = key_body_pos[
            ..., right_idx * 3 : right_idx * 3 + 3
        ]
        key_body_pos_switched[..., right_idx * 3 : right_idx * 3 + 3] = key_body_pos[
            ..., left_idx * 3 : left_idx * 3 + 3
        ]

        # Flip the y-coordinate to reflect left-right symmetry
        key_body_pos_switched[..., left_idx * 3 + 1] *= -1.0
        key_body_pos_switched[..., right_idx * 3 + 1] *= -1.0

    return key_body_pos_switched
