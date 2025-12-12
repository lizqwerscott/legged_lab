import numpy as np
import os
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names


def format_value(x):
    """Format values for YAML output."""
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir):
    """
    Export deployment configuration from environment to deploy.yaml file.
    
    This function extracts all necessary configuration for deployment:
    - Joint mappings and PD parameters
    - Action configurations (scale, clip, offset)
    - Observation configurations (scale, clip, history_length)
    - Command ranges
    
    Args:
        env: The ManagerBasedRLEnv environment instance
        log_dir: Directory where deploy.yaml will be saved (to log_dir/params/deploy.yaml)
    """
    asset: Articulation = env.scene["robot"]
    
    # Handle joint_sdk_names: use if available, otherwise try to get from unitree_rl_lab
    if hasattr(env.cfg.scene.robot, "joint_sdk_names") and env.cfg.scene.robot.joint_sdk_names is not None:
        joint_sdk_names = env.cfg.scene.robot.joint_sdk_names
        joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)
    else:
        # Try to import from unitree_rl_lab for G1 23DOF
        try:
            from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_23DOF_CFG
            joint_sdk_names = UNITREE_G1_23DOF_CFG.joint_sdk_names
            joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)
        except (ImportError, AttributeError):
            # If import fails, use joint_names directly (1:1 mapping)
            joint_sdk_names = asset.data.joint_names
            joint_ids_map = list(range(len(joint_sdk_names)))

    cfg = {}  # noqa: SIM904
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation
    stiffness = np.zeros(len(joint_sdk_names))
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy().tolist()
    cfg["stiffness"] = stiffness.tolist()
    damping = np.zeros(len(joint_sdk_names))
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy().tolist()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    # --- commands ---
    cfg["commands"] = {}
    if hasattr(env.cfg.commands, "base_velocity"):  # some environments do not have base_velocity command
        cfg["commands"]["base_velocity"] = {}
        if hasattr(env.cfg.commands.base_velocity, "limit_ranges"):
            ranges = env.cfg.commands.base_velocity.limit_ranges.to_dict()
        else:
            ranges = env.cfg.commands.base_velocity.ranges.to_dict()
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"]["base_velocity"]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    
    # Map action names to deployment class names
    action_name_to_class = {
        "joint_pos": "JointPositionAction",
        "joint_vel": "JointVelocityAction",
        "JointPositionAction": "JointPositionAction",  # Already correct
        "JointVelocityAction": "JointVelocityAction",  # Already correct
    }
    
    for action_name, action_term in action_terms:
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, float):
            term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
        else:  # dict
            term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        # Check if this is a joint position/velocity action (by class name or config name)
        is_joint_action = (action_name in ["JointPositionAction", "JointVelocityAction", "joint_pos", "joint_vel"] or
                          action_term.__class__.__name__ in ["JointPositionAction", "JointVelocityAction"])
        if is_joint_action:
            if hasattr(term_cfg, "use_default_offset") and term_cfg.use_default_offset:
                # Use default joint positions from asset
                term_cfg.offset = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()
            else:
                # Use zero offset for all joints
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = term_cfg.to_dict()

        for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            del term_cfg[_]

        if action_term._joint_ids == slice(None):
            term_cfg["joint_ids"] = None
        else:
            term_cfg["joint_ids"] = action_term._joint_ids
        
        # Use deployment class name instead of config name
        deploy_action_name = action_name_to_class.get(action_name, action_name)
        cfg["actions"][deploy_action_name] = term_cfg

    # --- observations ---
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    
    # Map observation names to deployment names
    obs_name_to_deploy = {
        "joint_pos": "joint_pos_rel",
        "joint_vel": "joint_vel_rel",
        "actions": "last_action",
        "velocity_commands": "velocity_commands",  # May need custom registration
        "base_ang_vel": "base_ang_vel",
        "projected_gravity": "projected_gravity",
        # Keep original names if already correct
        "joint_pos_rel": "joint_pos_rel",
        "joint_vel_rel": "joint_vel_rel",
        "last_action": "last_action",
    }
    
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = obs_cfg.copy()
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = term_cfg.to_dict()
        for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            if _ in term_cfg:
                del term_cfg[_]
        
        # Use deployment observation name
        deploy_obs_name = obs_name_to_deploy.get(obs_name, obs_name)
        cfg["observations"][deploy_obs_name] = term_cfg

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not isinstance(cfg, dict):
        cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
    
    print(f"[INFO] Deployment configuration exported to: {filename}")

