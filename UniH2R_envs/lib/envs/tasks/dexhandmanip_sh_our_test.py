from __future__ import annotations

import os
import random
from enum import Enum
from itertools import cycle
from time import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from ...utils import torch_jit_utils as torch_jit_utils
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul
import math
from UniH2R_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory

from main.dataset.oakink2_dataset_dexhand_rh import OakInk2DatasetDexHandRH
from main.dataset.oakink2_dataset_dexhand_lh import OakInk2DatasetDexHandLH
from main.dataset.oakink2_dataset_utils import oakink2_obj_scale, oakink2_obj_mass
from main.dataset.transform import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat, rot6d_to_aa
from torch import Tensor
from tqdm import tqdm
from ...asset_root import ASSET_ROOT


from ..core.config import ROBOT_HEIGHT, config
from ...envs.core.sim_config import sim_config
from ...envs.core.vec_task import VecTask
from ...utils.pose_utils import get_mat


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class OurTestDexHandManipRHEnv(VecTask):

    side = "right"

    def __init__(
        self,
        cfg,
        *,
        rl_device: int = 0,
        sim_device: int = 0,
        graphics_device_id: int = 0,
        display: bool = False,
        record: bool = False,
        headless: bool = True,
    ):
        self._record = record
        self.cfg = cfg

        use_quat_rot = self.use_quat_rot = self.cfg["env"]["useQuatRot"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.training = self.cfg["env"]["training"]

        if not hasattr(self, "dexhand"):
            self.dexhand = DexHandFactory.create_hand(self.cfg["env"]["dexhand"], "right")

        self.use_pid_control = self.cfg["env"]["usePIDControl"]
        if self.use_pid_control:
            self.Kp_rot = self.dexhand.Kp_rot
            self.Ki_rot = self.dexhand.Ki_rot
            self.Kd_rot = self.dexhand.Kd_rot

            self.Kp_pos = self.dexhand.Kp_pos
            self.Ki_pos = self.dexhand.Ki_pos
            self.Kd_pos = self.dexhand.Kd_pos

                                                                                                                      
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.translation_scale = self.cfg["env"]["translationScale"]
        self.orientation_scale = self.cfg["env"]["orientationScale"]

                                                                      
                               
        self._prop_dump_info = self.cfg["env"]["propDumpInfo"]

                                           
        self.states = {}
        self.dexhand_handles = {}                                                      
        self.objs_handles = {}                    
        self.objs_assets = {}
        self.num_dofs = None                                
        self.actions = None                                  

        self.dataIndices = self.cfg["env"]["dataIndices"]
        self.obs_future_length = self.cfg["env"]["obsFutureLength"]
        self.rollout_state_init = self.cfg["env"]["rolloutStateInit"]
        self.random_state_init = self.cfg["env"]["randomStateInit"]

        self.tighten_method = self.cfg["env"]["tightenMethod"]
        self.tighten_factor = self.cfg["env"]["tightenFactor"]
        self.tighten_steps = self.cfg["env"]["tightenSteps"]

        self.rollout_len = self.cfg["env"].get("rolloutLen", None)
        self.rollout_begin = self.cfg["env"].get("rolloutBegin", None)

                        
        self.debug_obs = self.cfg["env"].get("debug_observations", False)              

        assert len(self.dataIndices) == 1 or self.rollout_len is None, "rolloutLen only works with one data"
        assert len(self.dataIndices) == 1 or self.rollout_begin is None, "rolloutBegin only works with one data"

                             
        self._root_state = None                                          
        self._dof_state = None                                             
        self._q = None                                             
        self._qd = None                                             
        self._rigid_body_state = None                                                                
        self.net_cf = None                 
        self._eef_state = None                                          
        self._ftip_center_state = None                        
        self._eef_lf_state = None                                          
        self._eef_rf_state = None                                          
        self._j_eef = None                             
        self._mm = None               
        self._pos_control = None                    
        self._effort_control = None                  
        self._dexhand_effort_limits = None                                        
        self._dexhand_dof_speed_limits = None                                       
        self._global_dexhand_indices = None                                                               

        self.sim_device = torch.device(sim_device)
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
        
                            
                 
                 
                 
                   
                     
                     
                     
                     
                     
                     
                                                   
                     
                     
                     
                     
                     
                     
                     
                                         
               
                                      
                   
                                
        TARGET_OBS_DIM = self.cfg["env"]["target_max_dim"]
        
        self.obs_dict.update(
            {
                "target": torch.zeros((self.num_envs, TARGET_OBS_DIM), device=self.device),
            }
        )
        obs_space = self.obs_space.spaces
        obs_space["target"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(TARGET_OBS_DIM,),
        )
        self.obs_space = spaces.Dict(obs_space)

        default_pose = torch.ones(self.dexhand.n_dofs, device=self.device) * np.pi / 12
        if self.cfg["env"]["dexhand"] == "inspire":
            default_pose[8] = 0.3
            default_pose[9] = 0.01
        self.dexhand_default_dof_pos = torch.tensor(default_pose, device=self.sim_device)
                        
        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere", n_bps_points=128, radius=0.2, randomize=False, device=self.device
        )

        obj_verts = self.demo_data["obj_verts"]
        self.obj_bps = self.bps_layer.encode(obj_verts, feature_type=self.bps_feat_type)[self.bps_feat_type]

                                
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

                         
        self._refresh()

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8
        self.sim = super().create_sim(
            self.device_id,
            self.graphics_device_id,
            self.physics_engine,
            self.sim_params,
        )
        self._create_ground_plane()
        self._create_envs()

        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

                                  
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True

        table_width_offset = 0.2
        table_asset = self.gym.create_box(self.sim, 0.8 + table_width_offset, 1.6, 0.03, table_asset_options)

        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self.dexhand_pose = gymapi.Transform()
        table_half_height = 0.015
        table_half_width = 0.4

        self._table_surface_z = table_surface_z = table_pos.z + table_half_height
        self.dexhand_pose.p = gymapi.Vec3(-table_half_width, 0, table_surface_z + ROBOT_HEIGHT)
        self.dexhand_pose.r = gymapi.Quat.from_euler_zyx(0, -np.pi / 2, 0)

        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))

        self.demo_dataset_dict = {}
        for dataset_type in dataset_list:
            self.demo_dataset_dict[dataset_type] = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side=self.side,
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=self.dexhand,
                embodiment=self.cfg["env"]["dexhand"],
            )

        dexhand_asset_file = self.dexhand.urdf_path
        asset_options = gymapi.AssetOptions()
        asset_options.thickness = 0.001
        asset_options.angular_damping = 20
        asset_options.linear_damping = 20
        asset_options.max_linear_velocity = 50
        asset_options.max_angular_velocity = 100
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        dexhand_asset = self.gym.load_asset(self.sim, *os.path.split(dexhand_asset_file), asset_options)
        dexhand_dof_stiffness = torch.tensor(
            [500] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_dof_damping = torch.tensor(
            [30] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_asset)
        for element in rigid_shape_props_asset:
            element.friction = 4.0
            element.rolling_friction = 0.01
            element.torsion_friction = 0.01

        self.gym.set_asset_rigid_shape_properties(dexhand_asset, rigid_shape_props_asset)

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        print(f"Num dexhand Bodies: {self.num_dexhand_bodies}")
        print(f"Num dexhand DOFs: {self.num_dexhand_dofs}")

        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.dexhand_dof_lower_limits = []
        self.dexhand_dof_upper_limits = []
        self._dexhand_effort_limits = []
        self._dexhand_dof_speed_limits = []
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = dexhand_dof_stiffness[i]
            dexhand_dof_props["damping"][i] = dexhand_dof_damping[i]

            self.dexhand_dof_lower_limits.append(dexhand_dof_props["lower"][i])
            self.dexhand_dof_upper_limits.append(dexhand_dof_props["upper"][i])
            self._dexhand_effort_limits.append(dexhand_dof_props["effort"][i])
            self._dexhand_dof_speed_limits.append(dexhand_dof_props["velocity"][i])

        self.dexhand_dof_lower_limits = torch.tensor(self.dexhand_dof_lower_limits, device=self.sim_device)
        self.dexhand_dof_upper_limits = torch.tensor(self.dexhand_dof_upper_limits, device=self.sim_device)
        self._dexhand_effort_limits = torch.tensor(self._dexhand_effort_limits, device=self.sim_device)
        self._dexhand_dof_speed_limits = torch.tensor(self._dexhand_dof_speed_limits, device=self.sim_device)

                                
        num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        num_dexhand_shapes = self.gym.get_asset_rigid_shape_count(dexhand_asset)

        self.dexhand_rs = []
        self.envs = []

        assert len(self.dataIndices) == 1 or not self.rollout_state_init, "rollout_state_init only works with one data"

        dataset_list = list(set([ManipDataFactory.dataset_type(data_idx) for data_idx in self.dataIndices]))

        def segment_data(k):
            todo_list = self.dataIndices
            idx = todo_list[k % len(todo_list)]
            return self.demo_dataset_dict[ManipDataFactory.dataset_type(idx)][idx]

        self.demo_data = [segment_data(i) for i in tqdm(range(self.num_envs))]
        self.demo_data = self.pack_data(self.demo_data)

                             
        self.manip_obj_mass = []
        self.manip_obj_com = []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
                                 
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            current_asset, sum_rigid_body_count, sum_rigid_shape_count, obj_scale, obj_mass = self._create_obj_assets(i)
            max_agg_bodies = (
                num_dexhand_bodies
                + 1
                + sum_rigid_body_count
                + (5 + (0 + self.dexhand.n_bodies if not self.headless else 0))
            )               
            max_agg_shapes = (
                num_dexhand_shapes
                + 1
                + sum_rigid_shape_count
                + (5 + (0 + self.dexhand.n_bodies if not self.headless else 0))
                + (1 if self._record else 0)
            )
            print("max_agg_bodies",max_agg_bodies)
            print("max_agg_shapes",max_agg_shapes)
                                                                                         
                                                                   
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

                                               
            if self.camera_handlers is not None:
                self.camera_handlers.append(
                    self.create_camera(
                        env=env_ptr,
                        isaac_gym=self.gym,
                    )
                )

                              
            dexhand_actor = self.gym.create_actor(
                env_ptr,
                dexhand_asset,
                self.dexhand_pose,
                "dexhand",
                i,
                (1 if self.dexhand.self_collision else 0),                                            
            )
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_actor)
            self.gym.set_actor_dof_properties(env_ptr, dexhand_actor, dexhand_dof_props)

                                        
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)
            table_props[0].friction = 0.1                                      
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
                                               
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            self.obj_handle, _ = self._create_obj_actor(
                env_ptr, i, current_asset
            )                                           
            self.gym.set_actor_scale(env_ptr, self.obj_handle, obj_scale)
            obj_props = self.gym.get_actor_rigid_body_properties(env_ptr, self.obj_handle)
            obj_props[0].mass = min(0.5, obj_props[0].mass)                                              
                                        
            if obj_mass is not None:
                obj_props[0].mass = obj_mass

                                                                                    
                                                                                                                   
            self.gym.set_actor_rigid_body_properties(env_ptr, self.obj_handle, obj_props)
            self.manip_obj_mass.append(obj_props[0].mass)
            self.manip_obj_com.append(torch.tensor([obj_props[0].com.x, obj_props[0].com.y, obj_props[0].com.z]))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

                                            
            self.envs.append(env_ptr)
            self.dexhand_rs.append(dexhand_actor)

        self.manip_obj_mass = torch.tensor(self.manip_obj_mass, device=self.device)
        self.manip_obj_com = torch.stack(self.manip_obj_com, dim=0).to(self.device)

                    
        self.init_data()

    def init_data(self):
                           
        env_ptr = self.envs[0]
        dexhand_handle = self.gym.find_actor_handle(env_ptr, "dexhand")
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.body_names
        }
        self.dexhand_cf_weights = {
            k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) for k in self.dexhand.body_names
        }
                        
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

                              
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        print("self._root_state.shape:", self._root_state.shape)
                                                          
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]

                                 
        if not self.headless:

            self.mano_joint_points = [
                self._root_state[:, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :]
                for i in range(self.dexhand.n_bodies)
            ]
               

        self._manip_obj_handle = self.gym.find_actor_handle(env_ptr, "manip_obj")
        self._manip_obj_root_state = self._root_state[:, self._manip_obj_handle, :]
        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.num_envs, -1)
        self._manip_obj_rigid_body_handle = self.gym.find_actor_rigid_body_handle(
            env_ptr, self._manip_obj_handle, "base"
        )
        self._manip_obj_cf = self.net_cf[:, self._manip_obj_rigid_body_handle, :]
        print("self._manip_obj_handle",self._manip_obj_handle)
        print("self._manip_obj_root_state.shape",self._manip_obj_root_state.shape)
        print("self._manip_obj_rigid_body_handle",self._manip_obj_rigid_body_handle)
        print("self.net_cf.shape:", self.net_cf.shape)
        print("self._manip_obj_cf.shape:", self._manip_obj_cf.shape)
                                                        
                                  
                                                              
                                              
                                                   
                                                      


        self.dexhand_root_state = self._root_state[:, dexhand_handle, :]

        self.apply_forces = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curr_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        if self.use_pid_control:
            self.prev_pos_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.prev_rot_error = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.pos_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
            self.rot_error_integral = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

                            
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

                            
        self._global_dexhand_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        self._global_manip_obj_indices = torch.tensor(
            [self.gym.find_actor_index(env, "manip_obj", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.num_envs, -1)

        CONTACT_HISTORY_LEN = 3
        self.tips_contact_history = torch.ones(
            self.num_envs, CONTACT_HISTORY_LEN, len(self.dexhand.contact_body_names), device=self.device
        ).bool()
        
                  
        self.collected_metrics = {
            'E_r': [],            
            'E_t': [],            
            'E_j': [],              
            'E_ft': []              
        }
                  
        self.fingertip_indices = self._get_fingertip_indices()

    def _get_fingertip_indices(self):
        weight_idx = self.dexhand.weight_idx
        fingertip_keys = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
        fingertips = []
        for key in fingertip_keys:
            if key in weight_idx:
                fingertips.extend([k - 1 for k in weight_idx[key]])                     
        return fingertips

    def compute_error_metrics(self, current_states, target_states):
                       
        current_eef_quat = current_states["base_state"][:, 3:7]                
        target_eef_quat = target_states["wrist_quat"]
        
                                  
        current_quat_xyzw = current_eef_quat[:, [1, 2, 3, 0]]
        target_quat_xyzw = target_eef_quat[:, [1, 2, 3, 0]]
        
        current_rotmat = quat_to_rotmat(current_quat_xyzw)
        target_rotmat = quat_to_rotmat(target_quat_xyzw)
        
        res_rotmat = target_rotmat @ current_rotmat.transpose(-1, -2)
        res_aa = rotmat_to_aa(res_rotmat)
        diff_angle = torch.norm(res_aa, dim=-1)
        diff_angle = torch.min(diff_angle, 2 * np.pi - diff_angle)
        E_r = diff_angle / np.pi * 180
        
                       
        current_eef_pos = current_states["base_state"][:, :3]
        target_eef_pos = target_states["wrist_pos"]
        E_t = torch.norm(target_eef_pos - current_eef_pos, dim=-1)
        
                         
        current_joints_pos = current_states["joints_state"][:, 1:, :3]             
        target_joints_pos = target_states["joints_pos"]
        joint_diffs = torch.norm(target_joints_pos - current_joints_pos, dim=-1)
        E_j = joint_diffs.mean(dim=-1)             
        
                          
        if len(self.fingertip_indices) > 0:
            fingertip_diffs = joint_diffs[:, self.fingertip_indices]
            E_ft = fingertip_diffs.mean(dim=-1)             
        else:
            E_ft = torch.zeros_like(E_j)
            
        return E_r, E_t, E_j, E_ft


    def get_metrics_summary(self):
        if not any(self.collected_metrics.values()):
            return None
            
        summary = {
            'E_r_mean': np.mean(self.collected_metrics['E_r']),
            'E_t_mean': np.mean(self.collected_metrics['E_t']),
            'E_j_mean': np.mean(self.collected_metrics['E_j']),
            'E_ft_mean': np.mean(self.collected_metrics['E_ft']),
            'num_successful_sequences': len(self.collected_metrics['E_r'])
        }
        
                               
        if len(self.collected_metrics['E_r']) > 0:
            print("=" * 50)
        
        return summary

    def pack_data(self, data):
        packed_data = {}
        packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        max_len = packed_data["seq_len"].max()
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"

        def fill_data(stack_data):
            for i in range(len(stack_data)):
                if len(stack_data[i]) < max_len:
                    stack_data[i] = torch.cat(
                        [
                            stack_data[i],
                            stack_data[i][-1]
                            .unsqueeze(0)
                            .repeat(max_len - len(stack_data[i]), *[1 for _ in stack_data[i].shape[1:]]),
                        ],
                        dim=0,
                    )
            return torch.stack(stack_data).squeeze()

        for k in data[0].keys():
            if k == "mano_joints" or k == "mano_joints_velocity":
                mano_joints = []
                for d in data:
                    joints_data = torch.concat(
                        [
                            d[k][self.dexhand.to_hand(j_name)[0]]
                            for j_name in self.dexhand.body_names
                            if self.dexhand.to_hand(j_name)[0] != "wrist"
                        ],
                        dim=-1,
                    )
                                                                 
                    target_joints = 27
                    current_joints = joints_data.shape[-1] // 3              
                    if current_joints < target_joints:
                        pad_size = (target_joints - current_joints) * 3
                        padding = torch.zeros(joints_data.shape[:-1] + (pad_size,), device=joints_data.device, dtype=joints_data.dtype)
                        joints_data = torch.concat([joints_data, padding], dim=-1)
                    mano_joints.append(joints_data)
                packed_data[k] = fill_data(mano_joints)
            elif type(data[0][k]) == torch.Tensor:
                stack_data = [d[k] for d in data]
                if k != "obj_verts":
                    packed_data[k] = fill_data(stack_data)
                else:
                    packed_data[k] = torch.stack(stack_data).squeeze()
            elif type(data[0][k]) == np.ndarray:
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]
        return packed_data

    def allocate_buffers(self):
                                                                                  
        super().allocate_buffers()

                           
        if not self.training:
            self.dump_fileds = {
                k: torch.zeros(
                    (self.num_envs, v),
                    device=self.device,
                    dtype=torch.float,
                )
                for k, v in self._prop_dump_info.items()
            }

    def _create_obj_assets(self, i):
        obj_id = self.demo_data["obj_id"][i]

        if obj_id in self.objs_assets:
            current_asset = self.objs_assets[obj_id]
        else:
            asset_options = gymapi.AssetOptions()
            asset_options.override_com = True
            asset_options.override_inertia = True
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.thickness = 0.001
            asset_options.max_linear_velocity = 50
            asset_options.max_angular_velocity = 100
            asset_options.fix_base_link = False
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params = gymapi.VhacdParams()
            asset_options.vhacd_params.resolution = 200000
            asset_options.density = 200                                                            
            current_asset = self.gym.load_asset(
                self.sim, *os.path.split(self.demo_data["obj_urdf_path"][i]), asset_options
            )

            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
            for element in rigid_shape_props_asset:
                element.friction = 2.0                                                                                                                                       
                element.rolling_friction = 0.05
                element.torsion_friction = 0.05
            self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)
            self.objs_assets[obj_id] = current_asset

                                                                    
        if obj_id in oakink2_obj_scale:
            scale = oakink2_obj_scale[obj_id]
        else:
            scale = 1.0

        if obj_id in oakink2_obj_mass:
            mass = oakink2_obj_mass[obj_id]
        else:
            mass = None

        sum_rigid_body_count = self.gym.get_asset_rigid_body_count(current_asset)
        sum_rigid_shape_count = self.gym.get_asset_rigid_shape_count(current_asset)
        return current_asset, sum_rigid_body_count, sum_rigid_shape_count, scale, mass

    def _create_obj_actor(self, env_ptr, i, current_asset):

        obj_transf = self.demo_data["obj_trajectory"][i][0]

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(obj_transf[0, 3], obj_transf[1, 3], obj_transf[2, 3])
        obj_aa = rotmat_to_aa(obj_transf[:3, :3])
        obj_aa_angle = torch.norm(obj_aa)
        obj_aa_axis = obj_aa / obj_aa_angle
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle)

                                               
        obj_actor = self.gym.create_actor(env_ptr, current_asset, pose, "manip_obj", i, 0)
        obj_index = self.gym.get_actor_index(env_ptr, obj_actor, gymapi.DOMAIN_SIM)

        scene_objs = self.demo_data["scene_objs"][i]
        scene_asset_options = gymapi.AssetOptions()
        scene_asset_options.fix_base_link = True

        for so_id, scene_obj in enumerate(scene_objs):
            scene_obj_type = scene_obj["obj"].type
            scene_obj_size = scene_obj["obj"].size
            scene_obj_pose = scene_obj["pose"]
            if scene_obj_type == "cube":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0],
                    scene_obj_size[1],
                    scene_obj_size[2],
                    scene_asset_options,
                )
                offset = np.eye(4)
                offset[:3, 3] = np.array(scene_obj_size) / 2
                scene_obj_pose = scene_obj_pose @ offset
            elif scene_obj_type == "cylinder":
                scene_asset = self.gym.create_box(
                    self.sim,
                    scene_obj_size[0] * 2,
                    scene_obj_size[0] * 2,
                    scene_obj_size[1],
                    scene_asset_options,
                )
            else:
                raise NotImplementedError
            scene_obj_pose = self.mujoco2gym_transf @ torch.tensor(
                scene_obj_pose, device=self.sim_device, dtype=torch.float32
            )
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(scene_obj_pose[0, 3], scene_obj_pose[1, 3], scene_obj_pose[2, 3])
            obj_aa = rotmat_to_aa(scene_obj_pose[:3, :3])
            obj_aa_angle = torch.norm(obj_aa)
            obj_aa_axis = obj_aa / obj_aa_angle
            pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(obj_aa_axis[0], obj_aa_axis[1], obj_aa_axis[2]), obj_aa_angle
            )
            self.gym.create_actor(env_ptr, scene_asset, pose, f"scene_obj_{so_id}", i, 0)
                                
        MAX_SCENE_OBJS = 5 + (0 if not self.headless else 0)
        for so_id in range(MAX_SCENE_OBJS - len(scene_objs)):
            scene_asset = self.gym.create_box(self.sim, 0.02, 0.04, 0.06, scene_asset_options)
                                                                                                          
            a = self.gym.create_actor(
                env_ptr,
                scene_asset,
                gymapi.Transform(),
                f"scene_obj_{so_id +  len(scene_objs)}",
                self.num_envs + 1,
                0b1,
            )
            c = [
                gymapi.Vec3(1, 1, 0.5),
                gymapi.Vec3(0.5, 1, 1),
                gymapi.Vec3(1, 0, 1),
                gymapi.Vec3(1, 1, 0),
                gymapi.Vec3(0, 1, 1),
                gymapi.Vec3(0, 0, 1),
                gymapi.Vec3(0, 1, 0),
                gymapi.Vec3(1, 0, 0),
            ][so_id + len(scene_objs)]
            self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

                                                                                       
        if not self.headless:
            for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
                joint_name = self.dexhand.to_hand(joint_name)[0]
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env_ptr, joint_point, gymapi.Transform(), f"mano_joint_{joint_vis_id}", self.num_envs + 1, 0b1
                )
                if "index" in joint_name:
                    inter_c = 70
                elif "middle" in joint_name:
                    inter_c = 130
                elif "ring" in joint_name:
                    inter_c = 190
                elif "pinky" in joint_name:
                    inter_c = 250
                elif "thumb" in joint_name:
                    inter_c = 10
                else:
                    inter_c = 0
                if "tip" in joint_name:
                    c = gymapi.Vec3(inter_c / 255, 200 / 255, 200 / 255)
                elif "proximal" in joint_name:
                    c = gymapi.Vec3(200 / 255, inter_c / 255, 200 / 255)
                elif "intermediate" in joint_name:
                    c = gymapi.Vec3(200 / 255, 200 / 255, inter_c / 255)
                else:
                    c = gymapi.Vec3(100 / 255, 150 / 255, 200 / 255)
                self.gym.set_rigid_body_color(env_ptr, a, 0, gymapi.MESH_VISUAL, c)

        return obj_actor, obj_index

    def _update_states(self):
        self.states.update(
            {
                "q": self._q[:, :],
                "cos_q": torch.cos(self._q[:, :]),
                "sin_q": torch.sin(self._q[:, :]),
                "dq": self._qd[:, :],
                "base_state": self._base_state[:, :],
            }
        )

        joints_state = torch.stack(
            [self._rigid_body_state[:, self.dexhand_handles[k], :][:, :10] for k in self.dexhand.body_names],
            dim=1,
        )
        
                            
        target_joints = 28
        current_joints = joints_state.shape[1]
        if current_joints < target_joints:
            pad_size = target_joints - current_joints
            padding = torch.zeros(joints_state.shape[0], pad_size, joints_state.shape[2], device=joints_state.device, dtype=joints_state.dtype)
            joints_state = torch.concat([joints_state, padding], dim=1)
        
        self.states["joints_state"] = joints_state

        self.states.update(
            {
                "manip_obj_pos": self._manip_obj_root_state[:, :3],
                "manip_obj_quat": self._manip_obj_root_state[:, 3:7],
                "manip_obj_vel": self._manip_obj_root_state[:, 7:10],
                "manip_obj_ang_vel": self._manip_obj_root_state[:, 10:],
            }
        )

    def _refresh(self):

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

                        
        self._update_states()

    def compute_reward(self, actions):
        target_state = {}
        max_length = torch.clip(self.demo_data["seq_len"], 0, self.max_episode_length).float()
        cur_idx = self.progress_buf
        cur_wrist_pos = self.demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_pos"] = cur_wrist_pos
        cur_wrist_rot = self.demo_data["wrist_rot"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot)[:, [1, 2, 3, 0]]

        target_state["wrist_vel"] = self.demo_data["wrist_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_ang_vel"] = self.demo_data["wrist_angular_velocity"][torch.arange(self.num_envs), cur_idx]

        target_state["tips_distance"] = self.demo_data["tips_distance"][torch.arange(self.num_envs), cur_idx]
                                                 
                                                       
                                                                                 
                      
                                                             
                                                 
                                                
                                              
                                                         
                                                                 
                                             
                             
                                                                       

                  
                                                           

        cur_joints_pos = self.demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx]
        target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)
        target_state["joints_vel"] = self.demo_data["mano_joints_velocity"][
            torch.arange(self.num_envs), cur_idx
        ].reshape(self.num_envs, -1, 3)

        cur_obj_transf = self.demo_data["obj_trajectory"][torch.arange(self.num_envs), cur_idx]
        target_state["manip_obj_pos"] = cur_obj_transf[:, :3, 3]
        target_state["manip_obj_quat"] = rotmat_to_quat(cur_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]

        target_state["manip_obj_vel"] = self.demo_data["obj_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["manip_obj_ang_vel"] = self.demo_data["obj_angular_velocity"][torch.arange(self.num_envs), cur_idx]

        target_state["tip_force"] = torch.stack(
            [self.net_cf[:, self.dexhand_handles[k], :] for k in self.dexhand.contact_body_names],
            axis=1,
        )
        self.tips_contact_history = torch.concat(
            [
                self.tips_contact_history[:, 1:],
                (torch.norm(target_state["tip_force"], dim=-1) > 0)[:, None],
            ],
            dim=1,
        )
        target_state["tip_contact_state"] = self.tips_contact_history

        power = torch.abs(torch.multiply(self.dof_force, self.states["dq"])).sum(dim=-1)
        target_state["power"] = power

        wrist_power = torch.abs(
            torch.sum(
                self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
                * self.states["base_state"][:, 7:10],
                dim=-1,
            )
        )                                    
        wrist_power += torch.abs(
            torch.sum(
                self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
                * self.states["base_state"][:, 10:],
                dim=-1,
            )
        )                               
        target_state["wrist_power"] = wrist_power

        if self.training:
            last_step = self.gym.get_frame_count(self.sim)
            if self.tighten_method == "None":
                scale_factor = 1.0
            elif self.tighten_method == "const":
                scale_factor = self.tighten_factor
            elif self.tighten_method == "linear_decay":
                scale_factor = 1 - (1 - self.tighten_factor) / self.tighten_steps * min(last_step, self.tighten_steps)
            elif self.tighten_method == "exp_decay":
                scale_factor = (np.e * 2) ** (-1 * last_step / self.tighten_steps) * (
                    1 - self.tighten_factor
                ) + self.tighten_factor
            elif self.tighten_method == "cos":
                scale_factor = (self.tighten_factor) + np.abs(
                    -1 * (1 - self.tighten_factor) * np.cos(last_step / self.tighten_steps * np.pi)
                ) * (2 ** (-1 * last_step / self.tighten_steps))
            else:
                raise NotImplementedError
        else:
            scale_factor = 1.0

        assert not self.headless or isinstance(compute_imitation_reward, torch.jit.ScriptFunction)

        if self.rollout_len is not None:
            max_length = torch.clamp(max_length, 0, self.rollout_len + self.rollout_begin + 3 + 1)

        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
            self.failure_buf[:],
            self.reward_dict,
            self.error_buf[:],
        ) = compute_imitation_reward(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            self.actions,
            self.states,
            target_state,
            max_length,
            scale_factor,
            self.dexhand.weight_idx,
            self.cfg["env"]["dexhand"],
        )
        self.total_rew_buf += self.rew_buf

    def compute_observations(self):    
        self._refresh()
        
                        
        should_debug = self.debug_obs
        
                        
        from ...utils.padding_utils import pad_joint_data, get_hand_embedding, pad_to_max_dim
        
                          
        k_max = self.cfg["env"].get("k_max", 22)             
        target_max_dim = self.cfg["env"].get("target_max_dim", 450)            
        privileged_max_dim = self.cfg["env"].get("privileged_max_dim", 59)            
        embedding_dim = self.cfg["env"].get("embedding_dim", 3)            
        
                          
        obs_values = []
        
        for ob in self._obs_keys:
            if ob == "base_state":
                obs_values.append(
                    torch.cat([torch.zeros_like(self.states[ob][:, :3]), self.states[ob][:, 3:]], dim=-1)
                )                          
            elif ob in ["q", "cos_q", "sin_q"]:
                             
                padded_data = pad_joint_data(self.states[ob], self.dexhand.n_dofs, k_max)
                obs_values.append(padded_data)
            else:
                obs_values.append(self.states[ob])
        
                          
        hand_embedding = get_hand_embedding(self.cfg["env"]["dexhand"], device=self.device)
        hand_embedding = hand_embedding.repeat(self.num_envs, 1)
        obs_values.append(hand_embedding)
        
                                                                           
        self.obs_dict["proprioception"][:] = torch.cat(obs_values, dim=-1)
        if should_debug:
            pass
        
                          
        if len(self._privileged_obs_keys) > 0:
            pri_obs_values = []
                                      
            if should_debug:
                pass
            for ob in self._privileged_obs_keys:
                if ob == "manip_obj_pos":
                    pri_obs_values.append(self.states[ob] - self.states["base_state"][:, :3])
                    if should_debug:
                        print(f"  - {ob}: {pri_obs_values[-1].shape}")
                elif ob == "manip_obj_com":
                    cur_com_pos = (
                        quat_to_rotmat(self.states["manip_obj_quat"][:, [1, 2, 3, 0]])
                        @ self.manip_obj_com.unsqueeze(-1)
                    ).squeeze(-1) + self.states["manip_obj_pos"]
                    pri_obs_values.append(cur_com_pos - self.states["base_state"][:, :3])
                    if should_debug:
                        print(f"  - {ob}: {pri_obs_values[-1].shape}")
                elif ob == "manip_obj_weight":
                    prop = self.gym.get_sim_params(self.sim)
                    pri_obs_values.append((self.manip_obj_mass * -1 * prop.gravity.z).unsqueeze(-1))
                    if should_debug:
                        print(f"  - {ob}: {pri_obs_values[-1].shape}")
                elif ob == "tip_force":
                    tip_force = torch.stack(
                        [self.net_cf[:, self.dexhand_handles[k], :] for k in self.dexhand.contact_body_names],
                        axis=1,
                    )
                    tip_force = torch.cat(
                        [tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1
                    )                       
                    pri_obs_values.append(tip_force.reshape(self.num_envs, -1))
                    if should_debug:
                        print(f"  - {ob}: {pri_obs_values[-1].shape}")
                elif ob == "dq":
                                
                    padded_dq = pad_joint_data(self.states["dq"], self.dexhand.n_dofs, k_max)
                    pri_obs_values.append(padded_dq)
                    if should_debug:
                        pass
                else:
                    pri_obs_values.append(self.states[ob])
                    if should_debug:
                        print(f"  - {ob}: {pri_obs_values[-1].shape}")
            
                            
            privileged_obs = torch.cat(pri_obs_values, dim=-1)
            privileged_obs = pad_to_max_dim(privileged_obs, privileged_max_dim)
            self.obs_dict["privileged"][:] = privileged_obs
            if should_debug:
                pass

        next_target_state = {}

        cur_idx = self.progress_buf + 1
        cur_idx = torch.clamp(cur_idx, torch.zeros_like(self.demo_data["seq_len"]), self.demo_data["seq_len"] - 1)

        cur_idx = torch.stack(
            [cur_idx + t for t in range(self.obs_future_length)], dim=-1
        )                                 
        nE, nT = self.demo_data["wrist_pos"].shape[:2]
        nF = self.obs_future_length

        def indicing(data, idx):
            assert data.shape[0] == nE and data.shape[1] == nT
            remaining_shape = data.shape[2:]
            expanded_idx = idx
            for _ in remaining_shape:
                expanded_idx = expanded_idx.unsqueeze(-1)
            expanded_idx = expanded_idx.expand(-1, -1, *remaining_shape)
            return torch.gather(data, 1, expanded_idx)

        target_wrist_pos = indicing(self.demo_data["wrist_pos"], cur_idx)             
        cur_wrist_pos = self.states["base_state"][:, :3]          
        next_target_state["delta_wrist_pos"] = (target_wrist_pos - cur_wrist_pos[:, None]).reshape(nE, -1)

        target_wrist_vel = indicing(self.demo_data["wrist_velocity"], cur_idx)
        cur_wrist_vel = self.states["base_state"][:, 7:10]
        next_target_state["wrist_vel"] = target_wrist_vel.reshape(nE, -1)
        next_target_state["delta_wrist_vel"] = (target_wrist_vel - cur_wrist_vel[:, None]).reshape(nE, -1)

        target_wrist_rot = indicing(self.demo_data["wrist_rot"], cur_idx)
        cur_wrist_rot = self.states["base_state"][:, 3:7]

        next_target_state["wrist_quat"] = aa_to_quat(target_wrist_rot.reshape(nE * nF, -1))[:, [1, 2, 3, 0]]
        next_target_state["delta_wrist_quat"] = quat_mul(
            cur_wrist_rot[:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["wrist_quat"]),
        ).reshape(nE, -1)
        next_target_state["wrist_quat"] = next_target_state["wrist_quat"].reshape(nE, -1)

        target_wrist_ang_vel = indicing(self.demo_data["wrist_angular_velocity"], cur_idx)
        cur_wrist_ang_vel = self.states["base_state"][:, 10:13]
        next_target_state["wrist_ang_vel"] = target_wrist_ang_vel.reshape(nE, -1)
        next_target_state["delta_wrist_ang_vel"] = (target_wrist_ang_vel - cur_wrist_ang_vel[:, None]).reshape(nE, -1)

        target_joints_pos = indicing(self.demo_data["mano_joints"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_pos = self.states["joints_state"][:, 1:, :3]                       
        next_target_state["delta_joints_pos"] = (target_joints_pos - cur_joint_pos[:, None]).reshape(self.num_envs, -1)

        target_joints_vel = indicing(self.demo_data["mano_joints_velocity"], cur_idx).reshape(nE, nF, -1, 3)
        cur_joint_vel = self.states["joints_state"][:, 1:, 7:10]                       
        next_target_state["joints_vel"] = target_joints_vel.reshape(self.num_envs, -1)
        next_target_state["delta_joints_vel"] = (target_joints_vel - cur_joint_vel[:, None]).reshape(self.num_envs, -1)

        target_obj_transf = indicing(self.demo_data["obj_trajectory"], cur_idx)
        target_obj_transf = target_obj_transf.reshape(nE * nF, 4, 4)
        next_target_state["delta_manip_obj_pos"] = (
            target_obj_transf[:, :3, 3].reshape(nE, nF, -1) - self.states["manip_obj_pos"][:, None]
        ).reshape(nE, -1)

        target_obj_vel = indicing(self.demo_data["obj_velocity"], cur_idx)
        cur_obj_vel = self.states["manip_obj_vel"]
        next_target_state["manip_obj_vel"] = target_obj_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_vel"] = (target_obj_vel - cur_obj_vel[:, None]).reshape(nE, -1)

        next_target_state["manip_obj_quat"] = rotmat_to_quat(target_obj_transf[:, :3, :3])[:, [1, 2, 3, 0]]
        next_target_state["delta_manip_obj_quat"] = quat_mul(
            self.states["manip_obj_quat"][:, None].repeat(1, nF, 1).reshape(nE * nF, -1),
            quat_conjugate(next_target_state["manip_obj_quat"]),
        ).reshape(nE, -1)
        next_target_state["manip_obj_quat"] = next_target_state["manip_obj_quat"].reshape(nE, -1)

        target_obj_ang_vel = indicing(self.demo_data["obj_angular_velocity"], cur_idx)
        cur_obj_ang_vel = self.states["manip_obj_ang_vel"]
        next_target_state["manip_obj_ang_vel"] = target_obj_ang_vel.reshape(nE, -1)
        next_target_state["delta_manip_obj_ang_vel"] = (target_obj_ang_vel - cur_obj_ang_vel[:, None]).reshape(nE, -1)

        next_target_state["obj_to_joints"] = torch.norm(
            self.states["manip_obj_pos"][:, None] - self.states["joints_state"][:, :, :3], dim=-1
        ).reshape(self.num_envs, -1)

        next_target_state["gt_tips_distance"] = indicing(self.demo_data["tips_distance"], cur_idx).reshape(nE, -1)

        next_target_state["bps"] = self.obj_bps
                               
        if not hasattr(self, '_target_dims_printed') and should_debug:
            total_dim = 0
            obs_keys = [
                "delta_wrist_pos", "wrist_vel", "delta_wrist_vel", "wrist_quat", "delta_wrist_quat",
                "wrist_ang_vel", "delta_wrist_ang_vel", "delta_joints_pos", "joints_vel", "delta_joints_vel",
                "delta_manip_obj_pos", "manip_obj_vel", "delta_manip_obj_vel", "manip_obj_quat", "delta_manip_obj_quat",
                "manip_obj_ang_vel", "delta_manip_obj_ang_vel", "obj_to_joints", "gt_tips_distance", "bps",
            ]
            for ob in obs_keys:
                if ob in next_target_state:
                    dim = next_target_state[ob].shape[1]
                    total_dim += dim
                    print(f"  - {ob:<25}: {next_target_state[ob].shape} (dim: {dim})")
                else:
                    print(f"  - {ob:<25}: Not found in next_target_state")
            self._target_dims_printed = True

                          
                    
        target_obs = torch.cat(
            [
                next_target_state[ob]
                for ob in [                                                
                    "delta_wrist_pos",
                    "wrist_vel",
                    "delta_wrist_vel",
                    "wrist_quat",
                    "delta_wrist_quat",
                    "wrist_ang_vel",
                    "delta_wrist_ang_vel",
                    "delta_joints_pos",
                    "joints_vel",
                    "delta_joints_vel",
                    "delta_manip_obj_pos",
                    "manip_obj_vel",
                    "delta_manip_obj_vel",
                    "manip_obj_quat",
                    "delta_manip_obj_quat",
                    "manip_obj_ang_vel",
                    "delta_manip_obj_ang_vel",
                    "obj_to_joints",
                    "gt_tips_distance",
                    "bps",
                ]
            ],
            dim=-1,
        )
        
                     
        target_obs = pad_to_max_dim(target_obs, target_max_dim)
        self.obs_dict["target"][:] = target_obs
        if should_debug:
            pass

                               
        if should_debug:
            total_actor_obs = self.obs_dict["proprioception"].shape[1] + self.obs_dict["target"].shape[1]
            if "privileged" in self.obs_dict:
                total_critic_obs = total_actor_obs + self.obs_dict["privileged"].shape[1]
            else:
                pass

                               
                     
        if not self.training:
            for prop_name in self._prop_dump_info.keys():
                if prop_name == "state_rh" or prop_name == "state_lh":
                    self.dump_fileds[prop_name][:] = self.states["base_state"]
                elif prop_name == "state_manip_obj_rh" or prop_name == "state_manip_obj_lh":
                    self.dump_fileds[prop_name][:] = self._manip_obj_root_state
                elif prop_name == "joint_state_rh" or prop_name == "joint_state_lh":
                    self.dump_fileds[prop_name][:] = torch.stack(
                        [self._rigid_body_state[:, self.dexhand_handles[k], :] for k in self.dexhand.body_names],
                        dim=1,
                    ).reshape(self.num_envs, -1)
                elif prop_name == "tip_force_rh" or prop_name == "tip_force_lh":
                    tip_force = torch.stack(
                        [self.net_cf[:, self.dexhand_handles[k], :] for k in self.dexhand.contact_body_names],
                        axis=1,
                    )
                    self.dump_fileds[prop_name][:] = tip_force.reshape(self.num_envs, -1)
                elif prop_name == "q_rh" or prop_name == "q_lh":
                    self.dump_fileds[prop_name][:] = self.states["q"][:]
                elif prop_name == "dq_rh" or prop_name == "dq_lh":
                    self.dump_fileds[prop_name][:] = self.states["dq"][:]
                elif prop_name == "reward":
                    self.dump_fileds[prop_name][:] = self.rew_buf.reshape(self.num_envs, -1).detach()
                else:           
                    self.dump_fileds[prop_name][:] = self.states[prop_name][:]
        return self.obs_dict

    def _reset_default(self, env_ids):
        if self.random_state_init:
            if self.rollout_begin is not None:
                seq_idx = (
                    torch.floor(
                        self.rollout_len * 0.98 * torch.rand_like(self.demo_data["seq_len"][env_ids].float())
                    ).long()
                    + self.rollout_begin
                )
                seq_idx = torch.clamp(
                    seq_idx,
                    torch.zeros(1, device=self.device).long(),
                    torch.floor(self.demo_data["seq_len"][env_ids] * 0.98).long(),
                )
            else:
                seq_idx = torch.floor(
                    self.demo_data["seq_len"][env_ids]
                    * 0.98
                    * torch.rand_like(self.demo_data["seq_len"][env_ids].float())
                ).long()
        else:
            if self.rollout_begin is not None:
                seq_idx = self.rollout_begin * torch.ones_like(self.demo_data["seq_len"][env_ids].long())
            else:
                seq_idx = torch.zeros_like(self.demo_data["seq_len"][env_ids].long())

        dof_pos = self.demo_data["opt_dof_pos"][env_ids, seq_idx]
        dof_pos = torch_jit_utils.tensor_clamp(
            dof_pos,
            self.dexhand_dof_lower_limits.unsqueeze(0),
            self.dexhand_dof_upper_limits.unsqueeze(0),
        )
        dof_vel = self.demo_data["opt_dof_velocity"][env_ids, seq_idx]
        dof_vel = torch_jit_utils.tensor_clamp(
            dof_vel,
            -1 * self._dexhand_dof_speed_limits.unsqueeze(0),
            self._dexhand_dof_speed_limits.unsqueeze(0),
        )

        opt_wrist_pos = self.demo_data["opt_wrist_pos"][env_ids, seq_idx]
        opt_wrist_rot = aa_to_quat(self.demo_data["opt_wrist_rot"][env_ids, seq_idx])
        opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]

        opt_wrist_vel = self.demo_data["opt_wrist_velocity"][env_ids, seq_idx]
        opt_wrist_ang_vel = self.demo_data["opt_wrist_angular_velocity"][env_ids, seq_idx]

        opt_hand_pose_vel = torch.concat([opt_wrist_pos, opt_wrist_rot, opt_wrist_vel, opt_wrist_ang_vel], dim=-1)

        self._base_state[env_ids, :] = opt_hand_pose_vel

        self._q[env_ids, :] = dof_pos
        self._qd[env_ids, :] = dof_vel
        self._pos_control[env_ids, :] = dof_pos

                         
        obj_pos_init = self.demo_data["obj_trajectory"][env_ids, seq_idx, :3, 3]
        obj_rot_init = self.demo_data["obj_trajectory"][env_ids, seq_idx, :3, :3]
        obj_rot_init = rotmat_to_quat(obj_rot_init)
                                      
        obj_rot_init = obj_rot_init[:, [1, 2, 3, 0]]

        obj_vel = self.demo_data["obj_velocity"][env_ids, seq_idx]
        obj_ang_vel = self.demo_data["obj_angular_velocity"][env_ids, seq_idx]

        self._manip_obj_root_state[env_ids, :3] = obj_pos_init
        self._manip_obj_root_state[env_ids, 3:7] = obj_rot_init
        self._manip_obj_root_state[env_ids, 7:10] = obj_vel
        self._manip_obj_root_state[env_ids, 10:13] = obj_ang_vel

        dexhand_multi_env_ids_int32 = self._global_dexhand_indices[env_ids].flatten()
                                                                          
        manip_obj_multi_env_ids_int32 = self._global_manip_obj_indices[env_ids].flatten()
                                                                              

        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
            len(torch.concat([dexhand_multi_env_ids_int32, manip_obj_multi_env_ids_int32])),
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._pos_control),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )

        self.progress_buf[env_ids] = seq_idx
        self.running_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0
        self.error_buf[env_ids] = 0
        self.total_rew_buf[env_ids] = 0
        self.apply_forces[env_ids] = 0
        self.apply_torque[env_ids] = 0
        self.curr_targets[env_ids] = 0
        self.prev_targets[env_ids] = 0
        


        if self.use_pid_control:
            self.prev_pos_error[env_ids] = 0
            self.prev_rot_error[env_ids] = 0
            self.pos_error_integral[env_ids] = 0
            self.rot_error_integral[env_ids] = 0

        self.tips_contact_history[env_ids] = torch.ones_like(self.tips_contact_history[env_ids]).bool()

    def reset_idx(self, env_ids):
        self._refresh()
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        last_step = self.gym.get_frame_count(self.sim)
        if self.training and len(self.dataIndices) == 1 and last_step >= self.tighten_steps:
            running_steps = self.running_progress_buf[env_ids] - 1
            max_running_steps, max_running_idx = running_steps.max(dim=0)
            max_running_env_id = env_ids[max_running_idx]
            if max_running_steps > self.best_rollout_len:
                self.best_rollout_len = max_running_steps
                self.best_rollout_begin = self.progress_buf[max_running_env_id] - 1 - max_running_steps

        self._reset_default(env_ids)

    def reset_done(self):
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
            self.compute_observations()

        if not self.dict_obs_cls:
            self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

                                     
            if self.num_states > 0:
                self.obs_dict["states"] = self.get_state()

        return self.obs_dict, done_env_ids

    def step(self, actions):
        obs, rew, done, info = super().step(actions)
        info["reward_dict"] = self.reward_dict
        info["total_rewards"] = self.total_rew_buf
        info["total_steps"] = self.progress_buf
        return obs, rew, done, info

    def pre_physics_step(self, actions):
                          
        should_debug = self.debug_obs
        if should_debug:
            pass

                                 
        if not self.headless:

            cur_idx = self.progress_buf

            self.gym.clear_lines(self.viewer)

            cur_wrist_pos = self.demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]

            cur_mano_joint_pos = self.demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx].reshape(
                self.num_envs, -1, 3
            )
            cur_mano_joint_pos = torch.concat([cur_wrist_pos[:, None], cur_mano_joint_pos], dim=1)
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = cur_mano_joint_pos[:, k]
            for env_id, env_ptr in enumerate(self.envs):
                for k in self.dexhand.body_names:
                    self.set_force_vis(
                        env_ptr, k, torch.norm(self.net_cf[env_id, self.dexhand_handles[k]], dim=-1) != 0
                    )



                                                                     
                                                                                                        
                                                             
                                                                                                                 
                                        
                                                                             

                color = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)
                                                                           
                dexhand_joints = []
                for body_name in self.dexhand.body_names:
                    hand_body = self.dexhand.to_hand(body_name)[0]
                    if hand_body == "wrist":
                        dexhand_joints.append(cur_wrist_pos[env_id])
                    else:
                                                                                
                                                                                              
                        joint_idx = 0
                        for j_name in self.dexhand.body_names:
                            if self.dexhand.to_hand(j_name)[0] != "wrist":
                                if j_name == body_name:
                                    break
                                joint_idx += 1
                        if joint_idx < cur_mano_joint_pos.shape[1] - 1:                                          
                            dexhand_joints.append(cur_mano_joint_pos[env_id, joint_idx + 1])                    
                        else:
                                                                           
                            dexhand_joints.append(cur_wrist_pos[env_id])
                
                                                                  
                                                                        
                                                                                  

                                 
        
                          
                  
        k_max = self.cfg["env"].get("k_max", 22)             
        total_action_dim = 6 + k_max                  
        
                         
        res_split_idx = total_action_dim      
        
        if should_debug:
            pass
        
                                
        base_action = actions[:, :res_split_idx]                     
        residual_action = actions[:, res_split_idx:] * 2              
        
        if should_debug:
            pass
        
                        
        root_control_dim = 9 if self.use_pid_control else 6
        actual_dofs = self.num_dexhand_dofs                          
        
                              
        base_joint_action = base_action[:, root_control_dim:root_control_dim + actual_dofs]
        residual_joint_action = residual_action[:, 6:6 + actual_dofs] 
        
        if should_debug:
            pass
        
                    
                                                                          

        dof_pos = 0.0 * base_joint_action + residual_joint_action
        dof_pos = torch.clamp(dof_pos, -1, 1)
        
        if should_debug:
            pass

        curr_act_moving_average = self.act_moving_average

        self.curr_targets = torch_jit_utils.scale(
            dof_pos,                             
            self.dexhand_dof_lower_limits,
            self.dexhand_dof_upper_limits,
        )
        self.curr_targets = (
            curr_act_moving_average * self.curr_targets + (1.0 - curr_act_moving_average) * self.prev_targets
        )
        self.curr_targets = torch_jit_utils.tensor_clamp(
            self.curr_targets,
            self.dexhand_dof_lower_limits,
            self.dexhand_dof_upper_limits,
        )

        if self.use_pid_control:
            position_error = base_action[:, 0:3]
            self.pos_error_integral += position_error * self.dt
            self.pos_error_integral = torch.clamp(self.pos_error_integral, -1, 1)
            pos_derivative = (position_error - self.prev_pos_error) / self.dt
            force = self.Kp_pos * position_error + self.Ki_pos * self.pos_error_integral + self.Kd_pos * pos_derivative
            self.prev_pos_error = position_error

            force = 0.0 * force + residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
            )

            rotation_error = base_action[:, 3:root_control_dim]
            rotation_error = rot6d_to_aa(rotation_error)
            self.rot_error_integral += rotation_error * self.dt
            self.rot_error_integral = torch.clamp(self.rot_error_integral, -1, 1)
            rot_derivative = (rotation_error - self.prev_rot_error) / self.dt
            torque = self.Kp_rot * rotation_error + self.Ki_rot * self.rot_error_integral + self.Kd_rot * rot_derivative
            self.prev_rot_error = rotation_error

            torque = 0.0 * torque + residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
            )

        else:
            force = 0.0 * (base_action[:, 0:3] * self.dt * self.translation_scale * 500) + (
                residual_action[:, 0:3] * self.dt * self.translation_scale * 500
            )
            torque = 0.0 * (base_action[:, 3:6] * self.dt * self.orientation_scale * 200) + (
                residual_action[:, 3:6] * self.dt * self.orientation_scale * 200
            )

            self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * force
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
            )
            self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :] = (
                curr_act_moving_average * torque
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[:, self.dexhand_handles[self.dexhand.to_dex("wrist")[0]], :]
            )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        self.prev_targets[:] = self.curr_targets[:]
        self._pos_control[:] = self.prev_targets[:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        
        if should_debug:
            pass

    def post_physics_step(self):

        self.compute_observations()        
        self.compute_reward(self.actions)

                          
        if hasattr(self, 'success_buf') and torch.any(self.success_buf):
            succeeded_envs = torch.nonzero(self.success_buf, as_tuple=True)[0]
            
            if len(succeeded_envs) > 0:
                            
                cur_idx = self.progress_buf
                target_wrist_pos = self.demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]
                cur_wrist_rot = self.demo_data["wrist_rot"][torch.arange(self.num_envs), cur_idx]
                target_wrist_quat = aa_to_quat(cur_wrist_rot)[:, [1, 2, 3, 0]]
                cur_joints_pos = self.demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx]
                target_joints_pos = cur_joints_pos.reshape(self.num_envs, -1, 3)
                
                              
                for env_id in succeeded_envs:
                    env_id = env_id.item()
                    try:
                              
                        current_eef_pos = self.states["base_state"][env_id, :3].unsqueeze(0)
                        current_eef_quat = self.states["base_state"][env_id, 3:7].unsqueeze(0)
                        current_joints_pos = self.states["joints_state"][env_id, 1:, :3].unsqueeze(0)
                        
                              
                        target_pos = target_wrist_pos[env_id].unsqueeze(0)
                        target_quat = target_wrist_quat[env_id].unsqueeze(0)
                        target_joints = target_joints_pos[env_id].unsqueeze(0)
                        
                                  
                                   
                        current_quat_xyzw = current_eef_quat[:, [1, 2, 3, 0]]
                        target_quat_xyzw = target_quat[:, [1, 2, 3, 0]]
                        current_rotmat = quat_to_rotmat(current_quat_xyzw)
                        target_rotmat = quat_to_rotmat(target_quat_xyzw)
                        res_rotmat = target_rotmat @ current_rotmat.transpose(-1, -2)
                        res_aa = rotmat_to_aa(res_rotmat)
                        diff_angle = torch.norm(res_aa, dim=-1)
                        diff_angle = torch.min(diff_angle, 2 * np.pi - diff_angle)
                        E_r = (diff_angle / np.pi * 180).item()
                        
                                   
                        E_t = torch.norm(target_pos - current_eef_pos, dim=-1).item()
                        
                                     
                        joint_diffs = torch.norm(target_joints - current_joints_pos, dim=-1)
                        E_j = joint_diffs.mean().item()
                        
                                      
                        if len(self.fingertip_indices) > 0:
                            fingertip_diffs = joint_diffs[0, self.fingertip_indices]
                            E_ft = fingertip_diffs.mean().item()
                        else:
                            E_ft = E_j
                        
                              
                        self.collected_metrics['E_r'].append(E_r)
                        self.collected_metrics['E_t'].append(E_t)
                        self.collected_metrics['E_j'].append(E_j)
                        self.collected_metrics['E_ft'].append(E_ft)
                        
                                       
                        seq_num = len(self.collected_metrics['E_r'])
                        print("=" * 40)
                        
                    except Exception as e:
                        continue
                
                                       
                                            

        self.progress_buf += 1
        self.running_progress_buf += 1
        self.randomize_buf += 1

    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
        """
        Only create front camera for view purpose
        """
        if self._record:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 1280
            camera_cfg.height = 720
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.80, -0.00, 0.7)
            cam_target = gymapi.Vec3(-1, -0.00, 0.3)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        else:
            camera_cfg = gymapi.CameraProperties()
            camera_cfg.enable_tensors = True
            camera_cfg.width = 320
            camera_cfg.height = 180
            camera_cfg.horizontal_fov = 69.4

            camera = isaac_gym.create_camera_sensor(env, camera_cfg)
            cam_pos = gymapi.Vec3(0.97, 0, 0.74)
            cam_target = gymapi.Vec3(-1, 0, 0.5)
            isaac_gym.set_camera_location(camera, env, cam_pos, cam_target)
        return camera

    def set_force_vis(self, env_ptr, part_k, has_force):
        self.gym.set_rigid_body_color(
            env_ptr,
            0,
            self.dexhand_handles[part_k],
            gymapi.MESH_VISUAL,
            (
                gymapi.Vec3(
                    1.0,
                    1,
                    1,
                )
                if has_force
                else gymapi.Vec3(1.0, 1.0, 1.0)
            ),
        )


@torch.jit.script
def quat_to_angle_axis(q):
                                             
                                                          
                          
    min_theta = 1e-5
    qx, qy, qz, qw = 0, 1, 2, 3

    sin_theta = torch.sqrt(1 - q[..., qw] * q[..., qw])
    angle = 2 * torch.acos(q[..., qw])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., qx:qw] / sin_theta_expand

    mask = torch.abs(sin_theta) > min_theta
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1

    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return angle, axis


@torch.jit.script
def compute_imitation_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
    dexterous_hand: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

                                                                                                                                                                                                

                              
    current_eef_pos = states["base_state"][:, :3]
    current_eef_quat = states["base_state"][:, 3:7]

    target_eef_pos = target_states["wrist_pos"]
    target_eef_quat = target_states["wrist_quat"]
    diff_eef_pos = target_eef_pos - current_eef_pos
    diff_eef_pos_dist = torch.norm(diff_eef_pos, dim=-1)

    current_eef_vel = states["base_state"][:, 7:10]
    current_eef_ang_vel = states["base_state"][:, 10:13]
    target_eef_vel = target_states["wrist_vel"]
    target_eef_ang_vel = target_states["wrist_ang_vel"]

    diff_eef_vel = target_eef_vel - current_eef_vel
    diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

    joints_pos = states["joints_state"][:, 1:, :3]
    target_joints_pos = target_states["joints_pos"]
    diff_joints_pos = target_joints_pos - joints_pos
    diff_joints_pos_dist = torch.norm(diff_joints_pos, dim=-1)

                                                    
                                                                         
    diff_thumb_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["thumb_tip"]]].mean(dim=-1)
    diff_index_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["index_tip"]]].mean(dim=-1)
    diff_middle_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["middle_tip"]]].mean(dim=-1)
    diff_ring_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["ring_tip"]]].mean(dim=-1)
    diff_pinky_tip_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["pinky_tip"]]].mean(dim=-1)
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)
    diff_level_2_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_2_joints"]]].mean(dim=-1)

    joints_vel = states["joints_state"][:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
    reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
    reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
    reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))
    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

                        
    current_obj_pos = states["manip_obj_pos"]
    current_obj_quat = states["manip_obj_quat"]

    target_obj_pos = target_states["manip_obj_pos"]
    target_obj_quat = target_states["manip_obj_quat"]
    diff_obj_pos = target_obj_pos - current_obj_pos
    diff_obj_pos_dist = torch.norm(diff_obj_pos, dim=-1)

    reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

    diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
    diff_obj_rot_angle = quat_to_angle_axis(diff_obj_rot)[0]
    reward_obj_rot = torch.exp(-3 * (diff_obj_rot_angle).abs())

    current_obj_vel = states["manip_obj_vel"]
    target_obj_vel = target_states["manip_obj_vel"]
    diff_obj_vel = target_obj_vel - current_obj_vel
    reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))

    current_obj_ang_vel = states["manip_obj_ang_vel"]
    target_obj_ang_vel = target_states["manip_obj_ang_vel"]
    diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
    reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))

    reward_power = torch.exp(-10 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    finger_tip_force = target_states["tip_force"]

    if dexterous_hand == "franka_panda":
        finger_tip_distance = target_states["tips_distance"][:, :2]                              
    else:
        finger_tip_distance = target_states["tips_distance"]

    contact_range = [0.02, 0.03]
    finger_tip_weight = torch.clamp(
        (contact_range[1] - finger_tip_distance) / (contact_range[1] - contact_range[0]), 0, 1
    )
    finger_tip_force_masked = finger_tip_force * finger_tip_weight[:, :, None]

    reward_finger_tip_force = torch.exp(-1 * (1 / (torch.norm(finger_tip_force_masked, dim=-1).sum(-1) + 1e-5)))

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
        | (torch.norm(current_obj_vel, dim=-1) > 100)
        | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
    )                

    if dexterous_hand == "franka_panda":
        failed_execute = (
            (
                (diff_obj_pos_dist > 0.03*2)
                | (diff_obj_rot_angle.abs() / np.pi * 180 > 30*2)
            )
            & (running_progress_buf >= 8)
        ) | error_buf
        
        reward_execute = (
            0.1 * reward_eef_pos
            + 0.6 * reward_eef_rot
            + 0.9 * reward_thumb_tip_pos
            + 0.8 * reward_index_tip_pos
            + 5.0 * reward_obj_pos
            + 1.0 * reward_obj_rot
            + 0.1 * reward_eef_vel
            + 0.05 * reward_eef_ang_vel
            + 0.1 * reward_joints_vel
            + 0.1 * reward_obj_vel
            + 0.1 * reward_obj_ang_vel
            + 1.0 * reward_finger_tip_force
            + 0.5 * reward_power
            + 0.5 * reward_wrist_power
        )
    elif dexterous_hand == "casiahand3finger":
        failed_execute = (
            (
                (diff_obj_pos_dist > 0.03*2)
                | (diff_obj_rot_angle.abs() / np.pi * 180 > 30*2)
            )
            & (running_progress_buf >= 8)
        ) | error_buf
        
        reward_execute = (
            0.1 * reward_eef_pos
            + 0.6 * reward_eef_rot
            + 0.9 * reward_thumb_tip_pos
            + 0.8 * reward_index_tip_pos
            + 0.6 * reward_ring_tip_pos
            + 0.5 * reward_level_1_pos
            + 0.3 * reward_level_2_pos
            + 5.0 * reward_obj_pos
            + 1.0 * reward_obj_rot
            + 0.1 * reward_eef_vel
            + 0.05 * reward_eef_ang_vel
            + 0.1 * reward_joints_vel
            + 0.1 * reward_obj_vel
            + 0.1 * reward_obj_ang_vel
            + 1.0 * reward_finger_tip_force
            + 0.5 * reward_power
            + 0.5 * reward_wrist_power
        )
    else:
        failed_execute = (
            (
                (diff_obj_pos_dist > 0.03)
                | (diff_obj_rot_angle.abs() / np.pi * 180 > 30)
            )
            & (running_progress_buf >= 8)
        ) | error_buf
        reward_execute = (
            0.1 * reward_eef_pos
            + 0.6 * reward_eef_rot
            + 0.9 * reward_thumb_tip_pos
            + 0.8 * reward_index_tip_pos
            + 0.75 * reward_middle_tip_pos
            + 0.6 * reward_pinky_tip_pos
            + 0.6 * reward_ring_tip_pos
            + 0.5 * reward_level_1_pos
            + 0.3 * reward_level_2_pos
            + 5.0 * reward_obj_pos
            + 1.0 * reward_obj_rot
            + 0.1 * reward_eef_vel
            + 0.05 * reward_eef_ang_vel
            + 0.1 * reward_joints_vel
            + 0.1 * reward_obj_vel
            + 0.1 * reward_obj_ang_vel
            + 1.0 * reward_finger_tip_force
            + 0.5 * reward_power
            + 0.5 * reward_wrist_power
        )

    succeeded = (
        progress_buf + 1 + 3 >= max_length
    ) & ~failed_execute                                                                
    reset_buf = torch.where(
        succeeded | failed_execute,
        torch.ones_like(reset_buf),
        reset_buf,
    )
    reward_dict = {
        "reward_eef_pos": reward_eef_pos,
        "reward_eef_rot": reward_eef_rot,
        "reward_eef_vel": reward_eef_vel,
        "reward_eef_ang_vel": reward_eef_ang_vel,
        "reward_joints_vel": reward_joints_vel,
        "reward_obj_pos": reward_obj_pos,
        "reward_obj_rot": reward_obj_rot,
        "reward_obj_vel": reward_obj_vel,
        "reward_obj_ang_vel": reward_obj_ang_vel,
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_middle_tip_pos
            + reward_pinky_tip_pos
            + reward_ring_tip_pos
            + reward_level_1_pos
            + reward_level_2_pos
        ),
        "reward_power": reward_power,
        "reward_wrist_power": reward_wrist_power,
        "reward_finger_tip_force": reward_finger_tip_force,
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict, error_buf


class OurTestDexHandManipLHEnv(OurTestDexHandManipRHEnv):
    side = "left"

    def __init__(
        self,
        cfg,
        *,
        rl_device=0,
        sim_device=0,
        graphics_device_id=0,
        display=False,
        record=False,
        headless=True,
    ):
        self.dexhand = DexHandFactory.create_hand(cfg["env"]["dexhand"], "left")
        super().__init__(
            cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
