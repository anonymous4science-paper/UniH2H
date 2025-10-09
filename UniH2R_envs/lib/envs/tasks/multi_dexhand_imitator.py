from __future__ import annotations

import os
import random
from enum import Enum
from itertools import cycle
from time import sleep, time
from typing import Dict, List, Tuple

import numpy as np
import torch
from ...utils import torch_jit_utils as torch_jit_utils
from ...utils.padding_utils import pad_joint_data, pad_to_max_dim, get_hand_embedding
from bps_torch.bps import bps_torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
from main.dataset.factory import ManipDataFactory


from main.dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    rot6d_to_aa,
    rot6d_to_quat,
    quat_to_aa,
)
from torch import Tensor
from tqdm import tqdm
from ...asset_root import ASSET_ROOT


from ..core.config import ROBOT_HEIGHT, config
from ...envs.core.sim_config import sim_config
from ...envs.core.vec_task import VecTask
from ...utils.pose_utils import get_mat
import pickle


import time


MANO_BONE_LINKS = [
    [0, 1], [0, 5], [0, 9], [0, 13],
    [1, 2], [2, 3], [3, 4],
    [5, 6], [6, 7], [7, 8],
    [9, 10], [10, 11], [11, 12],
    [13, 14], [14, 15], [15, 16],
    [0, 17],
    [1, 5], [5, 9], [9, 13]
]


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class MultiDexHandImitatorRHEnv(VecTask):
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

        self.hand_types = self.cfg["env"]["hand_types"]
        
        self.dexhands = {}
        for hand_type in self.hand_types:
            self.dexhands[hand_type] = DexHandFactory.create_hand(hand_type, "right")
        
        self.env_hand_mapping = {}
        
        self.dexhand = list(self.dexhands.values())[0]

        if "numActions" not in self.cfg["env"]:
            raise ValueError("numActions must be configured in the task config file")

        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.translation_scale = self.cfg["env"]["translationScale"]
        self.orientation_scale = self.cfg["env"]["orientationScale"]

        self._prop_dump_info = self.cfg["env"]["propDumpInfo"]

        self.states = {}
        self.dexhand_handles = {}
        self.dexhand_cf_weights = {} 
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
        
        num_envs_from_config = self.cfg["env"]["numEnvs"]
        
        if num_envs_from_config % len(self.hand_types) != 0:
            raise ValueError(f"num_envs ({num_envs_from_config}) must be divisible by number of hand types ({len(self.hand_types)})")
        
        self.agg_num_envs = num_envs_from_config // len(self.hand_types)
        
        def get_env_mapping(env_id):
            i_robot = env_id % len(self.hand_types)
            i_agg_env = env_id // len(self.hand_types)
            return i_robot, i_agg_env
        
        self.get_env_mapping = get_env_mapping
        
        super().__init__(
            config=self.cfg,
            rl_device=rl_device,
            sim_device=sim_device,
            graphics_device_id=graphics_device_id,
            display=display,
            record=record,
            headless=headless,
        )
        
        if self.num_envs != num_envs_from_config:
            pass
        
        self.num_envs_per_hand = self.num_envs // len(self.hand_types)
        self.total_envs = self.num_envs
        
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

        self.dexhand_default_dof_pos = {}
        for hand_type in self.hand_types:
            hand_instance = self.dexhands[hand_type]
            default_pose = torch.ones(hand_instance.n_dofs, device=self.device) * np.pi / 36
            
            if hand_type == "inspire" or hand_type == "inspire2":
                default_pose[8] = 0.3
                default_pose[9] = 0.01
            
            if hand_type == "xarm":
                default_pose = torch.zeros(hand_instance.n_dofs, device=self.device)
            
            self.dexhand_default_dof_pos[hand_type] = torch.tensor(default_pose, device=self.sim_device)

        self.bps_feat_type = "dists"
        self.bps_layer = bps_torch(
            bps_type="grid_sphere", n_bps_points=128, radius=0.2, randomize=False, device=self.device
        )

        obj_verts = self.demo_data["obj_verts"]
        self.obj_bps = self.bps_layer.encode(obj_verts, feature_type=self.bps_feat_type)[self.bps_feat_type]

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
            
            if hasattr(self, 'hand_types'):
                self.demo_dataset_dict[dataset_type] = self._load_multi_hand_dataset(dataset_type)
            else:
                self.demo_dataset_dict[dataset_type] = self._load_single_hand_dataset(dataset_type)
        
        if hasattr(self, 'hand_types'):
            self._create_multi_hand_demo_data()
        else:
            self._create_single_hand_demo_data()
        
        self.dexhand_assets = {}
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
        
        for hand_type in self.hand_types:
            dexhand_asset_file = self.dexhands[hand_type].urdf_path
            self.dexhand_assets[hand_type] = self.gym.load_asset(self.sim, *os.path.split(dexhand_asset_file), asset_options)
        
        self.dexhand_properties = {}
        
        for hand_type in self.hand_types:
            hand_asset = self.dexhand_assets[hand_type]
            hand_instance = self.dexhands[hand_type]
            
            dof_props = self.gym.get_asset_dof_properties(hand_asset)
            n_dofs = self.gym.get_asset_dof_count(hand_asset)
            
            hand_dof_stiffness = torch.tensor([500] * n_dofs, dtype=torch.float, device=self.sim_device)
            hand_dof_damping = torch.tensor([30] * n_dofs, dtype=torch.float, device=self.sim_device)
            
            for i in range(n_dofs):
                dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
                dof_props["stiffness"][i] = hand_dof_stiffness[i]
                dof_props["damping"][i] = hand_dof_damping[i]
            
            self.dexhand_properties[hand_type] = {
                'dof_lower_limits': torch.tensor([dof_props["lower"][i] for i in range(n_dofs)], device=self.sim_device),
                'dof_upper_limits': torch.tensor([dof_props["upper"][i] for i in range(n_dofs)], device=self.sim_device),
                'effort_limits': torch.tensor([dof_props["effort"][i] for i in range(n_dofs)], device=self.sim_device),
                'speed_limits': torch.tensor([dof_props["velocity"][i] for i in range(n_dofs)], device=self.sim_device),
                'n_dofs': n_dofs,
                'n_bodies': self.gym.get_asset_rigid_body_count(hand_asset),
                'n_shapes': self.gym.get_asset_rigid_shape_count(hand_asset),
                'dof_props': dof_props
            }

            rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(hand_asset)
            for element in rigid_shape_props_asset:
                element.friction = 4.0
                element.rolling_friction = 0.01
                element.torsion_friction = 0.01
            self.gym.set_asset_rigid_shape_properties(hand_asset, rigid_shape_props_asset)
            
        self.dexhand_actors = []
        self.robot_indices = []
        self.envs = []
        
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            hand_idx = i % len(self.hand_types)
            
            hand_type = self.hand_types[hand_idx]
            current_dexhand = self.dexhands[hand_type]
            current_dexhand_asset = self.dexhand_assets[hand_type]
            
            self.env_hand_mapping[i] = hand_type
            
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            
            current_num_bodies = self.gym.get_asset_rigid_body_count(current_dexhand_asset)
            current_num_shapes = self.gym.get_asset_rigid_shape_count(current_dexhand_asset)
            
            max_agg_bodies = (
                current_num_bodies + 1 + (5 + (0 + self.dexhands[hand_type].n_bodies if not self.headless else 0))
            )
            max_agg_shapes = (
                current_num_shapes
                + 1
                + (5 + (0 + self.dexhands[hand_type].n_bodies if not self.headless else 0))
                + (1 if self._record else 0)
            )
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            dexhand_actor = self.gym.create_actor(
                env_ptr,
                current_dexhand_asset,
                self.dexhand_pose,
                "dexhand",
                i,
                (1 if current_dexhand.self_collision else 0),
            )
            
            robot_idx = self.gym.get_actor_index(env_ptr, dexhand_actor, gymapi.DOMAIN_SIM)
            self.robot_indices.append(robot_idx)
            
            self.gym.enable_actor_dof_force_sensors(env_ptr, dexhand_actor)
            
            current_dexhand_dof_props = self.gym.get_asset_dof_properties(current_dexhand_asset)
            current_dof_stiffness = torch.tensor([500] * current_dexhand.n_dofs, dtype=torch.float, device=self.sim_device)
            current_dof_damping = torch.tensor([30] * current_dexhand.n_dofs, dtype=torch.float, device=self.sim_device)
            
            for j in range(current_dexhand.n_dofs):
                current_dexhand_dof_props["driveMode"][j] = gymapi.DOF_MODE_POS
                current_dexhand_dof_props["stiffness"][j] = current_dof_stiffness[j]
                current_dexhand_dof_props["damping"][j] = current_dof_damping[j]
            
            self.gym.set_actor_dof_properties(env_ptr, dexhand_actor, current_dexhand_dof_props)

            if current_dexhand.name == "xarm":
                props = self.gym.get_actor_rigid_shape_properties(env_ptr, dexhand_actor)
                
                props[0].filter = 2
                props[1].filter = 2
                props[2].filter = 2
                props[3].filter = 2
                props[4].filter = 2
                props[5].filter = 2
                props[6].filter = 2

                self.gym.set_actor_rigid_shape_properties(env_ptr, dexhand_actor, props)

            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table_actor = self.gym.create_actor(
                env_ptr, table_asset, table_pose, "table", i + self.num_envs, 0b11
            )
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_actor)  
            table_props[0].friction = 0.1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_actor, table_props)
            self.gym.set_rigid_body_color(env_ptr, table_actor, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.1, 0.1, 0.1))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.dexhand_actors.append(dexhand_actor)

        self.init_data()
    
    def _load_multi_hand_dataset(self, dataset_type):
        datasets = {}
        
        for i, hand_type in enumerate(self.hand_types):
            hand_instance = self.dexhands[hand_type]
            
            dataset = ManipDataFactory.create_data(
                manipdata_type=dataset_type,
                side=self.side,
                device=self.sim_device,
                mujoco2gym_transf=self.mujoco2gym_transf,
                max_seq_len=self.max_episode_length,
                dexhand=hand_instance,
                embodiment=hand_type,
            )
            
            datasets[hand_type] = dataset
        
        for hand_type in self.hand_types:
            hand_instance = self.dexhands[hand_type]
        
        return datasets
    
    def _load_single_hand_dataset(self, dataset_type):
        dataset = ManipDataFactory.create_data(
            manipdata_type=dataset_type,
            side=self.side,
            device=self.sim_device,
            mujoco2gym_transf=self.mujoco2gym_transf,
            max_seq_len=self.max_episode_length,
            dexhand=self.dexhand,
            embodiment=self.cfg["env"].get("dexhand", "default"),
        )
        
        return dataset
    
    def _create_multi_hand_demo_data(self):
        if not hasattr(self, 'env_hand_mapping') or not self.env_hand_mapping:
            self.env_hand_mapping = {}
            for i in range(self.num_envs):
                hand_idx = i % len(self.hand_types)
                self.env_hand_mapping[i] = self.hand_types[hand_idx]
            
        def segment_data(k):
            todo_list = self.dataIndices
            idx = todo_list[k % len(todo_list)]

            hand_idx = k % len(self.hand_types)
            hand_type = self.hand_types[hand_idx]

            dataset_type = ManipDataFactory.dataset_type(idx)
            return self.demo_dataset_dict[dataset_type][hand_type][idx]
        
        demo_data_list = []
        
        for i in range(self.num_envs):
            hand_type = self.env_hand_mapping[i]
            
            data = segment_data(i)
            demo_data_list.append(data)
        
        self.demo_data = self.pack_data(demo_data_list)
        
    def _create_single_hand_demo_data(self):
        def segment_data(k):
            todo_list = self.dataIndices
            idx = todo_list[k % len(todo_list)]
            return self.demo_dataset_dict[ManipDataFactory.dataset_type(idx)][idx]

        self.demo_data = [segment_data(i) for i in tqdm(range(self.num_envs))]
        self.demo_data = self.pack_data(self.demo_data)
        



    def init_data(self):
        self.unique_hand_types = list(set(self.env_hand_mapping.values()))
        
        for hand_type in self.unique_hand_types:
            template_env_idx = next(
                idx for idx, ht in self.env_hand_mapping.items() if ht == hand_type
            )
            
            env_ptr = self.envs[template_env_idx]
            dexhand_handle = self.gym.find_actor_handle(env_ptr, "dexhand")
            hand_instance = self.dexhands[hand_type]
            
            self.dexhand_handles[hand_type] = {
                k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) 
                for k in hand_instance.body_names
            }
            
            self.dexhand_cf_weights[hand_type] = {
                k: (1.0 if ("intermediate" in k or "distal" in k) else 0.0) 
                for k in hand_instance.body_names
            }
            
        print(f"   - 多手型句柄初始化完成，支持手型: {list(self.dexhand_handles.keys())}")
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.agg_num_envs

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)
        
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.agg_num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.agg_num_envs, -1, 13)
        
        print(f"     * _root_state shape: {self._root_state.shape}")
        print(f"     * _rigid_body_state shape: {self._rigid_body_state.shape}")
        
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]
        
        self._setup_multi_hand_dof_mapping()
        
        if not self.headless:
            max_joints = max(hand_obj.n_bodies for hand_obj in self.dexhands.values())
            self.mano_visualization_points = torch.zeros(
                (self.num_envs, max_joints, 3), 
                device=self.device, 
                dtype=torch.float32
            )
        
        self.net_cf = gymtorch.wrap_tensor(_net_cf).view(self.agg_num_envs, -1, 3)
        self.dof_force = gymtorch.wrap_tensor(_dof_force).view(self.agg_num_envs, -1)

        self.apply_forces = torch.zeros(
            (self.agg_num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.apply_torque = torch.zeros(
            (self.agg_num_envs, self._rigid_body_state.shape[1], 3), device=self.device, dtype=torch.float
        )
        self.prev_targets = torch.zeros((self.agg_num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.curr_targets = torch.zeros((self.agg_num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self._pos_control = torch.zeros((self.agg_num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        self._global_dexhand_indices = torch.tensor(
            [self.gym.find_actor_index(env, "dexhand", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.sim_device,
        ).view(self.agg_num_envs, -1)
        
        self._precompute_mappings()
    
    def _precompute_mappings(self):
        self._agg_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) // len(self.hand_types)
        
        env_hand_indices = []
        for env_id in range(self.num_envs):
            hand_type = self.env_hand_mapping[env_id]
            hand_idx = self.hand_types.index(hand_type)
            env_hand_indices.append(hand_idx)
        self._env_hand_indices = torch.tensor(env_hand_indices, device=self.device, dtype=torch.long)
        
        dof_ranges = []
        for env_id in range(self.num_envs):
            hand_type = self.env_hand_mapping[env_id] 
            dof_start, dof_end = self.hand_dof_ranges[hand_type]
            dof_ranges.append([dof_start, dof_end])
        self._env_dof_ranges = torch.tensor(dof_ranges, device=self.device, dtype=torch.long)
        
        self._env_hand_dofs = self._env_dof_ranges[:, 1] - self._env_dof_ranges[:, 0]
        
        self._hand_type_env_groups = {}
        for hand_type_idx, hand_type in enumerate(self.hand_types):
            hand_env_mask = (self._env_hand_indices == hand_type_idx)
            hand_env_ids = torch.where(hand_env_mask)[0]
            self._hand_type_env_groups[hand_type] = hand_env_ids
        
        self._hand_type_groups = {}
        for env_idx in range(self.num_envs):
            hand_type = self.env_hand_mapping[env_idx]
            if hand_type not in self._hand_type_groups:
                self._hand_type_groups[hand_type] = []
            self._hand_type_groups[hand_type].append(env_idx)
        
        embedding_dim = self.cfg["env"].get("embedding_dim", 3)
        self._precomputed_hand_embeddings = torch.zeros(
            self.num_envs, embedding_dim, device=self.device, dtype=torch.float32
        )
        
        for env_id in range(self.num_envs):
            hand_type = self.env_hand_mapping[env_id]
            hand_instance = self.dexhands[hand_type]
            embedding = get_hand_embedding(hand_instance.cfg["embedding"], device=self.device)
            self._precomputed_hand_embeddings[env_id] = embedding
            
        self._wrist_rigid_body_indices = {}
        cumulative_bodies = 0
        
        for hand_idx, hand_type in enumerate(self.hand_types):
            hand_instance = self.dexhands[hand_type]
            wrist_body_name = hand_instance.to_dex("wrist")[0]
            local_wrist_handle = self.dexhand_handles[hand_type][wrist_body_name]
            
            agg_wrist_index = cumulative_bodies + local_wrist_handle
            self._wrist_rigid_body_indices[hand_type] = agg_wrist_index
            
            cumulative_bodies += self.dexhand_properties[hand_type]['n_bodies'] + 1
            
    def _get_hand_actor_index(self, agg_env_id, hand_type):
        hand_idx = self.hand_types.index(hand_type)
        dexhand_actor_idx = hand_idx * 2
        return dexhand_actor_idx
    


    def _setup_multi_hand_dof_mapping(self):
        self.hand_dof_ranges = {}
        cumulative_dofs = 0
        
        for hand_type in self.hand_types:
            hand_dofs = self.dexhands[hand_type].n_dofs
            self.hand_dof_ranges[hand_type] = (cumulative_dofs, cumulative_dofs + hand_dofs)
            cumulative_dofs += hand_dofs
            
    def _get_agg_env_and_dof_range(self, env_id):
        hand_type = self.env_hand_mapping[env_id]
        agg_env_id = env_id // len(self.hand_types)
        dof_start, dof_end = self.hand_dof_ranges[hand_type]
        return agg_env_id, dof_start, dof_end, hand_type
    
    def get_hand_properties(self, env_idx):
        hand_type = self.env_hand_mapping[env_idx]
        return self.dexhand_properties[hand_type]
    
    def get_wrist_rigid_body_indices(self, agg_env_ids, hand_types):
        wrist_indices = torch.zeros_like(agg_env_ids, dtype=torch.long, device=self.device)
        
        for i, hand_type in enumerate(hand_types):
            wrist_indices[i] = self._wrist_rigid_body_indices[hand_type]
            
        return wrist_indices
    
    def get_hand_dof_limits(self, env_indices):
        if isinstance(env_indices, int):
            env_indices = [env_indices]
        
        limits_dict = {}
        for env_idx in env_indices:
            hand_type = self.env_hand_mapping[env_idx]
            if hand_type not in limits_dict:
                limits_dict[hand_type] = []
            limits_dict[hand_type].append(env_idx)
        
        return limits_dict
    
    def get_batch_dof_limits_for_hand_type(self, hand_type, num_envs):
        props = self.dexhand_properties[hand_type]
        lower_limits = props['dof_lower_limits'].unsqueeze(0).repeat(num_envs, 1)
        upper_limits = props['dof_upper_limits'].unsqueeze(0).repeat(num_envs, 1)
        return lower_limits, upper_limits

    def pack_data(self, data):
        packed_data = {}
        packed_data["seq_len"] = torch.tensor([len(d["obj_trajectory"]) for d in data], device=self.device)
        max_len = packed_data["seq_len"].max()
        assert max_len <= self.max_episode_length, "max_len should be less than max_episode_length"
        
        max_dofs = 0
        if hasattr(self, 'dexhands') and self.dexhands:
            max_dofs = 22

        MAX_JOINTS = 27

        def fill_data(stack_data, pad_dof_dim=False, pad_joints_dim=False):
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
                    
                if pad_dof_dim and max_dofs > 0 and stack_data[i].shape[-1] < max_dofs:
                    current_dofs = stack_data[i].shape[-1]
                    padding_size = max_dofs - current_dofs
                    padding_shape = list(stack_data[i].shape)
                    padding_shape[-1] = padding_size
                    padding = torch.zeros(padding_shape, device=stack_data[i].device, dtype=stack_data[i].dtype)
                    stack_data[i] = torch.cat([stack_data[i], padding], dim=-1)
                
                if pad_joints_dim and stack_data[i].shape[-1] < MAX_JOINTS * 3:
                    current_dim = stack_data[i].shape[-1]
                    current_joints = current_dim // 3
                    padding_joints = MAX_JOINTS - current_joints
                    padding_size = padding_joints * 3
                    padding_shape = list(stack_data[i].shape)
                    padding_shape[-1] = padding_size
                    padding = torch.zeros(padding_shape, device=stack_data[i].device, dtype=stack_data[i].dtype)
                    stack_data[i] = torch.cat([stack_data[i], padding], dim=-1)
                        
            return torch.stack(stack_data).squeeze()

        dof_related_fields = {
            'opt_dof_pos', 'opt_dof_velocity'
        }
        
        for k in data[0].keys():
            if "alt" in k:
                continue
            if k == "mano_joints" or k == "mano_joints_velocity":
                hand_type_groups = {}
                for i, d in enumerate(data):
                    hand_type = self.env_hand_mapping[i]
                    if hand_type not in hand_type_groups:
                        hand_type_groups[hand_type] = []
                    hand_type_groups[hand_type].append((i, d))

                mano_joints = [None] * len(data)
                
                for hand_type, group_data in hand_type_groups.items():
                    hand_instance = self.dexhands[hand_type]
                    for original_idx, d in group_data:
                        mano_joints[original_idx] = torch.concat(
                            [
                                d[k][hand_instance.to_hand(j_name)[0]]
                                for j_name in hand_instance.body_names
                                if hand_instance.to_hand(j_name)[0] != "wrist"
                            ],
                            dim=-1,
                        )
                
                packed_data[k] = fill_data(mano_joints, pad_dof_dim=False, pad_joints_dim=True)
            elif type(data[0][k]) == torch.Tensor:
                stack_data = [d[k] for d in data]
                if k != "obj_verts":
                    need_dof_padding = k in dof_related_fields
                    packed_data[k] = fill_data(stack_data, pad_dof_dim=need_dof_padding)
                else:
                    packed_data[k] = torch.stack(stack_data).squeeze()
            elif type(data[0][k]) == np.ndarray:
                raise RuntimeError("Using np is very slow.")
            else:
                packed_data[k] = [d[k] for d in data]

        def to_cuda(x):
            if type(x) == torch.Tensor:
                return x.to(self.device)
            elif type(x) == list:
                return [to_cuda(xx) for xx in x]
            elif type(x) == dict:
                return {k: to_cuda(v) for k, v in x.items()}
            else:
                return x

        packed_data = to_cuda(packed_data)

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

    def _update_states(self):
        if self._q is None:
            return
        
        k_max = self.cfg["env"].get("k_max", 22)
        
        env_q = torch.zeros((self.num_envs, k_max), device=self.device, dtype=torch.float32)
        env_qd = torch.zeros((self.num_envs, k_max), device=self.device, dtype=torch.float32)
        env_base_state = torch.zeros((self.num_envs, 13), device=self.device, dtype=torch.float32)
        
        for hand_type in self.hand_types:
            hand_env_ids = self._hand_type_env_groups[hand_type]
            
            if len(hand_env_ids) > 0:
                hand_agg_env_ids = self._agg_env_ids[hand_env_ids]
                dof_start, dof_end = self.hand_dof_ranges[hand_type]
                hand_dof_count = dof_end - dof_start
                
                env_q[hand_env_ids, :hand_dof_count] = self._q[hand_agg_env_ids, dof_start:dof_end]
                env_qd[hand_env_ids, :hand_dof_count] = self._qd[hand_agg_env_ids, dof_start:dof_end]
        
        env_actor_indices = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        for hand_type in self.hand_types:
            hand_env_ids = self._hand_type_env_groups[hand_type]
            if len(hand_env_ids) > 0:
                hand_agg_env_ids = self._agg_env_ids[hand_env_ids]
                sample_agg_env_id = hand_agg_env_ids[0].item()
                actor_idx = self._get_hand_actor_index(sample_agg_env_id, hand_type)
                env_actor_indices[hand_env_ids] = actor_idx
        
           
        self.states.update(
            {
                "q": env_q,
                "cos_q": torch.cos(env_q),
                "sin_q": torch.sin(env_q),
                "dq": env_qd,
                "base_state": env_base_state,
            }
        )
        
        MAX_JOINTS = 28
        joints_state = torch.zeros((self.num_envs, MAX_JOINTS, 10), device=self.device, dtype=torch.float32)
        
        for hand_type in self.hand_types:
            hand_env_ids = self._hand_type_env_groups[hand_type]
            if len(hand_env_ids) > 0:
                agg_env_ids = self._agg_env_ids[hand_env_ids]
                hand_instance = self.dexhands[hand_type]
                
                if not hasattr(self, '_hand_body_offsets'):
                    self._hand_body_offsets = {}
                    cumulative = 0
                    for ht in self.hand_types:
                        self._hand_body_offsets[ht] = cumulative
                        cumulative += self.dexhand_properties[ht]['n_bodies'] + 1
                
                body_offset = self._hand_body_offsets[hand_type]
                local_handles = self.dexhand_handles[hand_type]
                
                hand_joints_data = []
                for body_name in hand_instance.body_names:
                    agg_body_idx = body_offset + local_handles[body_name]
                    body_state = self._rigid_body_state[agg_env_ids, agg_body_idx, :10]
                    hand_joints_data.append(body_state)
                
                padded_joints_state = torch.zeros((len(hand_env_ids), MAX_JOINTS, 10), 
                                                device=self.device, dtype=torch.float32)
                
                actual_joints = len(hand_instance.body_names)
                if hand_joints_data:
                    hand_joints_tensor = torch.stack(hand_joints_data, dim=1)
                    padded_joints_state[:, :actual_joints, :] = hand_joints_tensor
                
                joints_state[hand_env_ids] = padded_joints_state
        
        self.states["joints_state"] = joints_state

    def _refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self._update_states()
        
    def compute_reward(self, actions):
        k_max = self.cfg["env"].get("k_max", 22)
        env_dof_force = torch.zeros((self.num_envs, k_max), device=self.device, dtype=torch.float32)
        
        for hand_type in self.hand_types:
            hand_env_ids = self._hand_type_env_groups[hand_type]
            if len(hand_env_ids) > 0:
                hand_agg_env_ids = self._agg_env_ids[hand_env_ids]
                dof_start, dof_end = self.hand_dof_ranges[hand_type]
                hand_dof_count = dof_end - dof_start
                env_dof_force[hand_env_ids, :hand_dof_count] = self.dof_force[hand_agg_env_ids, dof_start:dof_end]
        
        self.env_dof_force = env_dof_force
        
        self.env_apply_forces = self.apply_forces[self._agg_env_ids]
        self.env_apply_torque = self.apply_torque[self._agg_env_ids]
        
        target_state = {}
        max_length = torch.clip(self.demo_data["seq_len"], 0, self.max_episode_length).float()
        cur_idx = self.progress_buf
        cur_wrist_pos = self.demo_data["wrist_pos"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_pos"] = cur_wrist_pos
        cur_wrist_rot = self.demo_data["wrist_rot"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_quat"] = aa_to_quat(cur_wrist_rot)[:, [1, 2, 3, 0]]

        target_state["wrist_vel"] = self.demo_data["wrist_velocity"][torch.arange(self.num_envs), cur_idx]
        target_state["wrist_ang_vel"] = self.demo_data["wrist_angular_velocity"][torch.arange(self.num_envs), cur_idx]

        cur_joints_pos = self.demo_data["mano_joints"][torch.arange(self.num_envs), cur_idx]
        target_state["joints_pos"] = cur_joints_pos.reshape(self.num_envs, -1, 3)
        target_state["joints_vel"] = self.demo_data["mano_joints_velocity"][
            torch.arange(self.num_envs), cur_idx
        ].reshape(self.num_envs, -1, 3)

        power = torch.abs(torch.multiply(self.env_dof_force, self.states["dq"])).sum(dim=-1)
        target_state["power"] = power

        wrist_power = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        
        for hand_type in self.hand_types:
            hand_env_ids = self._hand_type_env_groups[hand_type]
            
            if len(hand_env_ids) > 0:
                wrist_rigid_body_idx = self._wrist_rigid_body_indices[hand_type]
                
                agg_env_ids = self._agg_env_ids[hand_env_ids]
                
                hand_wrist_power = torch.abs(
                    torch.sum(
                        self.apply_forces[agg_env_ids, wrist_rigid_body_idx, :]
                        * self.states["base_state"][hand_env_ids, 7:10],
                        dim=-1,
                    )
                )
                
                hand_wrist_power += torch.abs(
                    torch.sum(
                        self.apply_torque[agg_env_ids, wrist_rigid_body_idx, :]
                        * self.states["base_state"][hand_env_ids, 10:],
                        dim=-1,
                    )
                )
                
                wrist_power[hand_env_ids] = hand_wrist_power
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
                scale_factor = 1.0
        else:
            scale_factor = 1.0

        self.rew_buf[:], self.reset_buf[:], self.success_buf[:], self.failure_buf[:], self.reward_dict = (
            compute_imitation_reward(
                self.reset_buf,
                self.progress_buf,
                self.running_progress_buf,
                self.actions,
                self.states,
                target_state,
                max_length,
                scale_factor,
                self.dexhands,
                self.env_hand_mapping,
                self._hand_type_groups,
            )
        )
        self.total_rew_buf += self.rew_buf

    def compute_observations(self):
        self._refresh()
        
        k_max = self.cfg["env"].get("k_max", 22)
        embedding_dim = self.cfg["env"].get("embedding_dim", 3)
        
        obs_values = []
        
        if "q" in self._obs_keys:
            obs_values.append(self.states["q"])
        
        if "cos_q" in self._obs_keys:
            obs_values.append(self.states["cos_q"])
        
        if "sin_q" in self._obs_keys:
            obs_values.append(self.states["sin_q"])
        
        if "base_state" in self._obs_keys:
            obs_values.append(
                torch.cat([torch.zeros_like(self.states["base_state"][:, :3]), self.states["base_state"][:, 3:]], dim=-1)
            )  # ! 忽略基座位置
            
        if hasattr(self, 'hand_types'):
            hand_embedding = self._precomputed_hand_embeddings
        else:
            hand_embedding = get_hand_embedding(self.dexhand.cfg["embedding"], device=self.device)
            hand_embedding = hand_embedding.repeat(self.num_envs, 1)
        obs_values.append(hand_embedding)
        
        concatenated_obs = torch.cat(obs_values, dim=-1)
        self.obs_dict["proprioception"][:] = concatenated_obs
        
        if len(self._privileged_obs_keys) > 0:
            pri_obs_values = []
            for ob in self._privileged_obs_keys:
                if ob == "manip_obj_pos":
                    pri_obs_values.append(self.states[ob] - self.states["base_state"][:, :3])
                elif ob == "manip_obj_com":
                    cur_com_pos = (
                        quat_to_rotmat(self.states["manip_obj_quat"][:, [1, 2, 3, 0]])
                        @ self.manip_obj_com.unsqueeze(-1)
                    ).squeeze(-1) + self.states["manip_obj_pos"]
                    pri_obs_values.append(cur_com_pos - self.states["base_state"][:, :3])
                elif ob == "manip_obj_weight":
                    prop = self.gym.get_sim_params(self.sim)
                    pri_obs_values.append((self.manip_obj_mass * -1 * prop.gravity.z).unsqueeze(-1))
                elif ob == "tip_force":
                    expanded_net_cf = self.net_cf.repeat_interleave(len(self.hand_types), dim=0)
                    
                    tip_force_list = []
                    
                    for hand_type in self.hand_types:
                        hand_env_ids = self._hand_type_env_groups[hand_type]
                        
                        if len(hand_env_ids) > 0:
                            hand_instance = self.dexhands[hand_type]
                            hand_handles = self.dexhand_handles[hand_type]
                            
                            hand_tip_force = torch.stack(
                                [expanded_net_cf[hand_env_ids, hand_handles[k], :] for k in hand_instance.contact_body_names],
                                axis=1,
                            )  # shape: [len(hand_env_ids), num_contact_bodies, 3]
                            
                            tip_force_list.append((hand_env_ids, hand_tip_force))
                    
                    max_contact_bodies = max([len(self.dexhands[ht].contact_body_names) for ht in self.hand_types])
                    tip_force = torch.zeros((self.num_envs, max_contact_bodies, 3), device=self.device, dtype=torch.float32)
                    
                    for hand_env_ids, hand_tip_force in tip_force_list:
                        tip_force[hand_env_ids, :hand_tip_force.shape[1], :] = hand_tip_force
                    tip_force = torch.cat(
                        [tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1
                    )  # add force magnitude
                    pri_obs_values.append(tip_force.reshape(self.num_envs, -1))
                elif ob == "dq":
                    pri_obs_values.append(self.states["dq"])
                else:
                    pri_obs_values.append(self.states[ob])
            self.obs_dict["privileged"][:] = torch.cat(pri_obs_values, dim=-1)
        
        target_vector = torch.cat(
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
                ]
            ],
            dim=-1,
        )
        
        target_max_dim = self.cfg["env"].get("target_max_dim", 266)  # 默认为266，与Shadow手对应
        padded_target = pad_to_max_dim(target_vector, target_max_dim)
        self.obs_dict["target"][:] = padded_target
        
        if not self.training:
            for prop_name in self._prop_dump_info.keys():
                self.dump_fileds[prop_name][:] = self.states[prop_name][:]

        if not hasattr(self, '_dims_printed'):
            if not hasattr(self, '_dims_printed'):
                pass
            self._dims_printed = True

        return self.obs_dict

    def _reset_default(self, env_ids):
        if self.random_state_init:
            seq_idx = torch.floor(
                self.demo_data["seq_len"][env_ids] * 0.99 * torch.rand_like(self.demo_data["seq_len"][env_ids].float())
            ).long()
        else:
            seq_idx = torch.zeros_like(self.demo_data["seq_len"][env_ids].long())
        
        dof_pos_per_env = []
        dof_vel_per_env = []
        
        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item()
            hand_type = self.env_hand_mapping[env_id_item]
            hand_props = self.dexhand_properties[hand_type]
            hand_instance = self.dexhands[hand_type]
            
            default_dof_pos = self.dexhand_default_dof_pos[hand_type]
            
            noise_dof_pos = (
                torch.randn_like(default_dof_pos)
                * ((hand_props['dof_upper_limits'] - hand_props['dof_lower_limits']) / 8)
            )
            
            env_dof_pos = torch.clamp(
                default_dof_pos + noise_dof_pos,
                hand_props['dof_lower_limits'],
                hand_props['dof_upper_limits'],
            )
            
            env_dof_vel = torch.randn(hand_props['n_dofs'], device=self.device) * 0.1
            env_dof_vel = torch.clamp(
                env_dof_vel,
                -1 * hand_props['speed_limits'],
                hand_props['speed_limits'],
            )
            
            dof_pos_per_env.append(env_dof_pos)
            dof_vel_per_env.append(env_dof_vel)

        opt_wrist_pos = self.demo_data["wrist_pos"][env_ids, seq_idx]
        opt_wrist_pos = opt_wrist_pos + torch.randn_like(opt_wrist_pos) * 0.01
        opt_wrist_rot_aa = self.demo_data["wrist_rot"][env_ids, seq_idx]
        opt_wrist_rot = aa_to_rotmat(opt_wrist_rot_aa)
        noise_rot = torch.rand(opt_wrist_rot.shape[0], 3, device=self.device)
        noise_rot = aa_to_rotmat(
            noise_rot
            / torch.norm(noise_rot, dim=-1, keepdim=True)
            * torch.randn(opt_wrist_rot.shape[0], 1, device=self.device)
            * (np.pi / 18)
        )
        opt_wrist_rot = noise_rot @ opt_wrist_rot
        opt_wrist_rot = rotmat_to_quat(opt_wrist_rot)
        opt_wrist_rot = opt_wrist_rot[:, [1, 2, 3, 0]]

        opt_wrist_vel = self.demo_data["wrist_velocity"][env_ids, seq_idx]
        opt_wrist_vel = opt_wrist_vel + torch.randn_like(opt_wrist_vel) * 0.01
        opt_wrist_ang_vel = self.demo_data["wrist_angular_velocity"][env_ids, seq_idx]
        opt_wrist_ang_vel = opt_wrist_ang_vel + torch.randn_like(opt_wrist_ang_vel) * 0.01

        opt_hand_pose_vel = torch.concat([opt_wrist_pos, opt_wrist_rot, opt_wrist_vel, opt_wrist_ang_vel], dim=-1)

        for i, env_id in enumerate(env_ids):
            env_id_item = env_id.item()
            agg_env_id, dof_start, dof_end, hand_type = self._get_agg_env_and_dof_range(env_id_item)
            
            actor_idx = self._get_hand_actor_index(agg_env_id, hand_type)
            self._root_state[agg_env_id, actor_idx, :] = opt_hand_pose_vel[i]
            
            hand_dof_pos = dof_pos_per_env[i]  # 手型特定的DOF位置
            hand_dof_vel = dof_vel_per_env[i]  # 手型特定的DOF速度
            
            self._q[agg_env_id, dof_start:dof_end] = hand_dof_pos
            self._qd[agg_env_id, dof_start:dof_end] = hand_dof_vel
            self._pos_control[agg_env_id, dof_start:dof_end] = hand_dof_pos

        agg_env_ids = torch.unique(torch.tensor([env_id.item() // len(self.hand_types) for env_id in env_ids], device=self.device))
        
        if len(agg_env_ids) < len(self._global_dexhand_indices):
            dexhand_multi_env_ids_int32 = self._global_dexhand_indices[agg_env_ids].flatten()
        else:
            dexhand_multi_env_ids_int32 = self._global_dexhand_indices.flatten()
        
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._dof_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
        )
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
            len(dexhand_multi_env_ids_int32),
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
        agg_env_ids = env_ids // len(self.hand_types)
        self.apply_forces[agg_env_ids] = 0
        self.apply_torque[agg_env_ids] = 0
        self.curr_targets[agg_env_ids] = 0
        self.prev_targets[agg_env_ids] = 0

        return self.obs_dict, done_env_ids

    def reset_idx(self, env_ids):
        self._refresh()
        if self.randomize:
            self.apply_randomizations(self.dr_randomizations)

        return self._reset_default(env_ids)

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
        
        actions_per_hand_type = []
        
        for hand_idx, hand_type in enumerate(self.hand_types):
            hand_env_indices = [i for i in range(self.num_envs) if self.env_hand_mapping[i] == hand_type]
            
            if len(hand_env_indices) == 0:
                continue
                
            hand_actions = actions[hand_env_indices]
            actions_per_hand_type.append(hand_actions)
            
        dof_actions_per_hand_type = []
        
        for hand_type, hand_actions in zip(self.hand_types, actions_per_hand_type):
            hand_props = self.dexhand_properties[hand_type]
            hand_n_dofs = hand_props['n_dofs']
            num_hand_envs = hand_actions.shape[0]
            
            dof_actions = torch.zeros((num_hand_envs, hand_n_dofs), device=self.device)
            
            root_control_dim = 6
            if hand_actions.shape[1] > root_control_dim:
                joint_actions = hand_actions[:, root_control_dim:root_control_dim + hand_n_dofs]
                joint_actions = torch.clamp(joint_actions, -1, 1)
                
                dof_actions = torch_jit_utils.scale(
                    joint_actions,
                    hand_props['dof_lower_limits'],
                    hand_props['dof_upper_limits'],
                )
            
            dof_actions_per_hand_type.append(dof_actions)
            
        agg_num_envs = len(actions_per_hand_type[0]) if actions_per_hand_type else 1
        total_dofs = sum([self.dexhand_properties[ht]['n_dofs'] for ht in self.hand_types])
        
        agg_actions = torch.zeros((agg_num_envs, total_dofs), device=self.device)
        
        dof_start = 0

        
        if not hasattr(self, 'curr_targets') or self.curr_targets.shape != agg_actions.shape:
            self.curr_targets = torch.zeros_like(agg_actions)
            
        curr_act_moving_average = self.act_moving_average
        self.curr_targets = (
            curr_act_moving_average * self.curr_targets + 
            (1.0 - curr_act_moving_average) * self.prev_targets
        )
        
        dof_start = 0
        for hand_type in self.hand_types:
            hand_props = self.dexhand_properties[hand_type]
            hand_n_dofs = hand_props['n_dofs']
            dof_end = dof_start + hand_n_dofs
            
            self.curr_targets[:, dof_start:dof_end] = torch_jit_utils.tensor_clamp(
                self.curr_targets[:, dof_start:dof_end],
                hand_props['dof_lower_limits'],
                hand_props['dof_upper_limits'],
            )
            
            if hand_type == "xarm":
                main_joint_value = self.curr_targets[:, dof_start]  # 主关节的值
                for j in range(1, hand_n_dofs):  # 从索引1开始，跳过主关节
                    self.curr_targets[:, dof_start + j] = main_joint_value
            
            dof_start = dof_end

        for hand_type in self.hand_types:
            hand_env_ids = self._hand_type_env_groups[hand_type]
            if len(hand_env_ids) == 0:
                continue
                
            agg_env_ids = self._agg_env_ids[hand_env_ids]
            
            wrist_rigid_body_idx = self._wrist_rigid_body_indices[hand_type]

            
            self.apply_forces[agg_env_ids, wrist_rigid_body_idx, :] = (
                curr_act_moving_average * (actions[hand_env_ids, 3:6] * self.dt * self.translation_scale * 500)
                + (1.0 - curr_act_moving_average)
                * self.apply_forces[agg_env_ids, wrist_rigid_body_idx, :]
            )
            
            self.apply_torque[agg_env_ids, wrist_rigid_body_idx, :] = (
                curr_act_moving_average * (actions[hand_env_ids, 6:9] * self.dt * self.orientation_scale * 200)
                + (1.0 - curr_act_moving_average)
                * self.apply_torque[agg_env_ids, wrist_rigid_body_idx, :]
            )

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.apply_forces),
            gymtorch.unwrap_tensor(self.apply_torque),
            gymapi.ENV_SPACE,
        )

        self.prev_targets = self.curr_targets.clone()
        
        if hasattr(self, '_pos_control'):
            if self._pos_control.shape != self.curr_targets.shape:
                self._pos_control = torch.zeros_like(self.curr_targets)
            self._pos_control[:] = self.curr_targets[:]
        else:
            self._pos_control = self.curr_targets.clone()

        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        
        if False:
            pass

    def post_physics_step(self):
        
        self.compute_observations()
        
        self.compute_reward(self.actions)
        
        
        self.progress_buf += 1
        self.running_progress_buf += 1
        self.randomize_buf += 1
        
    def create_camera(
        self,
        *,
        env,
        isaac_gym,
    ):
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
                    0.6,
                    0.6,
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
def compute_four_five_finger_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
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

    current_dof_pos = states["q"]
    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

    reward_power = torch.exp(-10 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
    )  # sanity check

    failed_execute = (
        (
            (diff_thumb_tip_pos_dist > 0.04 / 0.7 * scale_factor)
            | (diff_index_tip_pos_dist > 0.045 / 0.7 * scale_factor)
            | (diff_middle_tip_pos_dist > 0.05 / 0.7 * scale_factor)
            | (diff_pinky_tip_pos_dist > 0.06 / 0.7 * scale_factor)
            | (diff_ring_tip_pos_dist > 0.06 / 0.7 * scale_factor)
            | (diff_level_1_pos_dist > 0.07 / 0.7 * scale_factor)
            | (diff_level_2_pos_dist > 0.08 / 0.7 * scale_factor)
        )
        & (running_progress_buf >= 20)
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
        + 0.1 * reward_eef_vel
        + 0.05 * reward_eef_ang_vel
        + 0.1 * reward_joints_vel
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
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict


@torch.jit.script
def compute_two_finger_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    actions: Tensor,
    states: Dict[str, Tensor],
    target_states: Dict[str, Tensor],
    max_length: List[int],
    scale_factor: float,
    dexhand_weight_idx: Dict[str, List[int]],
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
    
    diff_level_1_pos_dist = diff_joints_pos_dist[:, [k - 1 for k in dexhand_weight_idx["level_1_joints"]]].mean(dim=-1)

    joints_vel = states["joints_state"][:, 1:, 7:10]
    target_joints_vel = target_states["joints_vel"]
    diff_joints_vel = target_joints_vel - joints_vel

    reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
    reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
    reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
    reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)

    reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
    reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

    current_dof_pos = states["q"]
    current_dof_vel = states["dq"]

    diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
    diff_eef_rot_angle = quat_to_angle_axis(diff_eef_rot)[0]
    reward_eef_rot = torch.exp(-1 * (diff_eef_rot_angle).abs())

    reward_power = torch.exp(-10 * target_states["power"])
    reward_wrist_power = torch.exp(-2 * target_states["wrist_power"])

    error_buf = (
        (torch.norm(current_eef_vel, dim=-1) > 100)
        | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(current_dof_vel).mean(-1) > 200)
    )  # sanity check

    failed_execute = (
        (
            (diff_thumb_tip_pos_dist > 0.04 / 0.7 * scale_factor)
            | (diff_index_tip_pos_dist > 0.045 / 0.7 * scale_factor)
            | (diff_level_1_pos_dist > 0.07 / 0.7 * scale_factor)
        )
        & (running_progress_buf >= 20)
    ) | error_buf
    reward_execute = (
        0.1 * reward_eef_pos
        + 0.6 * reward_eef_rot
        + 0.9 * reward_thumb_tip_pos
        + 0.8 * reward_index_tip_pos
        + 0.5 * reward_level_1_pos
        + 0.1 * reward_eef_vel
        + 0.05 * reward_eef_ang_vel
        + 0.5 * reward_power
        + 0.5 * reward_wrist_power
    )

    succeeded = (
        progress_buf + 1 + 3 >= max_length
    ) & ~failed_execute  # reached the end of the trajectory, +3 for max future 3 steps
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
        "reward_joints_pos": (
            reward_thumb_tip_pos
            + reward_index_tip_pos
            + reward_level_1_pos
        ),
        "reward_power": reward_power,
        "reward_wrist_power": reward_wrist_power,
    }

    return reward_execute, reset_buf, succeeded, failed_execute, reward_dict


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
    dexhands: Dict[str, any],
    env_hand_mapping: Dict[int, str],
    hand_type_groups: Dict[str, List[int]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor]]:
    num_envs = reset_buf.shape[0]
    
    reset_buf = reset_buf.bool()
    progress_buf = progress_buf.float()
    running_progress_buf = running_progress_buf.float()
    
    device = reset_buf.device
    final_rewards = torch.zeros(num_envs, device=device)
    final_reset_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    final_success_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    final_failure_buf = torch.zeros(num_envs, dtype=torch.bool, device=device)
    final_reward_dict: Dict[str, Tensor] = {}
    
    for hand_type, env_indices in hand_type_groups.items():
        if len(env_indices) == 0:
            continue
        
        indices_tensor = torch.tensor(env_indices, dtype=torch.long, device=device)
        
        group_reset_buf = reset_buf[indices_tensor]
        group_progress_buf = progress_buf[indices_tensor]
        group_running_progress_buf = running_progress_buf[indices_tensor]
        group_actions = actions[indices_tensor] if actions is not None else actions
        
        group_states: Dict[str, Tensor] = {}
        for key, value in states.items():
            group_states[key] = value[indices_tensor]
        
        group_target_states: Dict[str, Tensor] = {}
        for key, value in target_states.items():
            group_target_states[key] = value[indices_tensor]
        
        group_max_length_list = [max_length[i] for i in env_indices]
        if len(group_max_length_list) > 0 and isinstance(group_max_length_list[0], torch.Tensor):
            group_max_length_tensor = torch.stack(group_max_length_list)
        else:
            group_max_length_tensor = torch.tensor(group_max_length_list, dtype=torch.float32, device=device)
        
        current_hand_instance = dexhands[hand_type]
        dexhand_weight_idx = current_hand_instance.weight_idx
        
        if hand_type == "franka_panda":
            group_rewards, group_reset, group_prog, group_run_prog, group_reward_dict = compute_two_finger_reward(
                group_reset_buf,
                group_progress_buf,
                group_running_progress_buf,
                group_actions,
                group_states,
                group_target_states,
                group_max_length_tensor,
                scale_factor,
                dexhand_weight_idx,
            )
        else:
            group_rewards, group_reset, group_prog, group_run_prog, group_reward_dict = compute_four_five_finger_reward(
                group_reset_buf,
                group_progress_buf,
                group_running_progress_buf,
                group_actions,
                group_states,
                group_target_states,
                group_max_length_tensor,
                scale_factor,
                dexhand_weight_idx,
            )
        
        final_rewards[indices_tensor] = group_rewards
        final_reset_buf[indices_tensor] = group_reset
        final_success_buf[indices_tensor] = group_prog
        final_failure_buf[indices_tensor] = group_run_prog
        
        for key, value in group_reward_dict.items():
            if key not in final_reward_dict:
                final_reward_dict[key] = torch.zeros(num_envs, *value.shape[1:], device=device)
            final_reward_dict[key][indices_tensor] = value
    
    return final_rewards, final_reset_buf, final_success_buf, final_failure_buf, final_reward_dict
 


class MultiDexHandImitatorLHEnv(MultiDexHandImitatorRHEnv):
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
        if "hand_types" in cfg["env"]:
            self.dexhand = DexHandFactory.create_hand(cfg["env"]["hand_types"][0], "left")
        else:
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