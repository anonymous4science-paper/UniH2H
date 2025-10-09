import math
import os
import pickle
import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import logging

                                    
logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

import torch
from termcolor import cprint

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rotmat_to_quat,
)
from UniH2R_envs.lib.envs.dexhands.factory import DexHandFactory


class RetargetingVisualizer:
    def __init__(self, args, dexhand):
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.dexhand = dexhand
        
                               
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        
                                                        
        self.headless = getattr(args, 'headless', False)
        if self.headless:
            self.graphics_device_id = -1
        
                              
        physics_engine = getattr(args, 'physics_engine', gymapi.SIM_PHYSX)
        
        self.sim_params.substeps = 2
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 8
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = getattr(args, 'num_threads', 0)
        self.sim_params.physx.use_gpu = getattr(args, 'use_gpu', True)
        self.sim_params.use_gpu_pipeline = getattr(args, 'use_gpu_pipeline', True)
        self.sim_device = getattr(args, 'sim_device', 'cuda:0') if self.sim_params.use_gpu_pipeline else "cpu"
        
                           
        compute_device_id = getattr(args, 'compute_device_id', 0)
        graphics_device_id = getattr(args, 'graphics_device_id', 0)
        self.sim = self.gym.create_sim(
            compute_device_id, graphics_device_id, physics_engine, self.sim_params
        )
        
                          
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
                                       
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        
                                                                       
                                                            
        table_width_offset = 0.2
        self.table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        table_half_height = 0.015
        table_surface_z = self.table_pos.z + table_half_height
        
                           
        cprint(f"初始化table_pos: x={self.table_pos.x}, y={self.table_pos.y}, z={self.table_pos.z}", "cyan")
        
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        mujoco2gym_transf[:3, 3] = np.array([0, 0, table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

    def load_dexhand_asset(self):
        """Load dexterous hand asset"""
        asset_root = os.path.split(self.dexhand.urdf_path)[0]
        asset_file = os.path.split(self.dexhand.urdf_path)[1]
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        
        self.dexhand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        
                             
        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(self.dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(self.dexhand_asset)
        
                                  
        dexhand_dof_props = self.gym.get_asset_dof_properties(self.dexhand_asset)
        
                                 
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.dexhand_asset)
        for element in rigid_shape_props_asset:
            element.friction = 1.0
            element.rolling_friction = 0.1
            element.torsion_friction = 0.1
        self.gym.set_asset_rigid_shape_properties(self.dexhand_asset, rigid_shape_props_asset)
        
                                        
        for i in range(self.num_dexhand_dofs):
            dexhand_dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            dexhand_dof_props["stiffness"][i] = 20.0
            dexhand_dof_props["damping"][i] = 2.0
        
        self.dexhand_dof_props = dexhand_dof_props
        
                          
        self.dexhand_dof_lower_limits = torch.tensor(
            [dexhand_dof_props["lower"][i] for i in range(self.num_dexhand_dofs)], 
            device=self.sim_device
        )
        self.dexhand_dof_upper_limits = torch.tensor(
            [dexhand_dof_props["upper"][i] for i in range(self.num_dexhand_dofs)], 
            device=self.sim_device
        )

    def load_object_asset(self, obj_urdf_path):
        """Load object asset"""
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.density = 100
        asset_options.thickness = 0.002
        asset_options.vhacd_enabled = False
        
        self.obj_asset = self.gym.load_asset(self.sim, *os.path.split(obj_urdf_path), asset_options)
        
                             
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(self.obj_asset)
        for element in rigid_shape_props_asset:
            element.friction = 1.0
            element.rolling_friction = 0.1
            element.torsion_friction = 0.1
        self.gym.set_asset_rigid_shape_properties(self.obj_asset, rigid_shape_props_asset)

    def load_table_asset(self):
        """Load table asset"""
                                                                  
        table_width_offset = 0.2
        table_dims = gymapi.Vec3(0.8 + table_width_offset, 1.6, 0.03)
        
        asset_options = gymapi.AssetOptions()
                               
        asset_options.fix_base_link = True
        
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, asset_options)
        
    def create_environment(self, initial_obj_pose):
        """Create single environment"""
                            
        env_lower = gymapi.Vec3(-1.0, -1.0, 0.0)
        env_upper = gymapi.Vec3(1.0, 1.0, 2.0)
        
        env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        
                                                                
        table_pose = gymapi.Transform()
                                           
        table_width_offset = 0.2
        table_pose.p = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        table_handle = self.gym.create_actor(env, self.table_asset, table_pose, "table", 0, 0)
        
                          
        self.gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.3, 0.3, 0.3))
        
                       
        table_half_height = 0.015
        cprint(f"桌子设置位置: (-0.1, 0, 0.4)", "yellow")
        cprint(f"桌子实际位置: {table_pose.p}", "yellow")
        cprint(f"桌子底部高度: {0.4 - table_half_height:.3f}m", "yellow")
        cprint(f"桌子顶部高度(桌面): {0.4 + table_half_height:.3f}m", "yellow")
        cprint(f"地面高度: 0.000m", "yellow")
        
                               
        hand_pose = gymapi.Transform()
        hand_pose.p = gymapi.Vec3(0, 0, 0.45)                        
        hand_pose.r = gymapi.Quat(0, 0, 0, 1)
        
        self.hand_handle = self.gym.create_actor(
            env, self.dexhand_asset, hand_pose, "dexhand", 0, 
            (1 if self.dexhand.self_collision else 0)
        )
        
                                     
        self.gym.set_actor_dof_properties(env, self.hand_handle, self.dexhand_dof_props)
        
                       
        self.obj_handle = self.gym.create_actor(env, self.obj_asset, initial_obj_pose, "object", 0, 0)
        
        self.env = env
        
                                         
        self.hand_rigid_body_handles = {
            body_name: self.gym.find_actor_rigid_body_handle(env, self.hand_handle, body_name)
            for body_name in self.dexhand.body_names
        }

    def setup_tensors(self):
        """Setup PyTorch tensors for simulation state"""
                           
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
                      
        self.root_state = gymtorch.wrap_tensor(_actor_root_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(_dof_state_tensor)
        self.rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor)
        
        cprint(f"Root state shape: {self.root_state.shape}", "yellow")
        cprint(f"DOF state shape: {self.dof_state.shape}", "yellow")
        cprint(f"Hand DOFs: {self.num_dexhand_dofs}", "yellow")

    def load_retargeting_data(self, data_idx):
        """Load retargeting data and original data"""
                                
        dataset_type = ManipDataFactory.dataset_type(data_idx)
        cprint(f"Dataset type: {dataset_type}", "blue")
        
                                                                        
        original_data = ManipDataFactory.create_data(
            manipdata_type=dataset_type,
            side="right" if str(self.dexhand).endswith("_rh") else "left",
            device=self.sim_device,
            mujoco2gym_transf=torch.eye(4, device=self.sim_device),
            dexhand=self.dexhand,
            verbose=False,
        )
        
        demo_data = original_data[data_idx]
        
                                                     
        obj_urdf_path = demo_data["obj_urdf_path"]
        obj_trajectory = demo_data["obj_trajectory"]
        
                                                        
        obj_trajectory_gym = self.mujoco2gym_transf @ obj_trajectory
        
        cprint(f"Object URDF: {obj_urdf_path}", "green")
        cprint(f"Object trajectory shape: {obj_trajectory_gym.shape}", "green")
        
                                                                  
        original_data_path = demo_data.get("data_path", None)
        if isinstance(original_data_path, (list, tuple)):
            original_data_path = original_data_path[0]
        
                                                                    
        retargeting_data = self.load_retargeting_file(data_idx, dataset_type, original_data_path)
        
        return retargeting_data, obj_urdf_path, obj_trajectory_gym

    def load_retargeting_file(self, data_idx, dataset_type, original_data_path=None):
        """Load retargeting pickle file"""
                                                               
        if dataset_type == "grabdemo":
            retargeting_path = f"data/retargeting/grab_demo/mano2{str(self.dexhand)}/102_sv_dict.pkl"
        elif dataset_type == "oakink2":
                                                                                
            if original_data_path is None:
                raise ValueError("original_data_path is required for OakInk-v2 dataset")
            
                                                                       
            stage = int(data_idx.split("@")[1])
            
                                                                      
            original_filename = os.path.split(original_data_path)[-1]
            retargeting_filename = original_filename.replace(".pkl", f"@{stage}.pkl")
            retargeting_path = f"data/retargeting/OakInk-v2/mano2{str(self.dexhand)}/{retargeting_filename}"
            
            cprint(f"Original data path: {original_data_path}", "cyan")
            cprint(f"Retargeting filename: {retargeting_filename}", "cyan")
        elif dataset_type == "favor":
            retargeting_path = f"data/retargeting/favor_pass1/mano2{str(self.dexhand)}/{data_idx}.pkl"
        elif dataset_type == "oakink2_mirrored":
            if original_data_path is None:
                raise ValueError("original_data_path is required for OakInk-v2 mirrored dataset")
            
            stage = int(data_idx.split("@")[1])
            original_filename = os.path.split(original_data_path)[-1]
            retargeting_filename = original_filename.replace(".pkl", f"@{stage}.pkl")
            retargeting_path = f"data/retargeting/OakInk-v2-mirrored/mano2{str(self.dexhand)}/{retargeting_filename}"
        elif dataset_type == "favor_mirrored":
            retargeting_path = f"data/retargeting/favor_pass1-mirrored/mano2{str(self.dexhand)}/{data_idx}.pkl"
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        cprint(f"Loading retargeting data from: {retargeting_path}", "cyan")
        
        if not os.path.exists(retargeting_path):
            raise FileNotFoundError(f"Retargeting file not found: {retargeting_path}")
        
        with open(retargeting_path, "rb") as f:
            retargeting_data = pickle.load(f)
        
        cprint(f"Retargeting data keys: {list(retargeting_data.keys())}", "cyan")
        for key, value in retargeting_data.items():
            if isinstance(value, np.ndarray):
                cprint(f"  {key}: shape {value.shape}", "cyan")
        
        return retargeting_data

    def run_visualization(self, retargeting_data, obj_trajectory_gym):
        """Run the visualization"""
                                             
        opt_wrist_pos = torch.tensor(retargeting_data["opt_wrist_pos"], device=self.sim_device, dtype=torch.float32)
        opt_wrist_rot = torch.tensor(retargeting_data["opt_wrist_rot"], device=self.sim_device, dtype=torch.float32)
        opt_dof_pos = torch.tensor(retargeting_data["opt_dof_pos"], device=self.sim_device, dtype=torch.float32)
        
        num_frames = opt_wrist_pos.shape[0]
        cprint(f"Playing {num_frames} frames", "green")
        
        frame_idx = 0
        
        while True:
                           
            hand_pos = opt_wrist_pos[frame_idx]
            hand_quat = aa_to_quat(opt_wrist_rot[frame_idx])
            hand_quat_gym = torch.tensor([hand_quat[1], hand_quat[2], hand_quat[3], hand_quat[0]], 
                                       device=self.sim_device)                                   
            
                                                                            
            self.root_state[1, :3] = hand_pos
            self.root_state[1, 3:7] = hand_quat_gym
            self.root_state[1, 7:] = 0                   
            
                                         
            if frame_idx == 0:
                obj_pos = obj_trajectory_gym[0, :3, 3]
                obj_rotmat = obj_trajectory_gym[0, :3, :3]
                obj_quat = rotmat_to_quat(obj_rotmat)
                obj_quat_gym = torch.tensor([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]], 
                                          device=self.sim_device)
                
                self.root_state[2, :3] = obj_pos
                self.root_state[2, 3:7] = obj_quat_gym
                self.root_state[2, 7:] = 0                   
            
                                               
            dof_targets = torch.clamp(opt_dof_pos[frame_idx], self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)
            
                                                                 
            if self.dexhand.name == "franka_panda":
                if dof_targets.shape[-1] >= 2:
                    dof_targets[1] = dof_targets[0]                        
            
                          
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_state))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(dof_targets))
            
                             
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            
                            
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            
                    
            if not self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                
                                        
                if self.gym.query_viewer_has_closed(self.viewer):
                    break
            
            self.gym.sync_frame_time(self.sim)
            
                           
            frame_idx = (frame_idx + 1) % num_frames
            
                                                      
            if self.headless and frame_idx == 0:
                cprint("Completed one full cycle in headless mode", "green")
                break

    def setup_camera(self):
        """Setup camera view"""
        if not self.headless:
                         
            cam_pos = gymapi.Vec3(1.5, 1.5, 1.2)          
            cam_target = gymapi.Vec3(0.0, 0.0, 0.45)          
            self.gym.viewer_camera_look_at(self.viewer, self.env, cam_pos, cam_target)

    def cleanup(self):
        """Cleanup resources"""
        if not self.headless and hasattr(self, 'viewer'):
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)


def main():
                                                      
    custom_params = [
        {
            "name": "--data_idx",
            "type": str,
            "default": "g0",
            "help": "Data index to visualize (e.g., g0, 1906, scene_01__A001++seq__9decc2b1d56c0e6351a8__2023-04-27-18-48-27@0)",
        },
        {
            "name": "--dexhand",
            "type": str,
            "default": "inspire",
            "help": "Dexterous hand type (inspire, allegro, franka_panda, etc.)",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "help": "Run without GUI",
        },
    ]
    
    args = gymutil.parse_arguments(
        description="Visualize Retargeted Dexterous Hand Manipulation",
        custom_parameters=custom_params,
    )
    
                 
    side = "right"                         
    dexhand = DexHandFactory.create_hand(args.dexhand, side)
    
    cprint(f"Using hand: {dexhand}", "blue")
    cprint(f"Data index: {args.data_idx}", "blue")
    cprint(f"Headless mode: {getattr(args, 'headless', False)}", "blue")
    
    visualizer = None                                  
    
    try:
                           
        visualizer = RetargetingVisualizer(args, dexhand)
        
                   
        retargeting_data, obj_urdf_path, obj_trajectory_gym = visualizer.load_retargeting_data(args.data_idx)
        
                     
        visualizer.load_dexhand_asset()
        visualizer.load_object_asset(obj_urdf_path)
        visualizer.load_table_asset()
        
                                                     
        initial_obj_pose = gymapi.Transform()
        initial_obj_pose.p = gymapi.Vec3(
            float(obj_trajectory_gym[0, 0, 3]),
            float(obj_trajectory_gym[0, 1, 3]),
            float(obj_trajectory_gym[0, 2, 3])
        )
        obj_rotmat = obj_trajectory_gym[0, :3, :3]
        obj_quat = rotmat_to_quat(obj_rotmat)
        initial_obj_pose.r = gymapi.Quat(float(obj_quat[1]), float(obj_quat[2]), float(obj_quat[3]), float(obj_quat[0]))
        
        visualizer.create_environment(initial_obj_pose)
        
                      
        visualizer.setup_camera()
        
                            
        visualizer.gym.prepare_sim(visualizer.sim)
        
                       
        visualizer.setup_tensors()
        
        cprint("Starting visualization...", "green")

        visualizer.gym.refresh_actor_root_state_tensor(visualizer.sim)
        
                           
        visualizer.run_visualization(retargeting_data, obj_trajectory_gym)
        
    except Exception as e:
        cprint(f"Error: {e}", "red")
        import traceback
        traceback.print_exc()
    finally:
                 
        if visualizer is not None:
            visualizer.cleanup()


if __name__ == "__main__":
    main()