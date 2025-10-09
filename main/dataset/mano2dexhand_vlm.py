import math
import os
import pickle
from isaacgym import gymapi, gymtorch, gymutil
import logging

logging.getLogger("gymapi").setLevel(logging.CRITICAL)
logging.getLogger("gymtorch").setLevel(logging.CRITICAL)
logging.getLogger("gymutil").setLevel(logging.CRITICAL)

import numpy as np
import pytorch_kinematics as pk
import torch
from termcolor import cprint
import base64
from openai import OpenAI
from PIL import Image

from main.dataset.factory import ManipDataFactory
from main.dataset.transform import (
    aa_to_quat,
    aa_to_rotmat,
    quat_to_rotmat,
    rot6d_to_aa,
    rot6d_to_quat,
    rot6d_to_rotmat,
    rotmat_to_aa,
    rotmat_to_quat,
    rotmat_to_rot6d,
)
from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory

API_SECRET_KEY = ""
BASE_URL = ""

def load_single_frame(dataset: str, data_idx: str, frame_idx: int, side="right"):
    """
    dataset: oakink2 / favor / grabdemo ...
    data_idx: 3e29c@1
    frame_idx: The frame index to get from the sequence (starts from 0)
    """
    # Create dexhand object without importing at module level
    dexhand = DexHandFactory.create_hand("mano", side)
    
    device = "cuda:0"
    mujoco2gym_transf = torch.eye(4, device=device)
    mujoco2gym_transf[:3, :3] = torch.tensor(aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0])), dtype=torch.float32, device=device)
    table_surface_z = 0.4 + 0.015  # table_pos.z + table_half_height
    mujoco2gym_transf[:3, 3] = torch.tensor([0, 0, table_surface_z], dtype=torch.float32, device=device)
    
    demo_d = ManipDataFactory.create_data(
        manipdata_type=dataset,
        side=side,
        device=device,
        mujoco2gym_transf=mujoco2gym_transf,
        dexhand=dexhand,
        verbose=False,
    )
    
    sequence_data = demo_d[data_idx]


    
    packed_data = pack_data([sequence_data], dexhand)
    
    mano_joints = packed_data["mano_joints"]
    obj_urdf = packed_data["obj_urdf_path"]
    obj_trajectory = packed_data["obj_trajectory"]
    
    print("=== MANO joint order debug ===")
    filtered_body_names = [j_name for j_name in dexhand.body_names 
                          if dexhand.to_hand(j_name)[0] != "wrist"]
    print(f"Number of filtered body_names: {len(filtered_body_names)}")
    print("Filtered joint order:")
    for i, name in enumerate(filtered_body_names):
        print(f"  Index {i}: {name} -> {dexhand.to_hand(name)[0]}")
    print("========================")

    print("mano_joints.shape",mano_joints.shape)
    
    if isinstance(mano_joints, torch.Tensor) and mano_joints.dim() > 1:
        if frame_idx < mano_joints.shape[0]:
            mano_joints = mano_joints[frame_idx].reshape(1, 27, 3)
            if isinstance(obj_trajectory, torch.Tensor) and obj_trajectory.dim() > 2:
                obj_transform = obj_trajectory[frame_idx]  # Get object transform for the corresponding frame
            else:
                obj_transform = obj_trajectory if isinstance(obj_trajectory, torch.Tensor) else torch.eye(4)
        else:
            raise IndexError(f"frame_idx {frame_idx} >= sequence length {mano_joints.shape[0]}")
    elif isinstance(mano_joints, torch.Tensor):
        mano_joints = mano_joints.reshape(1, 27, 3)
        obj_transform = obj_trajectory if isinstance(obj_trajectory, torch.Tensor) else torch.eye(4)
    
    if isinstance(obj_urdf, (list, tuple, torch.Tensor)):
        obj_urdf = obj_urdf[0] if len(obj_urdf) > 0 else obj_urdf
    

    print("frame_idx",frame_idx)
    
    return mano_joints, obj_urdf, obj_transform


def load_sequence_data(dataset: str, data_idx: str, side="right"):

    # Create dexhand object without importing at module level
    dexhand = DexHandFactory.create_hand("mano", side)
    
    device = "cuda:0"
    mujoco2gym_transf = torch.eye(4, device=device)
    mujoco2gym_transf[:3, :3] = torch.tensor(aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0])), dtype=torch.float32, device=device)
    table_surface_z = 0.4 + 0.015  # table_pos.z + table_half_height
    mujoco2gym_transf[:3, 3] = torch.tensor([0, 0, table_surface_z], dtype=torch.float32, device=device)

    demo_d = ManipDataFactory.create_data(
        manipdata_type=dataset,
        side=side,
        device=device,
        mujoco2gym_transf=mujoco2gym_transf,
        dexhand=dexhand,
        verbose=False,
    )
    
    sequence_data = demo_d[data_idx]

    packed_data = pack_data([sequence_data], dexhand)
    
    mano_joints = packed_data["mano_joints"]
    obj_urdf = packed_data["obj_urdf_path"]
    obj_trajectory = packed_data["obj_trajectory"]
    
    num_frames = mano_joints.shape[0]
    mano_joints = mano_joints.reshape(num_frames, 27, 3)
    
    if isinstance(obj_urdf, (list, tuple, torch.Tensor)):
        obj_urdf = obj_urdf[0] if len(obj_urdf) > 0 else obj_urdf
    
    return mano_joints, obj_urdf, obj_trajectory
    
    
    
    




def encode_image(image_path):
    """Encodes an image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

GRIPPER_PROMPTS = {
    "franka_panda": {
        "function_name": "process_mano_to_hand_mapping",
        "output_description": "numpy array shape (N, 4, 3) - N sets of mapping points for [panda_leftfinger, panda_leftfinger_tip, panda_rightfinger, panda_rightfinger_tip]",
        "specific_instructions": """
        For Franka Panda gripper mapping - Two-step process:
        
        STEP 1: Generate two optimal grasp points based on the object and hand pose
        - Analyze the MANO hand joints to understand the grasping intention
        - Identify two key contact points that would provide stable grasping
        - These should be positioned where the gripper fingers will make contact with the object
        
        STEP 2: Calculate finger positions from grasp points and wrist
        - Use the two grasp points and wrist position to determine finger base and tip positions
        - panda_leftfinger: position between wrist and left grasp point
        - panda_leftfinger_tip: the left grasp point itself
        - panda_rightfinger: position between wrist and right grasp point  
        - panda_rightfinger_tip: the right grasp point itself
        - CONSTRAINT: panda_leftfinger to panda_leftfinger_tip line must be parallel to panda_rightfinger to panda_rightfinger_tip line
        - Ensure finger positions form a realistic gripper configuration""",
        "body_names": ["panda_leftfinger", "panda_leftfinger_tip", "panda_rightfinger", "panda_rightfinger_tip"]
    },
    "inspire": {
        "function_name": "process_mano_to_hand_mapping", 
        "output_description": "numpy array shape (N, 26, 3) - N sets of mapping points for all finger joints excluding wrist",
        "specific_instructions": """
        For multi-finger hand mapping:
        - Generate mapping points for all finger joints excluding wrist
        - Focus on fingertips for primary contact points
        - Consider the dexterous manipulation capabilities""",
        "body_names": []  # Will be filled dynamically
    },
    "default": {
        "function_name": "process_mano_to_hand_mapping",
        "output_description": "numpy array shape (N, num_points, 3) - N sets of mapping points for gripper/hand components",
        "specific_instructions": """
        For general gripper/hand mapping:
        - Generate mapping points based on the gripper/hand structure
        - Exclude base/wrist components
        - Focus on end-effector contact points""",
        "body_names": []
    }
}

class CodeGenerationChain:
    """Code generation chain: VLM generates the processing function code"""
    def __init__(self, api_key, base_url, gripper_type="default"):
        import httpx
        http_client = httpx.Client(proxies={})
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        self.gripper_type = gripper_type
        
    def invoke(self, input_data):
        mano_pts = input_data["mano_joints"]
        image_b64 = input_data["image"] 
        task = input_data["task"]
        
        gripper_config = GRIPPER_PROMPTS.get(self.gripper_type, GRIPPER_PROMPTS["default"])
        
        if mano_pts.ndim == 3:
            vlm_frame = mano_pts[0]
        else:
            vlm_frame = mano_pts
        
        joints_str = "\n".join([f"{i}: {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
                                for i, p in enumerate(vlm_frame)])
        
        prompt = f"""YOU MUST GENERATE A PYTHON FUNCTION NAMED '{gripper_config['function_name']}'. THIS IS CRITICAL!

Task: {task}
MANO joints (single frame example):
{joints_str}

GRIPPER TYPE: {self.gripper_type.upper()}
{gripper_config['specific_instructions']}

MANDATORY FUNCTION TEMPLATE - YOU MUST FOLLOW THIS EXACTLY:

def {gripper_config['function_name']}(mano_joints, task_context):
    import numpy as np
    # YOUR CODE HERE
    return mapping_points

REQUIREMENTS (MUST FOLLOW ALL):
1. FUNCTION NAME MUST BE: {gripper_config['function_name']}
2. Input: mano_joints (numpy array shape (N, 27, 3) or (1, 27, 3)), task_context (string)
3. Output: {gripper_config['output_description']}
4. Handle both single frame (1, 27, 3) and sequence (N, 27, 3)
5. Coordinates in object frame, unit: meters
6. Only import numpy as np if needed
7. No other imports or dependencies

CRITICAL RULES:
- START with "def {gripper_config['function_name']}(mano_joints, task_context):"
- END with "return mapping_points"
- NO markdown formatting, NO ```python```, NO explanations
- OUTPUT ONLY THE FUNCTION CODE

EXAMPLE OUTPUT FORMAT:
def {gripper_config['function_name']}(mano_joints, task_context):
    import numpy as np
    # process the joints to find mapping points
    return mapping_points_array"""

        messages = [{"role": "user",
                     "content": [{"type": "text", "text": prompt},
                                 {"type": "image_url", 
                                  "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]}]
        
        response = self.client.chat.completions.create(
            # model="gemini-2.5-flash-preview-05-20",
            model="gemini-2.5-pro-preview-06-05",
            # model="gpt-4o-mini",
            messages=messages
        )
        
        return {"generated_code": response.choices[0].message.content.strip()}


class CodeExecutionChain:
    """Code execution chain: executes the generated function code"""
    def invoke(self, input_data):
        generated_code = input_data["generated_code"]
        mano_joints = input_data["mano_joints"]
        task = input_data["task"]
        
        clean_code = self._extract_python_code(generated_code)
        print("=== Cleaned code ===")
        print(clean_code)
        print("=================")
        
        safe_globals = {
            "__builtins__": {
                "tuple": tuple,
                "list": list, 
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "float": float,
                "int": int,
                "str": str,
                "bool": bool,
                "__import__": __import__
            },
            "numpy": np, 
            "np": np
        }
        local_vars = {}
        
        try:
            exec(clean_code, safe_globals, local_vars)
            
            if "process_mano_to_hand_mapping" in local_vars:
                func = local_vars["process_mano_to_hand_mapping"]
                result = func(mano_joints, task)
                return result
            else:
                raise ValueError("Function 'process_mano_to_hand_mapping' not found in generated code")
                
        except Exception as e:
            print(f"Code execution error: {e}")
            raise
    
    def _extract_python_code(self, text):
        """Extracts pure Python code, removing markdown formatting"""
        import re
        
        python_block = re.search(r'```python\s*\n(.*?)```', text, re.DOTALL)
        if python_block:
            return python_block.group(1).strip()
        
        code_block = re.search(r'```\s*\n(.*?)```', text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()
        
        lines = text.split('\n')
        code_lines = []
        in_function = False
        
        for line in lines:
            if line.strip().startswith('def process_mano_to_hand_mapping'):
                in_function = True
            
            if in_function:
                code_lines.append(line)
                
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        clean_text = re.sub(r'```\w*', '', text)
        clean_text = re.sub(r'```', '', clean_text)
        return clean_text.strip()




def vlm_mano_to_hand_mapping(training_mano_pts, full_mano_pts, rgb_path, task, gripper_type="default"):
    """Use VLM to get the grasp point generation function and process the complete data
    Args:
        training_mano_pts: Single frame data for training VLM (1, 27, 3)
        full_mano_pts: Complete data to be processed (N, 27, 3) or (1, 27, 3)
        rgb_path: Image path
        task: Task description
        gripper_type: Gripper type (default is "default")
    Returns:
        Grasp point result (N, 2, 3)
    """
    if hasattr(training_mano_pts, 'cpu'):
        training_mano_pts_cpu = training_mano_pts.cpu().numpy()
    else:
        training_mano_pts_cpu = training_mano_pts
        
    if hasattr(full_mano_pts, 'cpu'):
        full_mano_pts_cpu = full_mano_pts.cpu().numpy()
    else:
        full_mano_pts_cpu = full_mano_pts
    
    
    training_input = {
        "mano_joints": training_mano_pts_cpu,
        "image": encode_image(rgb_path),
        "task": task
    }
    
    code_gen = CodeGenerationChain(API_SECRET_KEY, BASE_URL, gripper_type)
    gen_result = code_gen.invoke(training_input)

    print("gen_result", gen_result)
    
    exec_input = {
        "generated_code": gen_result["generated_code"],
        "mano_joints": full_mano_pts_cpu,
        "task": task
    }
    
    code_exec = CodeExecutionChain()
    result = code_exec.invoke(exec_input)
    
    print("=== VLM generated function output ===")
    print(f"Training data shape: {training_mano_pts_cpu.shape}")
    print(f"Execution data shape: {full_mano_pts_cpu.shape}")
    print(f"Grasp points result: {np.array(result).shape}")
    print("====================")
    
    return result


def pack_data(data, dexhand):
    packed_data = {}
    for k in data[0].keys():
        if k == "mano_joints":
            mano_joints = []
            for d in data:
                mano_joints.append(
                    torch.concat(
                        [
                            d[k][dexhand.to_hand(j_name)[0]]
                            for j_name in dexhand.body_names
                            if dexhand.to_hand(j_name)[0] != "wrist"
                        ],
                        dim=-1,
                    )
                )
            packed_data[k] = torch.stack(mano_joints).squeeze()
        elif type(data[0][k]) == torch.Tensor:
            packed_data[k] = torch.stack([d[k] for d in data]).squeeze(0)
        elif type(data[0][k]) == np.ndarray:
            packed_data[k] = np.stack([d[k] for d in data]).squeeze(0)
        else:
            packed_data[k] = [d[k] for d in data]
    return packed_data


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


def visualize_vlm_hand_mapping(pts, mano2dexhand_instance, env_idx=0):
    """
    Function to visualize VLM grasp points (using pre-created spheres)
    Args:
        pts: List of grasp point coordinates [(x1,y1,z1), (x2,y2,z2), ...]
        mano2dexhand_instance: Mano2Dexhand instance containing pre-created visualization spheres
        env_idx: Environment index, default is 0
    """
    if len(pts) == 0:
        print("Warning: No grasp points to visualize")
        return
    
    if env_idx not in mano2dexhand_instance.mano_to_hand_mapping_vis_actors:
        print(f"Error: Environment index {env_idx} does not have pre-created visualization spheres")
        return
    
    gym = mano2dexhand_instance.gym
    env = mano2dexhand_instance.envs[env_idx]
    
    # Get the number of available spheres
    num_available_spheres = len(mano2dexhand_instance.mano_to_hand_mapping_vis_actors[env_idx])
    num_points = len(pts)
    
    print(f"Visualizing {min(num_points, num_available_spheres)} grasp points")
    
    for i, pt in enumerate(pts[:num_available_spheres]):
        if i in mano2dexhand_instance.mano_to_hand_mapping_vis_actors[env_idx]:
            vis_actor = mano2dexhand_instance.mano_to_hand_mapping_vis_actors[env_idx][i]
            
            # Get the root state index of the current actor
            actor_idx = gym.get_actor_index(env, vis_actor, gymapi.DOMAIN_SIM)
            
            # Update position to the _root_state tensor
            mano2dexhand_instance._root_state[env_idx, actor_idx, :3] = torch.tensor(
                [float(pt[0]), float(pt[1]), float(pt[2])], 
                device=mano2dexhand_instance.sim_device, dtype=torch.float32
            )
    
    # Hide extra spheres (move them far away)
    for i in range(num_points, num_available_spheres):
        if i in mano2dexhand_instance.mano_to_hand_mapping_vis_actors[env_idx]:
            vis_actor = mano2dexhand_instance.mano_to_hand_mapping_vis_actors[env_idx][i]
            actor_idx = gym.get_actor_index(env, vis_actor, gymapi.DOMAIN_SIM)
            mano2dexhand_instance._root_state[env_idx, actor_idx, :3] = torch.tensor(
                [100.0, 100.0, 100.0], 
                device=mano2dexhand_instance.sim_device, dtype=torch.float32
            )
    
    gym.set_actor_root_state_tensor(
        mano2dexhand_instance.sim, 
        gymtorch.unwrap_tensor(mano2dexhand_instance._root_state)
    )
    
    print(f"✓ Updated VLM grasp point visualization:")
    colors = ["red", "blue", "green", "yellow", "purple", "orange"]
    for i, pt in enumerate(pts[:num_available_spheres]):
        color = colors[i] if i < len(colors) else f"sphere_{i}"
        print(f"  Point {i+1} ({color}): ({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})")
    
    if num_points > num_available_spheres:
        print(f"Warning: Only displaying first {num_available_spheres} points, total {num_points} points")


class Mano2Dexhand:
    def __init__(self, args, dexhand, obj_urdf_path):
        self.gym = gymapi.acquire_gym()
        self.sim_params = gymapi.SimParams()
        self.dexhand = dexhand

        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        self.headless = args.headless
        if self.headless:
            self.graphics_device_id = -1

        assert args.physics_engine == gymapi.SIM_PHYSX

        self.sim_params.substeps = 1
        self.sim_params.physx.solver_type = 1
        self.sim_params.physx.num_position_iterations = 4
        self.sim_params.physx.num_velocity_iterations = 1
        self.sim_params.physx.num_threads = args.num_threads
        self.sim_params.physx.use_gpu = args.use_gpu

        self.sim_params.use_gpu_pipeline = args.use_gpu_pipeline
        self.sim_device = args.sim_device if args.use_gpu_pipeline else "cpu"

        self.sim = self.gym.create_sim(
            args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params
        )

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())

        asset_root = os.path.split(self.dexhand.urdf_path)[0]
        asset_file = os.path.split(self.dexhand.urdf_path)[1]

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = False
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        dexhand_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.chain = pk.build_chain_from_urdf(open(os.path.join(asset_root, asset_file), 'rb').read())
        self.chain = self.chain.to(dtype=torch.float32, device=self.sim_device)

        dexhand_dof_stiffness = torch.tensor(
            [10] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        dexhand_dof_damping = torch.tensor(
            [1] * self.dexhand.n_dofs,
            dtype=torch.float,
            device=self.sim_device,
        )
        self.limit_info = {}
        asset_rh_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        self.limit_info["rh"] = {
            "lower": np.asarray(asset_rh_dof_props["lower"]).copy().astype(np.float32),
            "upper": np.asarray(asset_rh_dof_props["upper"]).copy().astype(np.float32),
        }

        self.num_dexhand_bodies = self.gym.get_asset_rigid_body_count(dexhand_asset)
        self.num_dexhand_dofs = self.gym.get_asset_dof_count(dexhand_asset)

        dexhand_dof_props = self.gym.get_asset_dof_properties(dexhand_asset)
        rigid_shape_rh_props_asset = self.gym.get_asset_rigid_shape_properties(dexhand_asset)
        for element in rigid_shape_rh_props_asset:
            element.friction = 0.0001
            element.rolling_friction = 0.0001
            element.torsion_friction = 0.0001
        self.gym.set_asset_rigid_shape_properties(dexhand_asset, rigid_shape_rh_props_asset)

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
        default_dof_state = np.ones(self.num_dexhand_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] *= np.pi / 50
        

        self.dexhand_default_dof_pos = default_dof_state
        self.dexhand_default_pose = gymapi.Transform()
        self.dexhand_default_pose.p = gymapi.Vec3(0, 0, 0)
        self.dexhand_default_pose.r = gymapi.Quat(0, 0, 0, 1)

        table_width_offset = 0.2
        mujoco2gym_transf = np.eye(4)
        mujoco2gym_transf[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(
            np.array([np.pi / 2, 0, 0])
        )
        table_pos = gymapi.Vec3(-table_width_offset / 2, 0, 0.4)
        self.dexhand_pose = gymapi.Transform()
        table_half_height = 0.015
        self._table_surface_z = table_pos.z + table_half_height
        mujoco2gym_transf[:3, 3] = np.array([0, 0, self._table_surface_z])
        self.mujoco2gym_transf = torch.tensor(mujoco2gym_transf, device=self.sim_device, dtype=torch.float32)

        self.num_envs = args.num_envs
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_options = gymapi.AssetOptions()
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        asset_options.thickness = 0.001
        asset_options.fix_base_link = True
        asset_options.vhacd_enabled = False
        asset_options.disable_gravity = True
        asset_options.density = 200

        current_asset = self.gym.load_asset(self.sim, *os.path.split(obj_urdf_path), asset_options)

        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(current_asset)
        for element in rigid_shape_props_asset:
            element.friction = 0.00001
        self.gym.set_asset_rigid_shape_properties(current_asset, rigid_shape_props_asset)

        self.envs = []
        self.hand_idxs = []
        self.mano_to_hand_mapping_vis_actors = {}

        for i in range(self.num_envs):
            # Create env
            env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)
            dexhand_actor = self.gym.create_actor(
                env,
                dexhand_asset,
                self.dexhand_default_pose,
                "dexhand",
                i,
                (1 if self.dexhand.self_collision else 0),
            )

            # Set initial DOF states
            self.gym.set_actor_dof_states(env, dexhand_actor, self.dexhand_default_dof_pos, gymapi.STATE_ALL)

            # Set DOF control properties
            self.gym.set_actor_dof_properties(env, dexhand_actor, dexhand_dof_props)

            self.obj_actor = self.gym.create_actor(env, current_asset, gymapi.Transform(), "manip_obj", i, 0)

            scene_asset_options = gymapi.AssetOptions()
            scene_asset_options.fix_base_link = True
            for joint_vis_id, joint_name in enumerate(self.dexhand.body_names):
                mapped_joint = self.dexhand.to_hand(joint_name)[0]
                # print(f"Creating sphere {joint_vis_id}: {joint_name} -> {mapped_joint}")
                joint_point = self.gym.create_sphere(self.sim, 0.005, scene_asset_options)
                a = self.gym.create_actor(
                    env, joint_point, gymapi.Transform(), f"mano_joint_{joint_vis_id}", self.num_envs + 1, 0b1
                )
                joint_name = mapped_joint
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
                self.gym.set_rigid_body_color(env, a, 0, gymapi.MESH_VISUAL, c)
                
            sphere_opts = gymapi.AssetOptions()
            sphere_opts.fix_base_link = True
            
            if self.dexhand.name == "franka_panda":
                num_vis_spheres = 4
            else:
                num_vis_spheres = max(2, len([j for j in self.dexhand.body_names if self.dexhand.to_hand(j)[0] != "wrist"]))
            
            colors = [
                gymapi.Vec3(1.0, 0.0, 0.0),
                gymapi.Vec3(0.0, 0.0, 1.0),
                gymapi.Vec3(0.0, 1.0, 0.0),
                gymapi.Vec3(1.0, 1.0, 0.0),
                gymapi.Vec3(1.0, 0.0, 1.0),
                gymapi.Vec3(0.0, 1.0, 1.0),
            ]
            
            for vis_idx in range(num_vis_spheres):
                vis_sphere = self.gym.create_sphere(self.sim, 0.015, sphere_opts)
                transform = gymapi.Transform()
                transform.p = gymapi.Vec3(100.0, 100.0, 100.0)
                vis_actor = self.gym.create_actor(
                    env, vis_sphere, transform, f"mano_to_hand_mapping_vis_{vis_idx}", 
                    self.num_envs + 10 + vis_idx, 0b1
                )
                
                color = colors[vis_idx % len(colors)]
                self.gym.set_rigid_body_color(env, vis_actor, 0, gymapi.MESH_VISUAL, color)
                
                if i not in self.mano_to_hand_mapping_vis_actors:
                    self.mano_to_hand_mapping_vis_actors[i] = {}
                self.mano_to_hand_mapping_vis_actors[i][vis_idx] = vis_actor

        env_ptr = self.envs[0]
        dexhand_handle = 0
        self.dexhand_handles = {
            k: self.gym.find_actor_rigid_body_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.body_names
        }
        
        self.dexhand_dof_handles = {
            k: self.gym.find_actor_dof_handle(env_ptr, dexhand_handle, k) for k in self.dexhand.dof_names
        }
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)

        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._net_cf = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
        self.q = self._dof_state[..., 0]
        self.qd = self._dof_state[..., 1]
        self._base_state = self._root_state[:, 0, :]

        self.isaac2chain_order = [
            self.gym.get_actor_dof_names(env_ptr, dexhand_handle).index(j)
            for j in self.chain.get_joint_parameter_names()
        ]

        self.mano_joint_points = [
            self._root_state[:, self.gym.find_actor_handle(env_ptr, f"mano_joint_{i}"), :]
            for i in range(len(self.dexhand.body_names))
        ]

        if not self.headless:
            cam_pos = gymapi.Vec3(4, 3, 3)
            cam_target = gymapi.Vec3(-4, -3, 0)
            middle_env = self.envs[self.num_envs // 2 + num_per_row // 2]
            self.gym.viewer_camera_look_at(self.viewer, middle_env, cam_pos, cam_target)

        self.gym.prepare_sim(self.sim)



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

    def fitting(self, max_iter, obj_trajectory, target_wrist_pos, target_wrist_rot, target_mano_joints, skip_coord_transform=False):

        assert target_mano_joints.shape[0] == self.num_envs
        target_wrist_pos = (self.mujoco2gym_transf[:3, :3] @ target_wrist_pos.T).T + self.mujoco2gym_transf[:3, 3]
        target_wrist_rot = self.mujoco2gym_transf[:3, :3] @ aa_to_rotmat(target_wrist_rot)
        
        if not skip_coord_transform:
            target_mano_joints = target_mano_joints.view(-1, 3)
            target_mano_joints = (self.mujoco2gym_transf[:3, :3] @ target_mano_joints.T).T + self.mujoco2gym_transf[:3, 3]
            target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)
            print("[INFO] Applied coordinate transformation to target_mano_joints")
        else:
            target_mano_joints = target_mano_joints.view(self.num_envs, -1, 3)
            print("[INFO] Skipped coordinate transformation for target_mano_joints (VLM coordinates)")

        obj_trajectory = self.mujoco2gym_transf @ obj_trajectory

        if target_mano_joints.shape[1] > 3:
            middle_pos = (target_mano_joints[:, 3] + target_wrist_pos) / 2
        else:
            middle_pos = ((target_mano_joints[:, 0]+target_mano_joints[:, 1])/2 + target_wrist_pos) / 2
                
        obj_pos = obj_trajectory[:, :3, 3]
        offset = middle_pos - obj_pos
        offset = offset / torch.norm(offset, dim=-1, keepdim=True) * 0.2

        opt_wrist_pos = torch.tensor(
            target_wrist_pos + offset,
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opt_wrist_rot = torch.tensor(
            rotmat_to_rot6d(target_wrist_rot), device=self.sim_device, dtype=torch.float32, requires_grad=True
        )
        opt_dof_pos = torch.tensor(
            self.dexhand_default_dof_pos["pos"][None].repeat(self.num_envs, axis=0),
            device=self.sim_device,
            dtype=torch.float32,
            requires_grad=True,
        )
        opti = torch.optim.Adam(
            [{"params": [opt_wrist_pos, opt_wrist_rot], "lr": 0.0008}, {"params": [opt_dof_pos], "lr": 0.0004}]
        )

        weight = []
        for k in self.dexhand.body_names:
            k = self.dexhand.to_hand(k)[0]
            
            if self.dexhand.name == "franka_panda" or self.dexhand.name == "sawyer":
                if "panda_hand" in k or "wrist" in k:
                    weight.append(4)
                elif "tip" in k:
                    weight.append(25)
                else:
                    weight.append(5)
            else:
                if "tip" in k:
                    if "index" in k:
                        weight.append(20)
                    elif "middle" in k:
                        weight.append(10)
                    elif "ring" in k:
                        weight.append(7)
                    elif "pinky" in k:
                        weight.append(5)
                    elif "thumb" in k:
                        weight.append(25)
                    else:
                        weight.append(10)
                elif "proximal" in k:
                    weight.append(1)
                elif "intermediate" in k:
                    weight.append(1)
                else:
                    weight.append(1)
        weight = torch.tensor(weight, device=self.sim_device, dtype=torch.float32)
        
        cprint(f"Hand type: {self.dexhand.name}", "blue")
        cprint(f"DOF count: {self.num_dexhand_dofs}", "blue")
        cprint(f"Body count: {len(self.dexhand.body_names)}", "blue")
        cprint(f"Weight shape: {weight.shape}", "blue")
        
        iter = 0
        past_loss = 1e10
        while (self.headless and iter < max_iter) or (
            not self.headless and not self.gym.query_viewer_has_closed(self.viewer)
        ):
            iter += 1

            opt_wrist_quat = rot6d_to_quat(opt_wrist_rot)[:, [1, 2, 3, 0]]
            opt_wrist_rotmat = rot6d_to_rotmat(opt_wrist_rot)
            self._root_state[:, 0, :3] = opt_wrist_pos.detach()
            self._root_state[:, 0, 3:7] = opt_wrist_quat.detach()
            self._root_state[:, 0, 7:] = torch.zeros_like(self._root_state[:, 0, 7:])
            self._root_state[:, self.obj_actor, :3] = obj_trajectory[:, :3, 3]
            self._root_state[:, self.obj_actor, 3:7] = rotmat_to_quat(obj_trajectory[:, :3, :3])[:, [1, 2, 3, 0]]

            opt_dof_pos_clamped = torch.clamp(opt_dof_pos, self.dexhand_dof_lower_limits, self.dexhand_dof_upper_limits)
            
            if self.dexhand.name == "franka_panda":
                if opt_dof_pos_clamped.shape[-1] >= 2:
                    opt_dof_pos_clamped[:, 1] = opt_dof_pos_clamped[:, 0]

            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(opt_dof_pos_clamped))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self._root_state))

            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            if not self.headless:
                self.gym.step_graphics(self.sim)

            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            
            if not self.headless:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, False)
            self.gym.sync_frame_time(self.sim)

            isaac_joints = torch.stack(
                [self._rigid_body_state[:, self.dexhand_handles[k], :3] for k in self.dexhand.body_names],
                dim=1,
            )

            ret = self.chain.forward_kinematics(opt_dof_pos_clamped[:, self.isaac2chain_order])
            pk_joints = torch.stack([ret[k].get_matrix()[:, :3, 3] for k in self.dexhand.body_names], dim=1)
            pk_joints = (rot6d_to_rotmat(opt_wrist_rot) @ pk_joints.transpose(-1, -2)).transpose(
                -1, -2
            ) + opt_wrist_pos[:, None]
            target_joints = torch.cat([target_wrist_pos[:, None], target_mano_joints], dim=1)
            for k in range(len(self.mano_joint_points)):
                self.mano_joint_points[k][:, :3] = target_joints[:, k]
            loss = torch.mean(torch.norm(pk_joints - target_joints, dim=-1) * weight[None])
            opti.zero_grad()
            loss.backward()
            opti.step()

            if iter % 100 == 0:
                cprint(f"{iter} {loss.item()}", "green")
                if iter > 1 and past_loss - loss.item() < 1e-7:
                    break
                past_loss = loss.item()
        to_dump = {
            "opt_wrist_pos": opt_wrist_pos.detach().cpu().numpy(),
            "opt_wrist_rot": rot6d_to_aa(opt_wrist_rot).detach().cpu().numpy(),
            "opt_dof_pos": opt_dof_pos_clamped.detach().cpu().numpy(),
            "opt_joints_pos": isaac_joints.detach().cpu().numpy(),
            "vlm_mano_joints": target_mano_joints.detach().cpu().numpy(),
        }

        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        return to_dump


if __name__ == "__main__":
    _parser = gymutil.parse_arguments(
        description="Mano to Dexhand",
        headless=True,
        custom_parameters=[
            {
                "name": "--iter",
                "type": int,
                "default": 4000,
            },
            {
                "name": "--data_idx",
                "type": str,
                "default": "1906",
            },
            {
                "name": "--dexhand",
                "type": str,
                "default": "inspire",
            },
            {
                "name": "--side",
                "type": str,
                "default": "right",
            },
            {
                "name": "--mode",
                "type": str,
                "default": "sequence",
                "choices": ["sequence", "single"],
                "help": "Optimization mode: 'sequence' for full sequence optimization, 'single' for single frame optimization",
            },
            {
                "name": "--frame_idx",
                "type": int,
                "default": 0,
                "help": "Frame index to optimize when using single frame mode (0-based indexing)",
            },
        ],
    )

    dexhand = DexHandFactory.create_hand(_parser.dexhand, _parser.side)

    def run(parser, idx):
        print(f"Running optimization in {parser.mode} mode")
        if parser.mode == "single":
            print(f"Optimizing frame index: {parser.frame_idx}")

        dataset_type = ManipDataFactory.dataset_type(idx)
        demo_d = ManipDataFactory.create_data(
            manipdata_type=dataset_type,
            side=parser.side,
            device="cuda:0",
            mujoco2gym_transf=torch.eye(4, device="cuda:0"),
            dexhand=dexhand,
            verbose=False,
        )

        if parser.mode == "single":
            full_sequence_data = pack_data([demo_d[idx]], dexhand)
            print(f"Single frame mode: extracting frame {parser.frame_idx}")
            frame_idx = parser.frame_idx
            num_frames = full_sequence_data['mano_joints'].shape[0]

            if not (0 <= frame_idx < num_frames):
                raise IndexError(f"Error: Frame index {frame_idx} is out of range. The sequence has {num_frames} frames (indexed from 0 to {num_frames-1}).")

            demo_data = {}
            for key, value in full_sequence_data.items():
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    demo_data[key] = value[frame_idx:frame_idx+1]
                elif isinstance(value, list):
                    if len(value) == num_frames:
                        demo_data[key] = [value[frame_idx]]
                    else:
                        demo_data[key] = value
        else:
            demo_data = pack_data([demo_d[idx]], dexhand)
            print(f"Sequence optimization: processing {demo_data['mano_joints'].shape[0]} frames")

        parser.num_envs = demo_data["mano_joints"].shape[0]

        mano2inspire = Mano2Dexhand(parser, dexhand, demo_data["obj_urdf_path"][0])

        if parser.mode == "single":
            mano_joints, obj_urdf, obj_transform = load_single_frame("oakink2",
                                                                    _parser.data_idx,
                                                                    frame_idx=_parser.frame_idx,
                                                                    side=_parser.side)
        else:   
            mano_joints, obj_urdf, obj_transform = load_sequence_data("oakink2",
                                                                    _parser.data_idx,
                                                                    side=_parser.side)
        
        print("Full data shape: ", mano_joints.shape)

        if parser.mode == "sequence":
            training_frame = mano_joints[parser.frame_idx:parser.frame_idx+1]
            print(f"Sequence mode: using frame {parser.frame_idx} for VLM training")
        else:
            training_frame = mano_joints
            print("Single frame mode: using current frame for VLM training")
        
        print("Training data shape: ", training_frame.shape)
        
        pts = vlm_mano_to_hand_mapping(
            training_frame, 
            mano_joints,
            "/data/yuantingyu/ManipTrans/ManipTrans/main/dataset/001628.png", 
            "Hold the mug and pour the contents from the taller blue mug into the bowl",
            parser.dexhand
        )
        print("VLM grasp points:\n", pts)
        
        pts_tensor = torch.tensor(pts, device="cuda:0", dtype=torch.float32)
        
        to_dump = mano2inspire.fitting(
            parser.iter,
            demo_data["obj_trajectory"],
            demo_data["wrist_pos"],
            demo_data["wrist_rot"],
            pts_tensor,
            skip_coord_transform=True
        )

        base_filename = os.path.split(demo_data['data_path'][0])[-1]
        
        if dataset_type == "oakink2":
            if parser.mode == "single":
                dump_path = f"data/retargeting/OakInk-v2/mano2{str(dexhand)}/{base_filename.replace('.pkl', f'@{idx[-1]}_frame{parser.frame_idx}.pkl')}"
            else:
                dump_path = f"data/retargeting/OakInk-v2/mano2{str(dexhand)}/{base_filename.replace('.pkl', f'@{idx[-1]}.pkl')}"
        elif dataset_type == "favor":
            if parser.mode == "single":
                dump_path = f"data/retargeting/favor_pass1/mano2{str(dexhand)}/{base_filename.replace('.pkl', f'_frame{parser.frame_idx}.pkl')}"
            else:
                dump_path = f"data/retargeting/favor_pass1/mano2{str(dexhand)}/{base_filename}"
        elif dataset_type == "grabdemo":
            if parser.mode == "single":
                dump_path = f"data/retargeting/grab_demo/mano2{str(dexhand)}/{base_filename.replace('.npy', f'_frame{parser.frame_idx}.pkl')}"
            else:
                dump_path = f"data/retargeting/grab_demo/mano2{str(dexhand)}/{base_filename.replace('.npy', '.pkl')}"
        elif dataset_type == "oakink2_mirrored":
            if parser.mode == "single":
                dump_path = f"data/retargeting/OakInk-v2-mirrored/mano2{str(dexhand)}/{base_filename.replace('.pkl', f'@{idx[-1]}_frame{parser.frame_idx}.pkl')}"
            else:
                dump_path = f"data/retargeting/OakInk-v2-mirrored/mano2{str(dexhand)}/{base_filename.replace('.pkl', f'@{idx[-1]}.pkl')}"
        elif dataset_type == "favor_mirrored":
            if parser.mode == "single":
                dump_path = f"data/retargeting/favor_pass1-mirrored/mano2{str(dexhand)}/{base_filename.replace('.pkl', f'_frame{parser.frame_idx}.pkl')}"
            else:
                dump_path = f"data/retargeting/favor_pass1-mirrored/mano2{str(dexhand)}/{base_filename}"
        else:
            raise ValueError("Unsupported dataset type")

        os.makedirs(os.path.dirname(dump_path), exist_ok=True)
        with open(dump_path, "wb") as f:
            pickle.dump(to_dump, f)
            
        print(f"Results saved to: {dump_path}")

    run(_parser, _parser.data_idx)