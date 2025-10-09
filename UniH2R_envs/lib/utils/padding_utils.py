import torch
import numpy as np


def pad_joint_data(data: torch.Tensor, current_dofs: int, k_max: int) -> torch.Tensor:
    """将关节数据补零到全局最大维度
    
    Args:
        data: 需要补零的关节数据张量 [batch_size, current_dofs, ...]
        current_dofs: 当前手的关节自由度数量
        k_max: 全局最大关节自由度数量
        
    Returns:
        补零后的关节数据张量 [batch_size, k_max, ...]
    """
    if current_dofs >= k_max:
                                 
        return data[..., :k_max]
    
                    
    orig_shape = data.shape
    padding_size = k_max - current_dofs
    
               
    if len(orig_shape) == 2:
                                             
        batch_size = orig_shape[0]
        padding = torch.zeros((batch_size, padding_size), device=data.device, dtype=data.dtype)
        padded_data = torch.cat([data, padding], dim=1)
    elif len(orig_shape) == 3:
                                                       
        batch_size = orig_shape[0]
        feat_dim = orig_shape[2]
        padding = torch.zeros((batch_size, padding_size, feat_dim), device=data.device, dtype=data.dtype)
        padded_data = torch.cat([data, padding], dim=1)
    else:
                           
        padded_data = torch.cat([data, torch.zeros_like(data[..., :0]).expand(*data.shape[:-1], padding_size)], dim=-1)
    
    return padded_data


def pad_to_max_dim(data: torch.Tensor, max_dim: int) -> torch.Tensor:
    """将任意维度的张量补零到指定的最大维度
    
    Args:
        data: 需要补零的数据张量 [batch_size, current_dim]
        max_dim: 目标最大维度
        
    Returns:
        补零后的张量 [batch_size, max_dim]
    """
    current_dim = data.shape[-1]
    
    if current_dim >= max_dim:
                                 
        return data[..., :max_dim]
    
                 
    padding_size = max_dim - current_dim
    
               
    padding = torch.zeros((*data.shape[:-1], padding_size), device=data.device, dtype=data.dtype)
    padded_data = torch.cat([data, padding], dim=-1)
    
    return padded_data


def get_hand_embedding(hand_config, device=None) -> torch.Tensor:
    """获取手型嵌入向量
    
    Args:
        hand_config: 手型配置或手型名称字符串
        device: 指定设备，默认为None
        
    Returns:
        手型嵌入向量 [embedding_dim]
    """
                  
    hand_embedding_map = {
        "allegro": [0.0, 1.0, 0.0], 
        "inspire": [1.0, 0.0, 0.0],
        "franka": [0, 0, 1],                
    }
    
    if isinstance(hand_config, dict) and "embedding" in hand_config:
        embedding = hand_config["embedding"]
    elif isinstance(hand_config, str):
                              
        if hand_config in hand_embedding_map:
            embedding = hand_embedding_map[hand_config]
        else:
                                
            embedding = [0.0, 0.0, 0.0]
    else:
        embedding = hand_config
        
                   
    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(embedding, dtype=torch.float32)
    
            
    if device is not None:
        embedding = embedding.to(device)
        
    return embedding 