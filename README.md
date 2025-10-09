# UniH2R: A Unified Framework for Learning from Human Demonstrations to Robot Self-Exploration for Generalizable Manipulation

**A unified dynamic reinforcement learning framework that enables a single model to learn generalizable manipulation skills from human demonstrations for multiple, distinct robotic hand types.**

## Overview

UniH2R is a unified dynamic reinforcement learning framework that learns optimal manipulation policies from human demonstrations and exploration while achieving generalization across various robotic manipulator types. The framework addresses two key challenges in embodied intelligence:

1. **Data scarcity and high acquisition costs** in robot manipulation
2. **Cross-hand generalizability** for adaptation to different robotic hand configurations

Key innovations include:
- Universal base model trained on multi-hand hybrid dataset
- Hybrid offline-online dynamic exploration mechanism
- Dynamic reward composition that transitions from imitation to exploration
- Support for 2/3/5-fingered dexterous hands

## Project Structure

```
UniH2R/
├── main/                           # Main training and dataset processing
│   ├── cfg/                        # Configuration files
│   ├── dataset/                    # Dataset processing (mano2dexhand_vlm.py, etc.)
│   └── rl/                         # Reinforcement learning (train.py)
├── lib/                            # Core libraries
│   ├── learn/                      # Learning components
│   ├── nn/                         # Neural network modules
│   ├── rl/                         # RL framework
│   └── utils/                      # Utility functions
├── UniH2R_envs/                    # Simulation environments
│   ├── assets/                     # Robot and object assets
│   └── lib/                        # Environment core
└── DexManipNet/                    # Dataset visualization and processing
```

## Environment Setup

### Dependencies Installation

Create and activate a new conda environment:

```bash
conda create -y -n unih2r python=3.8
conda activate unih2r
```

Install PyTorch with CUDA support:

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install remaining dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

### Isaac Gym Setup

Download IsaacGym Preview 4 from the [official website](https://developer.nvidia.com/isaac-gym) and follow the installation instructions in the documentation. 

Test the installation by running an example script:

```bash
cd isaacgym/python/examples
python joint_monkey.py
```

### VLM Configuration

For multi-agent data generation, configure your VLM API credentials in your environment or configuration file:

```python
API_SECRET_KEY = ""
BASE_URL = ""
```

## Usage

### Data Preparation

Generate expert data by mapping human demonstrations to robotic hands:

```bash
python main/dataset/mano2dexhand_vlm.py --data_idx 667dd@1 --side right --dexhand franka_panda --iter 3000
```

### Stage 1: Universal Base Model Training

Train the multi-hand compatible base model:

```bash
python main/rl/train.py task=MultiDexHandImitator side=RH headless=true num_envs=4096 test=false dataIndices=[667dd@1] learning_rate=2e-4 actionsMovingAverage=0.4 randomStateInit=true
```

Expected output:
```
frames: 22675456
saving next best rewards: [701.53]
=> saving checkpoint 'runs/MultiDexHandImitator__09-11-18-23-12/nn/MultiDexHandImitator.pth'
```

### Stage 2: Dynamic Policy Exploration

Train hand-specific policy heads with dynamic exploration:

```bash
python main/rl/train.py task=UniH2RHand side=RH headless=true num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true rh_base_model_checkpoint=runs/MultiDexHandImitator__09-11-18-23-12/nn/MultiDexHandImitator.pth lh_base_model_checkpoint=runs/MultiDexHandImitator__09-11-18-23-12/nn/MultiDexHandImitator.pth dataIndices=[667dd@1] early_stop_epochs=100 actionsMovingAverage=0.4 experiment=cross_667dd@1_franka_panda dexhand=franka_panda
```

Expected output:
```
fps step: 232522 fps step and policy inference: 160666 fps total: 32914 epoch: 60/9999999999999 frames: 7733248
saving next best rewards: [7225.62]
=> saving checkpoint 'runs/cross_667dd@1_franka_panda__09-11-19-17-56/nn/cross_667dd@1_franka_panda.pth'
```

### Testing

Evaluate the trained model:

```bash
python main/rl/train.py task=OurTestUniH2RHand dexhand=franka_panda side=RH headless=false num_envs=4 learning_rate=2e-4 test=true randomStateInit=false dataIndices=[667dd@1] rh_base_model_checkpoint=runs/MultiDexHandImitator__09-11-18-23-12/nn/MultiDexHandImitator.pth lh_base_model_checkpoint=runs/MultiDexHandImitator__09-11-18-23-12/nn/MultiDexHandImitator.pth actionsMovingAverage=0.4 checkpoint=runs/cross_667dd@1_franka_panda__09-11-19-17-56/nn/cross_667dd@1_franka_panda.pth save_rollouts=true num_rollouts_to_save=32 num_rollouts_to_run=4096 save_successful_rollouts_only=true
```
