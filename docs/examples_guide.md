# Examples Guide

This guide provides detailed explanations and usage instructions for all examples in the OpenPI repository.

## Overview

The `examples/` directory contains working code examples that demonstrate how to use OpenPI with different robot platforms and environments. Each example is self-contained and includes setup instructions.

## Directory Structure

```
examples/
├── aloha_real/          # Real ALOHA robot examples
├── aloha_sim/           # ALOHA simulation examples
├── droid/               # DROID robot examples
├── libero/              # LIBERO simulation examples
├── simple_client/       # Simple client example
├── ur5/                 # UR5 robot examples
├── inference.ipynb      # Basic inference notebook
└── policy_records.ipynb # Policy recording notebook
```

## Jupyter Notebooks

### inference.ipynb

Basic model inference examples using pre-trained checkpoints.

**What it demonstrates:**
- Loading pre-trained models from checkpoints
- Running inference with dummy data
- Working with live models for training
- Computing training loss

**Key code snippets:**

```python
# Load a trained policy
config = get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
policy = create_trained_policy(config, checkpoint_dir)

# Run inference
example = droid_policy.make_droid_example()
result = policy.infer(example)
```

**How to run:**
```bash
cd examples/
jupyter notebook inference.ipynb
```

### policy_records.ipynb

Demonstrates how to record and analyze policy behavior.

**What it demonstrates:**
- Recording policy inputs and outputs
- Analyzing recorded data
- Debugging policy behavior

**How to run:**
```bash
cd examples/
jupyter notebook policy_records.ipynb
```

## Robot Platform Examples

### ALOHA Robot Examples

#### aloha_real/

Real ALOHA robot control and data collection.

**Files:**
- `main.py` - Main control loop
- `env.py` - Environment wrapper
- `real_env.py` - Real robot interface
- `robot_utils.py` - Robot utility functions
- `video_display.py` - Video display utilities
- `convert_aloha_data_to_lerobot.py` - Data conversion script

**Setup:**
```bash
# Install ALOHA dependencies
pip install interbotix_xs_modules

# Run the example
cd examples/aloha_real/
python main.py --policy_path gs://openpi-assets/checkpoints/pi0_aloha_real
```

**Key features:**
- Real-time robot control
- Camera integration
- Safety mechanisms
- Data recording

**Example usage:**

```python
from examples.aloha_real.real_env import make_real_env
from openpi.policies.policy_config import create_trained_policy

# Setup robot environment
env = make_real_env(init_node=True)

# Load policy
config = get_config("pi0_aloha_real")
policy = create_trained_policy(config, checkpoint_path)

# Control loop
obs = env.get_observation()
action = policy.infer(obs)
env.apply_action(action)
```

#### aloha_sim/

ALOHA simulation using gym-aloha.

**Files:**
- `main.py` - Main simulation loop
- `env.py` - Simulation environment wrapper
- `saver.py` - Data saving utilities

**Setup:**
```bash
# Install simulation dependencies
pip install gym-aloha

# Run simulation
cd examples/aloha_sim/
python main.py --task AlohaInsertion-v0 --policy_path gs://openpi-assets/checkpoints/pi0_aloha_sim
```

**Key features:**
- Physics simulation
- Multiple task environments
- Automatic data collection
- Episode management

### DROID Robot Examples

#### droid/

DROID robot control and data collection.

**Files:**
- `main.py` - Main control script

**Setup:**
```bash
# Install DROID dependencies (robot-specific)
# See DROID documentation for hardware setup

# Run the example
cd examples/droid/
python main.py --policy_path gs://openpi-assets/checkpoints/pi0_fast_droid
```

**Key features:**
- Real-time control
- Image processing
- State estimation
- Safety monitoring

**Example observation format:**

```python
obs = {
    "images": {
        "base_0_rgb": np.array(...),        # Base camera
        "left_wrist_0_rgb": np.array(...),  # Left wrist camera  
        "right_wrist_0_rgb": np.array(...), # Right wrist camera
    },
    "state": np.array([...]),  # Joint positions and velocities
    "prompt": "Pick up the object"
}
```

### LIBERO Simulation Examples

#### libero/

LIBERO simulation environment integration.

**Files:**
- `main.py` - Main evaluation script
- `convert_libero_data_to_lerobot.py` - Data conversion utilities

**Setup:**
```bash
# Install LIBERO
pip install libero

# Run evaluation
cd examples/libero/
python main.py --policy_path gs://openpi-assets/checkpoints/pi0_libero
```

**Key features:**
- Multiple manipulation tasks
- Standardized evaluation
- Automatic metrics collection
- Task randomization

**Available tasks:**
- Pick and place
- Stacking
- Assembly tasks
- Tool use

### UR5 Robot Examples

#### ur5/

Universal Robots UR5 integration examples.

**Setup:**
```bash
# Install UR5 dependencies
pip install ur_rtde

# Configure robot connection
# Run examples (specific files depend on setup)
```

## Client Examples

### simple_client/

Demonstrates basic client-server communication.

**Files:**
- `main.py` - Simple client implementation

**What it demonstrates:**
- WebSocket client usage
- Performance monitoring
- Data collection
- Error handling

**Setup:**

1. Start a policy server:
```bash
python -m openpi.serving.websocket_policy_server \
    --policy_path gs://openpi-assets/checkpoints/pi0_fast_droid \
    --port 8000
```

2. Run the client:
```bash
cd examples/simple_client/
python main.py --host localhost --port 8000 --robot droid
```

**Key features:**
- Latency measurement
- Throughput testing
- Data logging
- Statistics collection

**Example client code:**

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# Connect to server
client = WebsocketClientPolicy(host="localhost", port=8000)

# Performance monitoring
stats = PerformanceStats()

for i in range(1000):
    # Generate observation
    obs = generate_random_observation()
    
    # Time the inference
    start_time = time.time()
    result = client.infer(obs)
    inference_time = time.time() - start_time
    
    # Record statistics
    stats.record("inference_ms", inference_time * 1000)
```

## Data Conversion Examples

### convert_aloha_data_to_lerobot.py

Convert ALOHA dataset format to LeRobot format.

**Usage:**
```bash
cd examples/aloha_real/
python convert_aloha_data_to_lerobot.py \
    --raw_dir /path/to/aloha/data \
    --out_dir /path/to/lerobot/data \
    --repo_id your_username/aloha_dataset
```

**Features:**
- Automatic format conversion
- Data validation
- Metadata preservation
- HuggingFace Hub integration

### convert_libero_data_to_lerobot.py

Convert LIBERO dataset to LeRobot format.

**Usage:**
```bash
cd examples/libero/
python convert_libero_data_to_lerobot.py \
    --data_dir /path/to/libero/data \
    --push_to_hub
```

## Running Examples

### Prerequisites

1. **Install OpenPI:**
```bash
pip install -e .
```

2. **Install example-specific dependencies:**
```bash
# For ALOHA
pip install interbotix_xs_modules gym-aloha

# For LIBERO  
pip install libero

# For visualization
pip install matplotlib opencv-python
```

3. **Download model checkpoints:**
```bash
# Checkpoints are downloaded automatically when needed
# Or manually download:
python -c "from openpi.shared.download import maybe_download; maybe_download('gs://openpi-assets/checkpoints/pi0_fast_droid')"
```

### Common Usage Patterns

#### 1. Basic Inference

```python
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.shared.download import maybe_download

# Load model
config = get_config("pi0_fast_droid")
checkpoint_dir = maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
policy = create_trained_policy(config, checkpoint_dir)

# Run inference
result = policy.infer(observation)
```

#### 2. Environment Integration

```python
# Generic environment wrapper
class RobotEnvironment:
    def get_observation(self) -> dict:
        """Get current robot observation."""
        pass
    
    def apply_action(self, action: dict) -> None:
        """Apply action to robot."""
        pass
    
    def is_episode_complete(self) -> bool:
        """Check if episode is finished."""
        pass

# Control loop
env = RobotEnvironment()
while not env.is_episode_complete():
    obs = env.get_observation()
    action = policy.infer(obs)
    env.apply_action(action)
```

#### 3. Data Collection

```python
from examples.aloha_sim.saver import DataSaver

# Setup data collection
saver = DataSaver(out_dir="./collected_data")

# Episode loop
saver.on_episode_start()
for step in range(episode_length):
    obs = env.get_observation()
    action = policy.infer(obs)
    env.apply_action(action)
    saver.on_step(obs, action)
saver.on_episode_end()
```

#### 4. Performance Monitoring

```python
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.stats = defaultdict(list)
    
    def record(self, key: str, value: float):
        self.stats[key].append(value)
    
    def get_summary(self):
        summary = {}
        for key, values in self.stats.items():
            summary[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values)
            }
        return summary

# Usage
monitor = PerformanceMonitor()

for i in range(100):
    start_time = time.time()
    result = policy.infer(obs)
    inference_time = time.time() - start_time
    monitor.record("inference_ms", inference_time * 1000)

print(monitor.get_summary())
```

## Troubleshooting Examples

### Common Issues

1. **Import Errors:**
```bash
# Make sure OpenPI is installed
pip install -e .

# Check Python path
export PYTHONPATH=$PYTHONPATH:/path/to/openpi
```

2. **CUDA Out of Memory:**
```python
# Use smaller batch size
config = config.replace(batch_size=1)

# Or use CPU
import jax
jax.config.update('jax_platform_name', 'cpu')
```

3. **Robot Connection Issues:**
```python
# Check robot IP and ports
# Verify robot is powered on and accessible
# Check firewall settings
```

4. **Data Format Issues:**
```python
# Validate observation format
def validate_observation(obs):
    required_keys = ["images", "state"]
    for key in required_keys:
        if key not in obs:
            raise ValueError(f"Missing key: {key}")
    
    for img_key, img in obs["images"].items():
        if img.shape[-1] != 3:
            raise ValueError(f"Image {img_key} should have 3 channels")
```

### Debug Mode

Enable debug logging for examples:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("openpi").setLevel(logging.DEBUG)
```

### Performance Profiling

Profile example performance:

```python
import cProfile
import pstats

# Profile inference
profiler = cProfile.Profile()
profiler.enable()

for i in range(100):
    result = policy.infer(obs)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(10)
```

## Custom Examples

### Creating Your Own Example

1. **Create directory structure:**
```bash
mkdir examples/my_robot/
cd examples/my_robot/
```

2. **Create environment wrapper:**
```python
# my_robot_env.py
class MyRobotEnv:
    def __init__(self):
        # Initialize robot connection
        pass
    
    def get_observation(self):
        # Return observation in OpenPI format
        return {
            "images": {...},
            "state": np.array([...]),
            "prompt": "..."
        }
    
    def apply_action(self, action):
        # Send action to robot
        pass
```

3. **Create main script:**
```python
# main.py
import argparse
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config
from my_robot_env import MyRobotEnv

def main(args):
    # Load policy
    config = get_config(args.config_name)
    policy = create_trained_policy(config, args.policy_path)
    
    # Setup environment
    env = MyRobotEnv()
    
    # Control loop
    for episode in range(args.num_episodes):
        obs = env.get_observation()
        action = policy.infer(obs)
        env.apply_action(action)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--config_name", default="pi0_fast")
    parser.add_argument("--num_episodes", type=int, default=10)
    args = parser.parse_args()
    main(args)
```

### Integration Checklist

- [ ] Observation format matches OpenPI requirements
- [ ] Images are RGB with shape (H, W, 3)
- [ ] State vector is properly normalized
- [ ] Actions are applied correctly to robot
- [ ] Safety mechanisms are in place
- [ ] Error handling is implemented
- [ ] Performance is monitored
- [ ] Data collection is optional

## Resources

- [API Reference](api_reference.md) - Complete API documentation
- [Getting Started](getting_started.md) - Basic usage guide
- [Transforms Guide](transforms_guide.md) - Data preprocessing guide
- [GitHub Examples](../examples/) - Source code for all examples