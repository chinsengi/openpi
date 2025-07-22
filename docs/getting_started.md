# Getting Started with OpenPI

This guide will help you get started with OpenPI, Physical Intelligence's open source robotics framework.

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA 12 (for GPU acceleration)
- Git

### Install from Source

```bash
# Clone the repository
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Install dependencies
pip install -e .

# For development (optional)
pip install -e ".[dev]"
```

### Verify Installation

```python
import openpi
print("OpenPI installed successfully!")
```

## Quick Start

### 1. Basic Model Inference

Here's how to run inference with a pre-trained model:

```python
import jax
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.shared.download import maybe_download
from openpi.policies.droid_policy import make_droid_example

# Load a pre-trained model
config = get_config("pi0_fast_droid")
checkpoint_dir = maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
policy = create_trained_policy(config, checkpoint_dir)

# Create a sample observation
obs = make_droid_example()

# Run inference
result = policy.infer(obs)
print("Actions shape:", result["actions"].shape)
print("Action values:", result["actions"])
```

### 2. Working with Different Robot Platforms

OpenPI supports multiple robot platforms. Here are examples for each:

#### ALOHA Robot

```python
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.policies.aloha_policy import make_aloha_example
from openpi.shared.download import maybe_download

# Load ALOHA model
config = get_config("pi0_aloha_sim")
checkpoint_dir = maybe_download("gs://openpi-assets/checkpoints/pi0_aloha_sim")
policy = create_trained_policy(config, checkpoint_dir)

# Create ALOHA observation
obs = make_aloha_example()
result = policy.infer(obs)
```

#### DROID Robot

```python
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.policies.droid_policy import make_droid_example
from openpi.shared.download import maybe_download

# Load DROID model
config = get_config("pi0_fast_droid")
checkpoint_dir = maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
policy = create_trained_policy(config, checkpoint_dir)

# Create DROID observation
obs = make_droid_example()
result = policy.infer(obs)
```

#### LIBERO Simulation

```python
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.policies.libero_policy import make_libero_example
from openpi.shared.download import maybe_download

# Load LIBERO model
config = get_config("pi0_libero")
checkpoint_dir = maybe_download("gs://openpi-assets/checkpoints/pi0_libero")
policy = create_trained_policy(config, checkpoint_dir)

# Create LIBERO observation
obs = make_libero_example()
result = policy.infer(obs)
```

### 3. Remote Inference

OpenPI supports remote inference through WebSocket connections.

#### Server Setup

```python
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.shared.download import maybe_download

# Load policy
config = get_config("pi0_fast_droid")
checkpoint_dir = maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
policy = create_trained_policy(config, checkpoint_dir)

# Start server
server = WebsocketPolicyServer(
    policy=policy,
    host="0.0.0.0",
    port=8000,
    metadata={"model_type": "pi0_fast_droid"}
)

print("Starting server on port 8000...")
server.serve_forever()
```

#### Client Usage

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from openpi.policies.droid_policy import make_droid_example

# Connect to server
client = WebsocketClientPolicy(host="localhost", port=8000)

# Get server metadata
metadata = client.get_server_metadata()
print("Server metadata:", metadata)

# Run inference
obs = make_droid_example()
result = client.infer(obs)
print("Remote inference result:", result["actions"].shape)
```

## Understanding Data Formats

### Observation Format

OpenPI expects observations in a specific format:

```python
observation = {
    "images": {
        "base_0_rgb": np.array(...),        # Base camera image
        "left_wrist_0_rgb": np.array(...),  # Left wrist camera
        "right_wrist_0_rgb": np.array(...), # Right wrist camera
    },
    "state": np.array(...),  # Robot state (joint positions, etc.)
    "prompt": "Pick up the object"  # Optional text prompt
}
```

### Image Requirements

- **Format**: RGB images as numpy arrays
- **Shape**: (height, width, 3)
- **Data type**: uint8 or float32
- **Resolution**: Will be resized to (224, 224) automatically

### State Vector

The state vector typically includes:
- Joint positions
- Joint velocities (if available)
- Gripper positions
- Any other relevant sensor data

## Data Preprocessing

### Transforms

OpenPI uses transforms to preprocess data:

```python
from openpi.transforms import Group, ResizeImages, Normalize

# Create transform group
transforms = Group(
    inputs=[
        ResizeImages((224, 224)),
        Normalize(norm_stats)
    ]
)

# Apply transforms
processed_data = transforms.inputs[0](data)
```

### Custom Transforms

You can create custom transforms:

```python
from openpi.transforms import DataTransformFn

def my_custom_transform(data):
    """Custom data transformation."""
    # Modify data as needed
    if "state" in data:
        # Example: add noise to state
        data["state"] = data["state"] + np.random.normal(0, 0.01, data["state"].shape)
    return data

# Use in policy
from openpi.policies.policy import Policy
policy = Policy(model, transforms=[my_custom_transform])
```

## Model Configuration

### Available Models

1. **Pi0**: Full-featured model with best performance
2. **Pi0Fast**: Optimized for faster inference

### Model Variants

```python
from openpi.models.pi0 import Pi0Config

# Standard configuration
config = Pi0Config()

# Custom configuration
config = Pi0Config(
    action_dim=32,           # Number of action dimensions
    action_horizon=50,       # Number of future actions to predict
    max_token_len=48,        # Maximum sequence length
    dtype="bfloat16"         # Model precision
)
```

## Training (Advanced)

### Data Preparation

```python
from openpi.training.data_loader import create_data_loader
from openpi.training.config import get_config

# Load configuration
config = get_config("pi0_aloha_sim")

# Create data loader
loader = create_data_loader(
    config,
    num_batches=1000,
    skip_norm_stats=False
)

# Iterate through data
for batch_idx, (obs, actions) in enumerate(loader):
    print(f"Batch {batch_idx}: obs shape {obs.images['base_0_rgb'].shape}")
    if batch_idx >= 5:  # Just show first few batches
        break
```

### Computing Loss

```python
import jax
from openpi.models.pi0 import Pi0Config

# Create model
config = Pi0Config()
model = config.create(jax.random.key(0))

# Generate fake data
obs = config.fake_obs()
actions = config.fake_act()

# Compute loss
loss = model.compute_loss(jax.random.key(1), obs, actions)
print(f"Training loss: {loss}")
```

## Best Practices

### 1. Memory Management

```python
# Delete models when not needed
del model
del policy

# Use smaller batch sizes for large models
config = config.replace(batch_size=1)
```

### 2. Error Handling

```python
try:
    result = policy.infer(obs)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Model error: {e}")
```

### 3. Logging

```python
import logging

# Enable debug logging
logging.getLogger("openpi").setLevel(logging.DEBUG)

# Custom logging
logger = logging.getLogger(__name__)
logger.info("Starting inference...")
```

### 4. Performance Optimization

```python
# Use JIT compilation (automatic)
# First call is slower due to compilation
result1 = policy.infer(obs)  # Slow (compilation)
result2 = policy.infer(obs)  # Fast (compiled)

# Use appropriate data types
import jax.numpy as jnp
obs_float32 = jax.tree.map(lambda x: x.astype(jnp.float32), obs)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Solution: Use smaller batch size or model
   config = config.replace(batch_size=1)
   ```

2. **Shape Mismatch Errors**
   ```python
   # Check input shapes
   print("Image shape:", obs["images"]["base_0_rgb"].shape)
   print("State shape:", obs["state"].shape)
   ```

3. **Import Errors**
   ```bash
   # Reinstall package
   pip install -e .
   ```

4. **Slow Inference**
   ```python
   # Warm up the model
   dummy_obs = make_droid_example()
   _ = policy.infer(dummy_obs)  # Compilation step
   
   # Now inference will be fast
   result = policy.infer(real_obs)
   ```

### Getting Help

- Check the [API Reference](api_reference.md) for detailed documentation
- Look at the `examples/` directory for working code
- Enable debug logging to see detailed error messages
- Check input data shapes and types

## Next Steps

1. **Explore Examples**: Check the `examples/` directory for more use cases
2. **Custom Models**: Learn how to create custom model configurations
3. **Training**: Set up your own training pipeline
4. **Integration**: Integrate OpenPI with your robot control system

## Resources

- [API Reference](api_reference.md) - Complete API documentation
- [Examples](../examples/) - Working code examples
- [GitHub Repository](https://github.com/Physical-Intelligence/openpi) - Source code and issues