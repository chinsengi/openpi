# OpenPI Quick Reference

This quick reference provides common code snippets and usage patterns for OpenPI.

## Installation

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
pip install -e .
```

## Basic Model Loading

```python
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.shared.download import maybe_download

# Load pre-trained model
config = get_config("pi0_fast_droid")
checkpoint_dir = maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
policy = create_trained_policy(config, checkpoint_dir)
```

## Available Model Configs

```python
# ALOHA models
config = get_config("pi0_aloha_sim")      # ALOHA simulation
config = get_config("pi0_aloha_real")     # ALOHA real robot

# DROID models  
config = get_config("pi0_fast_droid")     # DROID (optimized)

# LIBERO models
config = get_config("pi0_libero")         # LIBERO simulation
```

## Inference

### Basic Inference

```python
# Create sample observation
from openpi.policies.droid_policy import make_droid_example
obs = make_droid_example()

# Run inference
result = policy.infer(obs)
actions = result["actions"]  # Shape: (action_horizon, action_dim)
```

### Custom Observation

```python
import numpy as np

obs = {
    "images": {
        "base_0_rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "left_wrist_0_rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "right_wrist_0_rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    },
    "state": np.random.randn(14),  # Joint positions
    "prompt": "Pick up the red cube"
}

result = policy.infer(obs)
```

## Remote Inference

### Server

```python
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

server = WebsocketPolicyServer(
    policy=policy,
    host="0.0.0.0",
    port=8000,
    metadata={"model_type": "pi0_fast_droid"}
)
server.serve_forever()
```

### Client

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="localhost", port=8000)
result = client.infer(obs)
```

## Data Transforms

### Basic Transforms

```python
from openpi.transforms import ResizeImages, Normalize, compose

# Individual transforms
resize = ResizeImages((224, 224))
normalize = Normalize(norm_stats)

# Compose transforms
transform = compose([resize, normalize])
transformed_data = transform(data)
```

### Custom Transform

```python
def add_noise(data):
    if "state" in data:
        data = data.copy()
        data["state"] = data["state"] + np.random.normal(0, 0.01, data["state"].shape)
    return data

# Use in policy
from openpi.policies.policy import Policy
policy = Policy(model, transforms=[add_noise])
```

## Model Creation

### From Scratch

```python
import jax
from openpi.models.pi0 import Pi0Config

config = Pi0Config(
    action_dim=32,
    action_horizon=50,
    dtype="bfloat16"
)

model = config.create(jax.random.key(0))
```

### Fake Data Generation

```python
# Generate fake observations and actions
obs = config.fake_obs()
actions = config.fake_act()

# Compute loss
loss = model.compute_loss(jax.random.key(1), obs, actions)
```

## Training

### Data Loading

```python
from openpi.training.data_loader import create_data_loader

loader = create_data_loader(config, num_batches=100)
for obs, actions in loader:
    # Training step
    loss = model.compute_loss(jax.random.key(step), obs, actions)
```

### Checkpointing

```python
from openpi.training.checkpoints import save_checkpoint, load_norm_stats

# Save checkpoint
save_checkpoint("./checkpoints/step_1000", train_state, config)

# Load normalization stats
norm_stats = load_norm_stats("./assets", "droid")
```

## Utilities

### Download Models

```python
from openpi.shared.download import maybe_download, get_cache_dir

# Download checkpoint
path = maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

# Get cache directory
cache_dir = get_cache_dir()
```

### Image Processing

```python
from openpi.shared.image_tools import resize_images, center_crop_images

# Resize images
resized = resize_images(images, (224, 224))

# Center crop
cropped = center_crop_images(images, (224, 224))
```

### Normalization

```python
from openpi.shared.normalize import compute_norm_stats

# Compute statistics
stats = compute_norm_stats(data, keys=["state", "actions"], use_quantiles=True)
```

## Platform-Specific Examples

### ALOHA

```python
from openpi.policies.aloha_policy import make_aloha_example

config = get_config("pi0_aloha_sim")
policy = create_trained_policy(config, checkpoint_dir)
obs = make_aloha_example()
result = policy.infer(obs)
```

### DROID

```python
from openpi.policies.droid_policy import make_droid_example

config = get_config("pi0_fast_droid")
policy = create_trained_policy(config, checkpoint_dir)
obs = make_droid_example()
result = policy.infer(obs)
```

### LIBERO

```python
from openpi.policies.libero_policy import make_libero_example

config = get_config("pi0_libero")
policy = create_trained_policy(config, checkpoint_dir)
obs = make_libero_example()
result = policy.infer(obs)
```

## Error Handling

### Basic Error Handling

```python
try:
    result = policy.infer(obs)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Model error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Input Validation

```python
def validate_observation(obs):
    """Validate observation format."""
    required_keys = ["images", "state"]
    for key in required_keys:
        if key not in obs:
            raise ValueError(f"Missing key: {key}")
    
    for img_key, img in obs["images"].items():
        if not isinstance(img, np.ndarray):
            raise ValueError(f"Image {img_key} must be numpy array")
        if img.shape[-1] != 3:
            raise ValueError(f"Image {img_key} must have 3 channels")
        if img.dtype not in [np.uint8, np.float32]:
            raise ValueError(f"Image {img_key} must be uint8 or float32")
```

## Performance Optimization

### Memory Management

```python
# Delete models when not needed
del model
del policy

# Use smaller batch size
config = config.replace(batch_size=1)

# Force garbage collection
import gc
gc.collect()
```

### JIT Compilation

```python
# Warm up the model (first call is slow)
dummy_obs = make_droid_example()
_ = policy.infer(dummy_obs)  # Compilation happens here

# Subsequent calls are fast
result = policy.infer(real_obs)
```

### GPU Memory

```python
# Check GPU memory
import jax
print("GPU memory:", jax.device_memory_allocated())

# Use CPU if needed
jax.config.update('jax_platform_name', 'cpu')
```

## Debugging

### Enable Logging

```python
import logging

# Debug logging
logging.getLogger("openpi").setLevel(logging.DEBUG)

# Custom logger
logger = logging.getLogger(__name__)
logger.info("Starting inference...")
```

### Profile Performance

```python
import time
import cProfile

# Time inference
start = time.time()
result = policy.infer(obs)
print(f"Inference time: {(time.time() - start) * 1000:.2f}ms")

# Profile with cProfile
profiler = cProfile.Profile()
profiler.enable()
for _ in range(10):
    result = policy.infer(obs)
profiler.disable()
profiler.print_stats(sort='cumulative')
```

### Inspect Data

```python
# Print data shapes
def print_shapes(data, prefix=""):
    for key, value in data.items():
        if isinstance(value, dict):
            print_shapes(value, f"{prefix}{key}/")
        elif hasattr(value, 'shape'):
            print(f"{prefix}{key}: {value.shape}")

print_shapes(obs)
```

## Common Issues

### CUDA Out of Memory

```python
# Solution 1: Reduce batch size
config = config.replace(batch_size=1)

# Solution 2: Use CPU
import jax
jax.config.update('jax_platform_name', 'cpu')

# Solution 3: Clear cache
jax.clear_caches()
```

### Shape Mismatches

```python
# Check shapes
print("Image shapes:")
for key, img in obs["images"].items():
    print(f"  {key}: {img.shape}")
print(f"State shape: {obs['state'].shape}")

# Ensure correct format
obs["images"]["base_0_rgb"] = obs["images"]["base_0_rgb"].astype(np.uint8)
```

### Import Errors

```bash
# Reinstall package
pip install -e .

# Check Python path
export PYTHONPATH=$PYTHONPATH:/path/to/openpi

# Verify installation
python -c "import openpi; print('OK')"
```

## Environment Variables

```bash
# Cache directory
export OPENPI_DATA_HOME=/custom/cache/path

# JAX configuration
export JAX_PLATFORM_NAME=cpu  # Force CPU
export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
```

## Useful Constants

```python
from openpi.models.model import IMAGE_KEYS, IMAGE_RESOLUTION

# Standard image keys
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")

# Standard image resolution
IMAGE_RESOLUTION = (224, 224)

# Model types
from openpi.models.model import ModelType
ModelType.PI0       # "pi0"
ModelType.PI0_FAST  # "pi0_fast"
```

## Command Line Tools

```bash
# Start WebSocket server
python -m openpi.serving.websocket_policy_server \
    --policy_path gs://openpi-assets/checkpoints/pi0_fast_droid \
    --port 8000

# Run training
python scripts/train.py --config pi0_aloha_sim

# Convert data
python examples/aloha_real/convert_aloha_data_to_lerobot.py \
    --raw_dir /path/to/data --out_dir /path/to/output
```

This quick reference covers the most common OpenPI usage patterns. For detailed documentation, see the full guides.