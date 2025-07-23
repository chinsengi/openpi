# OpenPI API Reference

This document provides comprehensive documentation for all public APIs, functions, and components in the OpenPI repository.

## Table of Contents

1. [Models](#models)
2. [Policies](#policies)
3. [Transforms](#transforms)
4. [Training](#training)
5. [Serving](#serving)
6. [Client](#client)
7. [Shared Utilities](#shared-utilities)
8. [Examples](#examples)

## Models

### BaseModel

The base class for all OpenPI models.

```python
from openpi.models.model import BaseModel
```

#### Methods

##### `sample_actions(rng, observation, **kwargs)`

Sample actions from the model given observations.

**Parameters:**
- `rng` (jax.random.PRNGKey): Random number generator key
- `observation` (Observation): Input observations
- `**kwargs`: Additional sampling parameters

**Returns:**
- `Actions`: Sampled actions

##### `compute_loss(rng, observation, actions)`

Compute the training loss for given observations and actions.

**Parameters:**
- `rng` (jax.random.PRNGKey): Random number generator key
- `observation` (Observation): Input observations  
- `actions` (Actions): Target actions

**Returns:**
- `float`: Computed loss value

##### `fake_obs()` / `fake_act()`

Generate fake observations/actions for testing.

**Returns:**
- `Observation` / `Actions`: Fake data structures

### Pi0

The main Pi0 model implementation.

```python
from openpi.models.pi0 import Pi0, Pi0Config
```

#### Pi0Config

Configuration class for Pi0 model.

**Parameters:**
- `dtype` (str): Model data type (default: "bfloat16")
- `paligemma_variant` (str): PaliGemma variant to use (default: "gemma_2b")
- `action_expert_variant` (str): Action expert variant (default: "gemma_300m")
- `action_dim` (int): Action dimension (default: 32)
- `action_horizon` (int): Action horizon (default: 50)
- `max_token_len` (int): Maximum token length (default: 48)

#### Example Usage

```python
import jax
from openpi.models.pi0 import Pi0Config

# Create model configuration
config = Pi0Config(
    action_dim=32,
    action_horizon=50
)

# Create model
rng = jax.random.key(0)
model = config.create(rng)

# Generate fake data for testing
obs = config.fake_obs()
actions = model.sample_actions(rng, obs)
```

### Pi0Fast

Optimized version of Pi0 model for faster inference.

```python
from openpi.models.pi0_fast import Pi0Fast, Pi0FastConfig
```

Similar API to Pi0 but with optimizations for inference speed.

### Model Utilities

#### `restore_params(checkpoint_dir, dtype=None)`

Restore model parameters from a checkpoint directory.

```python
from openpi.models.model import restore_params

params = restore_params("path/to/checkpoint", dtype=jnp.bfloat16)
```

## Policies

### BasePolicy

Abstract base class for all policies.

```python
from openpi_client.base_policy import BasePolicy
```

#### Methods

##### `infer(obs)`

Infer actions from observations.

**Parameters:**
- `obs` (Dict): Observation dictionary

**Returns:**
- `Dict`: Action dictionary

##### `reset()`

Reset the policy to its initial state.

### Policy

Main policy implementation that wraps models with transforms.

```python
from openpi.policies.policy import Policy
```

#### Constructor

```python
Policy(
    model,
    rng=None,
    transforms=(),
    output_transforms=(),
    sample_kwargs=None,
    metadata=None
)
```

**Parameters:**
- `model` (BaseModel): The underlying model
- `rng` (jax.random.PRNGKey, optional): Random number generator
- `transforms` (Sequence[DataTransformFn]): Input transforms
- `output_transforms` (Sequence[DataTransformFn]): Output transforms
- `sample_kwargs` (Dict, optional): Sampling parameters
- `metadata` (Dict, optional): Policy metadata

#### Example Usage

```python
from openpi.policies.policy import Policy
from openpi.models.pi0 import Pi0Config
import jax

# Create model
config = Pi0Config()
model = config.create(jax.random.key(0))

# Create policy
policy = Policy(model)

# Run inference
obs = {"images": {...}, "state": {...}}
result = policy.infer(obs)
```

### PolicyRecorder

Wrapper that records policy behavior to disk.

```python
from openpi.policies.policy import PolicyRecorder

recorder = PolicyRecorder(policy, record_dir="./recordings")
result = recorder.infer(obs)  # Automatically saves to disk
```

### Policy Configuration

#### `create_trained_policy(train_config, checkpoint_dir, **kwargs)`

Create a policy from a trained checkpoint.

```python
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import get_config

config = get_config("pi0_fast_droid")
checkpoint_dir = "path/to/checkpoint"
policy = create_trained_policy(config, checkpoint_dir)
```

**Parameters:**
- `train_config` (TrainConfig): Training configuration
- `checkpoint_dir` (Path): Checkpoint directory
- `repack_transforms` (Group, optional): Additional transforms
- `sample_kwargs` (Dict, optional): Sampling parameters
- `default_prompt` (str, optional): Default prompt
- `norm_stats` (Dict, optional): Normalization statistics

### Domain-Specific Policies

#### AlohaPolicy

Policy for ALOHA robot tasks.

```python
from openpi.policies.aloha_policy import AlohaPolicy, make_aloha_example

# Create example observation
example = make_aloha_example()
```

#### DroidPolicy  

Policy for DROID robot tasks.

```python
from openpi.policies.droid_policy import DroidPolicy, make_droid_example

# Create example observation
example = make_droid_example()
```

#### LiberoPolicy

Policy for LIBERO simulation tasks.

```python
from openpi.policies.libero_policy import LiberoPolicy, make_libero_example

# Create example observation  
example = make_libero_example()
```

## Transforms

Data transformation utilities for preprocessing and postprocessing.

### DataTransformFn

Protocol for data transformation functions.

```python
from openpi.transforms import DataTransformFn

def my_transform(data: Dict) -> Dict:
    # Transform the data
    return transformed_data
```

### Group

Container for organizing input and output transforms.

```python
from openpi.transforms import Group

transforms = Group(
    inputs=[transform1, transform2],
    outputs=[output_transform1]
)
```

### Built-in Transforms

#### Normalize / Unnormalize

```python
from openpi.transforms import Normalize, Unnormalize

# Normalize data using statistics
normalize = Normalize(norm_stats, use_quantiles=False)
unnormalize = Unnormalize(norm_stats, use_quantiles=False)
```

#### InjectDefaultPrompt

```python
from openpi.transforms import InjectDefaultPrompt

inject_prompt = InjectDefaultPrompt("Pick up the object")
```

#### ResizeImages

```python
from openpi.transforms import ResizeImages

resize = ResizeImages(target_size=(224, 224))
```

#### Example Usage

```python
from openpi.transforms import compose, Normalize, ResizeImages

# Compose multiple transforms
transform_fn = compose([
    ResizeImages((224, 224)),
    Normalize(norm_stats)
])

# Apply to data
transformed_data = transform_fn(data)
```

## Training

### Configuration

#### `get_config(config_name)`

Get a predefined training configuration.

```python
from openpi.training.config import get_config

config = get_config("pi0_fast_droid")
```

Available configurations:
- `"pi0_aloha_sim"` - Pi0 for ALOHA simulation
- `"pi0_fast_droid"` - Pi0 Fast for DROID
- `"pi0_libero"` - Pi0 for LIBERO tasks

#### TrainConfig

Main training configuration class.

```python
@dataclasses.dataclass
class TrainConfig:
    model: BaseModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
    # ... other fields
```

### Data Loading

#### `create_data_loader(config, num_batches=None, skip_norm_stats=False)`

Create a data loader for training.

```python
from openpi.training.data_loader import create_data_loader

loader = create_data_loader(config, num_batches=100)
for obs, actions in loader:
    # Training loop
    pass
```

### Checkpoints

#### `save_checkpoint(checkpoint_dir, train_state, config)`

Save training checkpoint.

```python
from openpi.training.checkpoints import save_checkpoint

save_checkpoint("./checkpoints/step_1000", train_state, config)
```

#### `load_norm_stats(assets_dir, asset_id)`

Load normalization statistics.

```python
from openpi.training.checkpoints import load_norm_stats

norm_stats = load_norm_stats("./assets", "droid")
```

## Serving

### WebsocketPolicyServer

Serve policies over WebSocket for remote inference.

```python
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

server = WebsocketPolicyServer(
    policy=policy,
    host="0.0.0.0", 
    port=8000,
    metadata={"model_type": "pi0"}
)

server.serve_forever()
```

#### Constructor Parameters

- `policy` (BasePolicy): Policy to serve
- `host` (str): Server host (default: "0.0.0.0")
- `port` (int, optional): Server port
- `metadata` (Dict, optional): Server metadata

#### Health Check

The server provides a health check endpoint at `/healthz`.

## Client

### WebsocketClientPolicy

Client for connecting to remote policy servers.

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(
    host="localhost",
    port=8000,
    api_key="your-api-key"  # optional
)

# Get server metadata
metadata = client.get_server_metadata()

# Run inference
result = client.infer(observation)
```

#### Constructor Parameters

- `host` (str): Server hostname (default: "0.0.0.0")
- `port` (int, optional): Server port
- `api_key` (str, optional): API key for authentication

### Runtime Components

#### Agent

```python
from openpi_client.runtime.agent import Agent

agent = Agent()
```

#### Environment

```python
from openpi_client.runtime.environment import Environment

env = Environment()
```

#### Runtime

Main runtime orchestrator.

```python
from openpi_client.runtime.runtime import Runtime

runtime = Runtime(agent, environment)
runtime.run()
```

## Shared Utilities

### Download

#### `maybe_download(url, force_download=False, **kwargs)`

Download files or directories with caching.

```python
from openpi.shared.download import maybe_download

# Download model checkpoint
checkpoint_path = maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
```

#### `get_cache_dir()`

Get the cache directory path.

```python
from openpi.shared.download import get_cache_dir

cache_dir = get_cache_dir()  # ~/.cache/openpi by default
```

### Image Tools

#### `resize_images(images, target_size)`

Resize images to target size.

```python
from openpi.shared.image_tools import resize_images

resized = resize_images(images, (224, 224))
```

#### `center_crop_images(images, target_size)`

Center crop images.

```python
from openpi.shared.image_tools import center_crop_images

cropped = center_crop_images(images, (224, 224))
```

### Normalization

#### `compute_norm_stats(data, keys, use_quantiles=False)`

Compute normalization statistics from data.

```python
from openpi.shared.normalize import compute_norm_stats

stats = compute_norm_stats(data, keys=["actions"], use_quantiles=True)
```

### Array Typing

Type annotations for JAX arrays.

```python
from openpi.shared.array_typing import Array, Float, Int

# Type annotated function
def process_data(x: Float[Array, "batch height width channels"]) -> Array:
    return x.mean(axis=0)
```

## Examples

### Basic Inference

```python
import jax
from openpi.models.pi0 import Pi0Config
from openpi.policies.policy import Policy
from openpi.policies.droid_policy import make_droid_example

# Create model and policy
config = Pi0Config()
model = config.create(jax.random.key(0))
policy = Policy(model)

# Run inference
example = make_droid_example()
result = policy.infer(example)
print("Actions shape:", result["actions"].shape)
```

### Loading Trained Model

```python
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.shared.download import maybe_download

# Load configuration and checkpoint
config = get_config("pi0_fast_droid")
checkpoint_dir = maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")

# Create trained policy
policy = create_trained_policy(config, checkpoint_dir)

# Run inference
from openpi.policies.droid_policy import make_droid_example
example = make_droid_example()
result = policy.infer(example)
```

### Remote Inference

```python
# Server side
from openpi.serving.websocket_policy_server import WebsocketPolicyServer

server = WebsocketPolicyServer(policy, port=8000)
server.serve_forever()

# Client side  
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy(host="localhost", port=8000)
result = client.infer(observation)
```

### Custom Transforms

```python
from openpi.transforms import DataTransformFn, Group
import numpy as np

def add_noise(data):
    """Add noise to observations."""
    if "state" in data:
        data["state"] = data["state"] + np.random.normal(0, 0.01, data["state"].shape)
    return data

# Use in policy
transforms = Group(inputs=[add_noise])
policy = Policy(model, transforms=transforms.inputs)
```

### Training Loop

```python
import jax
from openpi.training.config import get_config
from openpi.training.data_loader import create_data_loader

# Setup
config = get_config("pi0_aloha_sim")
loader = create_data_loader(config, num_batches=1000)
model = config.model.create(jax.random.key(0))

# Training loop
for step, (obs, actions) in enumerate(loader):
    loss = model.compute_loss(jax.random.key(step), obs, actions)
    print(f"Step {step}, Loss: {loss}")
```

## Error Handling

### Common Exceptions

- `ValueError`: Invalid configuration or parameters
- `FileNotFoundError`: Missing checkpoint or data files
- `ConnectionRefusedError`: Server connection issues
- `RuntimeError`: Model inference errors

### Best Practices

1. Always use try-catch blocks for network operations
2. Validate input data shapes before inference
3. Check model compatibility with data format
4. Monitor memory usage for large models
5. Use appropriate data types (bfloat16 for efficiency)

## Performance Tips

1. **Batch Processing**: Process multiple observations together when possible
2. **JIT Compilation**: Models are automatically JIT compiled for speed
3. **Memory Management**: Delete large models when not needed
4. **Caching**: Use `maybe_download()` for automatic caching
5. **Data Types**: Use bfloat16 for better memory efficiency

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller model variants
2. **Slow Inference**: Ensure JIT compilation has occurred (first call is slower)
3. **Shape Mismatches**: Check data preprocessing and model input requirements
4. **Connection Issues**: Verify server is running and ports are accessible

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("openpi").setLevel(logging.DEBUG)
```