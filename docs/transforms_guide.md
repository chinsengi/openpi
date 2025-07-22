# Data Transforms Guide

This guide covers the OpenPI data transformation system, which is used to preprocess and postprocess data for model inference and training.

## Overview

OpenPI uses a flexible transform system to handle data preprocessing and postprocessing. Transforms are applied in sequence and can modify observations, actions, or any other data structures.

## Core Concepts

### DataTransformFn Protocol

All transforms implement the `DataTransformFn` protocol:

```python
from openpi.transforms import DataTransformFn

class MyTransform:
    def __call__(self, data: Dict) -> Dict:
        """Apply transformation to the data."""
        # Transform the data
        return transformed_data
```

### Transform Groups

Transforms are organized into groups for input and output processing:

```python
from openpi.transforms import Group

transforms = Group(
    inputs=[input_transform1, input_transform2],
    outputs=[output_transform1, output_transform2]
)
```

### Composition

Multiple transforms can be composed together:

```python
from openpi.transforms import compose

# Compose transforms into a single function
combined_transform = compose([transform1, transform2, transform3])

# Apply to data
result = combined_transform(data)
```

## Built-in Transforms

### Image Transforms

#### ResizeImages

Resize images to a target size.

```python
from openpi.transforms import ResizeImages

resize = ResizeImages(target_size=(224, 224))

# Apply to data
data = {
    "images": {
        "base_0_rgb": np.array([...]),  # Shape: (480, 640, 3)
    }
}

transformed = resize(data)
# Now images["base_0_rgb"] has shape (224, 224, 3)
```

**Parameters:**
- `target_size` (Tuple[int, int]): Target (height, width)
- `interpolation` (str, optional): Interpolation method

#### CenterCropImages

Center crop images to a target size.

```python
from openpi.transforms import CenterCropImages

crop = CenterCropImages(target_size=(224, 224))

# Crops from the center of each image
transformed = crop(data)
```

#### NormalizeImages

Normalize image pixel values.

```python
from openpi.transforms import NormalizeImages

# Normalize to [0, 1] range
normalize = NormalizeImages(mean=0.0, std=255.0)

# Normalize using ImageNet statistics
normalize_imagenet = NormalizeImages(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### State Transforms

#### Normalize / Unnormalize

Normalize numerical data using precomputed statistics.

```python
from openpi.transforms import Normalize, Unnormalize

# Load normalization statistics
norm_stats = {
    "state": {
        "mean": np.array([...]),
        "std": np.array([...])
    },
    "actions": {
        "mean": np.array([...]),
        "std": np.array([...])
    }
}

# Create transforms
normalize = Normalize(norm_stats, use_quantiles=False)
unnormalize = Unnormalize(norm_stats, use_quantiles=False)

# Apply normalization
normalized_data = normalize(data)

# Apply unnormalization (typically in output transforms)
original_data = unnormalize(normalized_data)
```

**Quantile Normalization:**

```python
# Use quantile normalization for better handling of outliers
normalize_quantile = Normalize(norm_stats, use_quantiles=True)
```

#### ClipActions

Clip action values to specified ranges.

```python
from openpi.transforms import ClipActions

clip = ClipActions(
    min_values=np.array([-1.0, -1.0, -1.0]),
    max_values=np.array([1.0, 1.0, 1.0])
)

# Clips actions to [-1, 1] range
clipped_data = clip(data)
```

### Text Transforms

#### InjectDefaultPrompt

Add a default prompt if none is provided.

```python
from openpi.transforms import InjectDefaultPrompt

inject_prompt = InjectDefaultPrompt("Pick up the red object")

# If data doesn't have a "prompt" key, adds the default
data_with_prompt = inject_prompt(data)
```

#### TokenizePrompt

Tokenize text prompts for model input.

```python
from openpi.transforms import TokenizePrompt
from openpi.models.tokenizer import create_tokenizer

tokenizer = create_tokenizer()
tokenize = TokenizePrompt(tokenizer, max_length=48)

# Converts text prompt to tokens
tokenized_data = tokenize(data)
```

### Utility Transforms

#### SelectKeys

Select only specific keys from the data.

```python
from openpi.transforms import SelectKeys

select = SelectKeys(["images", "state"])

# Only keeps specified keys
filtered_data = select(data)
```

#### RenameKeys

Rename keys in the data dictionary.

```python
from openpi.transforms import RenameKeys

rename = RenameKeys({
    "old_key": "new_key",
    "robot_state": "state"
})

# Renames keys according to mapping
renamed_data = rename(data)
```

#### FlattenDict

Flatten nested dictionaries.

```python
from openpi.transforms import FlattenDict

flatten = FlattenDict(sep="/")

# Converts {"a": {"b": 1}} to {"a/b": 1}
flattened = flatten(data)
```

## Custom Transforms

### Simple Function Transform

```python
def add_noise_to_state(data):
    """Add Gaussian noise to robot state."""
    if "state" in data:
        noise = np.random.normal(0, 0.01, data["state"].shape)
        data = data.copy()  # Don't modify original
        data["state"] = data["state"] + noise
    return data

# Use in policy
from openpi.policies.policy import Policy
policy = Policy(model, transforms=[add_noise_to_state])
```

### Class-based Transform

```python
class ScaleActions:
    """Scale actions by a constant factor."""
    
    def __init__(self, scale_factor: float):
        self.scale_factor = scale_factor
    
    def __call__(self, data):
        if "actions" in data:
            data = data.copy()
            data["actions"] = data["actions"] * self.scale_factor
        return data

# Usage
scale_actions = ScaleActions(scale_factor=2.0)
scaled_data = scale_actions(data)
```

### Stateful Transform

```python
class MovingAverageState:
    """Apply moving average to robot state."""
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = []
    
    def __call__(self, data):
        if "state" in data:
            self.history.append(data["state"])
            if len(self.history) > self.window_size:
                self.history.pop(0)
            
            # Compute moving average
            avg_state = np.mean(self.history, axis=0)
            
            data = data.copy()
            data["state"] = avg_state
        
        return data

# Usage
smooth_state = MovingAverageState(window_size=3)
```

## Transform Patterns

### Robot-Specific Preprocessing

```python
def create_aloha_transforms():
    """Create transforms for ALOHA robot."""
    return Group(
        inputs=[
            ResizeImages((224, 224)),
            NormalizeImages(mean=0.0, std=255.0),
            InjectDefaultPrompt("Perform bimanual manipulation"),
            Normalize(aloha_norm_stats)
        ],
        outputs=[
            Unnormalize(aloha_norm_stats),
            ClipActions(min_vals=aloha_min, max_vals=aloha_max)
        ]
    )
```

### Data Augmentation

```python
class RandomImageAugmentation:
    """Apply random augmentations to images."""
    
    def __init__(self, brightness_range=0.1, contrast_range=0.1):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def __call__(self, data):
        if "images" in data:
            data = data.copy()
            data["images"] = data["images"].copy()
            
            for key, image in data["images"].items():
                # Random brightness
                brightness = 1.0 + np.random.uniform(
                    -self.brightness_range, self.brightness_range
                )
                image = np.clip(image * brightness, 0, 255)
                
                # Random contrast
                contrast = 1.0 + np.random.uniform(
                    -self.contrast_range, self.contrast_range
                )
                image = np.clip((image - 128) * contrast + 128, 0, 255)
                
                data["images"][key] = image.astype(np.uint8)
        
        return data

# Use during training
augment = RandomImageAugmentation()
```

### Multi-Modal Processing

```python
def create_multimodal_transforms():
    """Process both visual and language inputs."""
    return Group(
        inputs=[
            # Image processing
            ResizeImages((224, 224)),
            NormalizeImages(mean=[0.485, 0.456, 0.406], 
                          std=[0.229, 0.224, 0.225]),
            
            # Text processing
            InjectDefaultPrompt("Follow the instruction"),
            TokenizePrompt(tokenizer, max_length=48),
            
            # State processing
            Normalize(norm_stats),
        ]
    )
```

## Advanced Usage

### Conditional Transforms

```python
class ConditionalTransform:
    """Apply transform only under certain conditions."""
    
    def __init__(self, condition_fn, transform_fn):
        self.condition_fn = condition_fn
        self.transform_fn = transform_fn
    
    def __call__(self, data):
        if self.condition_fn(data):
            return self.transform_fn(data)
        return data

# Example: Only apply noise during training
def is_training(data):
    return data.get("is_training", False)

conditional_noise = ConditionalTransform(
    condition_fn=is_training,
    transform_fn=add_noise_to_state
)
```

### Transform Pipelines

```python
class TransformPipeline:
    """Execute transforms in sequence with error handling."""
    
    def __init__(self, transforms, skip_on_error=False):
        self.transforms = transforms
        self.skip_on_error = skip_on_error
    
    def __call__(self, data):
        for i, transform in enumerate(self.transforms):
            try:
                data = transform(data)
            except Exception as e:
                if self.skip_on_error:
                    print(f"Skipping transform {i}: {e}")
                    continue
                else:
                    raise
        return data

# Usage
pipeline = TransformPipeline([
    ResizeImages((224, 224)),
    NormalizeImages(),
    Normalize(norm_stats)
], skip_on_error=True)
```

### Dynamic Transforms

```python
class DynamicNormalize:
    """Normalize using dynamically computed statistics."""
    
    def __init__(self, update_rate=0.01):
        self.update_rate = update_rate
        self.running_mean = None
        self.running_std = None
    
    def __call__(self, data):
        if "state" in data:
            state = data["state"]
            
            # Update running statistics
            if self.running_mean is None:
                self.running_mean = np.mean(state, axis=0)
                self.running_std = np.std(state, axis=0)
            else:
                batch_mean = np.mean(state, axis=0)
                batch_std = np.std(state, axis=0)
                
                self.running_mean = (
                    (1 - self.update_rate) * self.running_mean +
                    self.update_rate * batch_mean
                )
                self.running_std = (
                    (1 - self.update_rate) * self.running_std +
                    self.update_rate * batch_std
                )
            
            # Apply normalization
            normalized_state = (state - self.running_mean) / (self.running_std + 1e-8)
            
            data = data.copy()
            data["state"] = normalized_state
        
        return data
```

## Best Practices

### 1. Immutability

Always create copies when modifying data:

```python
def safe_transform(data):
    # Good: Create a copy
    data = data.copy()
    data["state"] = modify_state(data["state"])
    return data

def unsafe_transform(data):
    # Bad: Modifies original data
    data["state"] = modify_state(data["state"])
    return data
```

### 2. Error Handling

Handle missing keys gracefully:

```python
def robust_transform(data):
    if "state" not in data:
        return data  # Skip if key doesn't exist
    
    try:
        data = data.copy()
        data["state"] = process_state(data["state"])
    except Exception as e:
        print(f"Transform failed: {e}")
        # Return original data or handle error appropriately
    
    return data
```

### 3. Type Checking

Validate input types:

```python
def type_safe_transform(data):
    if not isinstance(data, dict):
        raise ValueError("Expected dict input")
    
    if "images" in data:
        for key, image in data["images"].items():
            if not isinstance(image, np.ndarray):
                raise ValueError(f"Expected numpy array for {key}")
    
    return transform_logic(data)
```

### 4. Performance

Optimize for common cases:

```python
def efficient_transform(data):
    # Early return if nothing to do
    if "state" not in data:
        return data
    
    # Avoid unnecessary copies
    state = data["state"]
    if not needs_processing(state):
        return data
    
    # Only copy when necessary
    data = data.copy()
    data["state"] = process_state(state)
    return data
```

## Debugging Transforms

### Logging Transform Applications

```python
import logging

logger = logging.getLogger(__name__)

def debug_transform(name):
    """Decorator to log transform applications."""
    def decorator(transform_fn):
        def wrapper(data):
            logger.debug(f"Applying transform: {name}")
            result = transform_fn(data)
            logger.debug(f"Transform {name} completed")
            return result
        return wrapper
    return decorator

@debug_transform("resize_images")
def resize_images_debug(data):
    return ResizeImages((224, 224))(data)
```

### Visualizing Transform Effects

```python
def visualize_transform_effect(transform, data):
    """Visualize the effect of a transform."""
    import matplotlib.pyplot as plt
    
    original = data.copy()
    transformed = transform(data)
    
    # Compare images
    if "images" in original:
        for key in original["images"]:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(original["images"][key])
            ax1.set_title(f"Original {key}")
            ax2.imshow(transformed["images"][key])
            ax2.set_title(f"Transformed {key}")
            plt.show()
    
    # Compare states
    if "state" in original:
        print("Original state shape:", original["state"].shape)
        print("Transformed state shape:", transformed["state"].shape)
        print("Original state range:", 
              original["state"].min(), "to", original["state"].max())
        print("Transformed state range:", 
              transformed["state"].min(), "to", transformed["state"].max())
```

## Integration with Policies

### Using Transforms in Policies

```python
from openpi.policies.policy import Policy
from openpi.transforms import Group

# Create transform groups
transforms = Group(
    inputs=[
        ResizeImages((224, 224)),
        Normalize(norm_stats)
    ],
    outputs=[
        Unnormalize(norm_stats),
        ClipActions(min_vals, max_vals)
    ]
)

# Create policy with transforms
policy = Policy(
    model=model,
    transforms=transforms.inputs,
    output_transforms=transforms.outputs
)
```

### Runtime Transform Modification

```python
class AdaptivePolicy:
    """Policy that adapts transforms at runtime."""
    
    def __init__(self, base_policy):
        self.base_policy = base_policy
        self.noise_level = 0.0
    
    def set_noise_level(self, level):
        """Adjust noise level for state."""
        self.noise_level = level
    
    def infer(self, obs):
        # Add adaptive noise transform
        if self.noise_level > 0:
            noise_transform = lambda data: add_noise(data, self.noise_level)
            obs = noise_transform(obs)
        
        return self.base_policy.infer(obs)
```

This comprehensive transforms guide covers all aspects of the OpenPI transform system, from basic usage to advanced patterns and best practices.