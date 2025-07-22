# OpenPI Documentation

Welcome to the comprehensive documentation for OpenPI, Physical Intelligence's open source robotics framework.

## Table of Contents

### Getting Started
- [**Getting Started Guide**](getting_started.md) - Installation, basic usage, and first steps
- [**API Reference**](api_reference.md) - Complete reference for all public APIs and functions
- [**Examples Guide**](examples_guide.md) - Detailed walkthrough of all examples
- [**Data Transforms Guide**](transforms_guide.md) - Complete guide to data preprocessing

### Quick Links
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Available Models](#available-models)
- [Robot Platforms](#robot-platforms)
- [Examples](#examples)

## Overview

OpenPI is a comprehensive framework for robot learning that provides:

- **Pre-trained Models**: State-of-the-art vision-language-action models (Pi0, Pi0Fast)
- **Multi-Platform Support**: ALOHA, DROID, LIBERO, UR5, and more
- **Flexible Data Pipeline**: Powerful transform system for data preprocessing
- **Remote Inference**: Client-server architecture for distributed deployment
- **Training Infrastructure**: Complete training pipeline with data loading and checkpointing

## Installation

### Quick Install

```bash
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi
pip install -e .
```

### System Requirements

- Python 3.11+
- CUDA 12 (for GPU acceleration)
- 16GB+ RAM recommended
- Linux, macOS, or Windows

For detailed installation instructions, see the [Getting Started Guide](getting_started.md).

## Basic Usage

### 1. Load a Pre-trained Model

```python
from openpi.training.config import get_config
from openpi.policies.policy_config import create_trained_policy
from openpi.shared.download import maybe_download

# Load DROID model
config = get_config("pi0_fast_droid")
checkpoint_dir = maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
policy = create_trained_policy(config, checkpoint_dir)
```

### 2. Run Inference

```python
from openpi.policies.droid_policy import make_droid_example

# Create sample observation
obs = make_droid_example()

# Run inference
result = policy.infer(obs)
print("Actions:", result["actions"].shape)
```

### 3. Remote Inference

```python
# Server
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
server = WebsocketPolicyServer(policy, port=8000)
server.serve_forever()

# Client
from openpi_client.websocket_client_policy import WebsocketClientPolicy
client = WebsocketClientPolicy(host="localhost", port=8000)
result = client.infer(obs)
```

## Available Models

### Pi0
- **Full-featured model** with best performance
- Vision-language-action architecture
- Supports multi-modal inputs (images + text + state)
- Action horizon: 50 steps

### Pi0Fast
- **Optimized for inference speed**
- Reduced model size for faster deployment
- Same API as Pi0
- Ideal for real-time applications

### Model Configurations

| Config Name | Robot Platform | Model Type | Use Case |
|-------------|----------------|------------|----------|
| `pi0_aloha_sim` | ALOHA Simulation | Pi0 | Bimanual manipulation |
| `pi0_aloha_real` | ALOHA Real | Pi0 | Real robot control |
| `pi0_fast_droid` | DROID | Pi0Fast | Mobile manipulation |
| `pi0_libero` | LIBERO | Pi0 | Simulation tasks |

## Robot Platforms

### ALOHA
- **Bimanual manipulation** robot
- Dual-arm setup with wrist cameras
- Real robot and simulation support
- Tasks: insertion, stacking, bimanual coordination

### DROID
- **Mobile manipulation** platform
- Base camera + dual wrist cameras
- Real-time control capabilities
- Tasks: pick and place, navigation

### LIBERO
- **Simulation environment** for manipulation
- Multiple standardized tasks
- Evaluation benchmarks
- Tasks: tool use, assembly, stacking

### UR5
- **Industrial robot arm**
- 6-DOF manipulation
- Real robot integration
- Tasks: precise manipulation

## Examples

### Jupyter Notebooks
- [`inference.ipynb`](../examples/inference.ipynb) - Basic model inference
- [`policy_records.ipynb`](../examples/policy_records.ipynb) - Policy recording and analysis

### Robot Examples
- [`aloha_real/`](../examples/aloha_real/) - Real ALOHA robot control
- [`aloha_sim/`](../examples/aloha_sim/) - ALOHA simulation
- [`droid/`](../examples/droid/) - DROID robot control
- [`libero/`](../examples/libero/) - LIBERO simulation
- [`simple_client/`](../examples/simple_client/) - Client-server communication

### Data Conversion
- Convert ALOHA data to LeRobot format
- Convert LIBERO data to LeRobot format
- Custom dataset integration

## Key Features

### 🤖 Multi-Modal Models
- **Vision**: RGB images from multiple cameras
- **Language**: Natural language task descriptions
- **Proprioception**: Robot joint states and sensor data
- **Actions**: Multi-step action sequences

### 🔄 Data Transforms
- **Image Processing**: Resize, crop, normalize
- **State Normalization**: Z-score and quantile normalization
- **Action Processing**: Clipping, scaling, temporal processing
- **Custom Transforms**: Easy to extend with custom preprocessing

### 🌐 Distributed Inference
- **WebSocket Server**: Serve models remotely
- **Client Library**: Connect from any Python application
- **Load Balancing**: Multiple server instances
- **Authentication**: API key support

### 📊 Training Infrastructure
- **Data Loading**: Efficient batch processing
- **Checkpointing**: Save and restore training state
- **Normalization**: Automatic statistics computation
- **Monitoring**: Weights & Biases integration

## Documentation Structure

### Core Documentation
1. **[Getting Started](getting_started.md)** - Your first steps with OpenPI
   - Installation and setup
   - Basic inference examples
   - Understanding data formats
   - Common troubleshooting

2. **[API Reference](api_reference.md)** - Complete API documentation
   - Models and policies
   - Training components
   - Serving and client APIs
   - Utility functions

3. **[Transforms Guide](transforms_guide.md)** - Data preprocessing system
   - Built-in transforms
   - Custom transform creation
   - Transform composition
   - Best practices

4. **[Examples Guide](examples_guide.md)** - Detailed example walkthrough
   - Platform-specific examples
   - Integration patterns
   - Performance monitoring
   - Custom example creation

### Specialized Guides

#### Data Formats
OpenPI uses standardized data formats:

```python
observation = {
    "images": {
        "base_0_rgb": np.array(...),        # (H, W, 3) RGB
        "left_wrist_0_rgb": np.array(...),  # (H, W, 3) RGB
        "right_wrist_0_rgb": np.array(...), # (H, W, 3) RGB
    },
    "state": np.array(...),    # Robot joint positions/velocities
    "prompt": "Task description"  # Optional text prompt
}
```

#### Transform Pipeline
Data flows through transform pipelines:

```
Raw Data → Input Transforms → Model → Output Transforms → Actions
```

#### Model Architecture
Pi0 models use a vision-language-action architecture:

```
Images → Vision Encoder → 
                        → Fusion → Action Decoder → Actions
Text → Language Encoder →
State → State Encoder →
```

## Advanced Topics

### Custom Model Training
- Setting up training data
- Configuring model architecture
- Distributed training
- Hyperparameter tuning

### Production Deployment
- Model optimization
- Serving infrastructure
- Monitoring and logging
- Safety considerations

### Integration Patterns
- Robot control loops
- Data collection pipelines
- Multi-robot coordination
- Human-robot interaction

## Community and Support

### Getting Help
- **Documentation**: Start with the guides above
- **Examples**: Check the `examples/` directory
- **Issues**: Report bugs on GitHub
- **Discussions**: Community discussions on GitHub

### Contributing
- **Code**: Submit pull requests
- **Documentation**: Improve guides and examples
- **Examples**: Add new robot platform examples
- **Testing**: Help with testing and validation

### Best Practices
- **Safety First**: Always implement safety mechanisms
- **Data Quality**: Ensure high-quality training data
- **Testing**: Thoroughly test in simulation before real robots
- **Monitoring**: Monitor performance and safety metrics

## Performance Guidelines

### Memory Usage
- **GPU Memory**: 8GB+ recommended for Pi0, 4GB+ for Pi0Fast
- **System RAM**: 16GB+ recommended
- **Batch Size**: Start with batch_size=1 for inference

### Inference Speed
- **First Call**: Slower due to JIT compilation
- **Subsequent Calls**: Fast inference (~10-50ms)
- **Optimization**: Use Pi0Fast for speed-critical applications

### Scaling
- **Multiple GPUs**: Distribute across multiple devices
- **Remote Inference**: Use WebSocket servers for scaling
- **Caching**: Cache models and preprocessing

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Slow Inference**: Ensure JIT compilation has occurred
3. **Import Errors**: Check installation and Python path
4. **Data Format**: Validate observation structure

### Debug Tools
- **Logging**: Enable debug logging
- **Profiling**: Use cProfile for performance analysis
- **Visualization**: Plot data and model outputs
- **Testing**: Use fake data for debugging

## What's Next?

1. **Start with [Getting Started](getting_started.md)** for basic usage
2. **Explore [Examples](examples_guide.md)** for your robot platform
3. **Read [API Reference](api_reference.md)** for detailed documentation
4. **Check [Transforms Guide](transforms_guide.md)** for data preprocessing
5. **Join the community** and contribute to the project

---

**OpenPI** - Enabling the future of robot learning through open source collaboration.