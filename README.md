# Model Optimization Pipeline

This project implements a comprehensive model optimization pipeline that includes pruning, knowledge distillation, and quantization. It's designed to reduce the size and increase the efficiency of deep learning models while maintaining performance.

## Features

- Model pruning using [Torch_pruning](https://github.com/VainF/Torch-Pruning) 
- Knowledge distillation 
- Model quantization using [TensorRT](https://github.com/NVIDIA-AI-IOT/torch2trt) 
- Flexible optimization targets (compression ratio or speed-up)
- Logging and experiment tracking with wandb


## Usage

You can use this colab notebook to try: [▶️ Example](https://colab.research.google.com/drive/1D9QxTZ4PcTbvc90nnfyVujgsEcGKfP_f?usp=sharing)

The main functions to use are
- `apply_pruning_and_kd`
- `apply_quantization`
- `optimize` 


Here's a basic example:

```python
from optimization_pipeline import optimize
import torch
import torchvision.models as models

# Load your model and data
model = models.resnet18(pretrained=True)
train_loader = ...  # Your training data loader
test_loader = ...   # Your test data loader
example_input = torch.randn(1, 3, 224, 224)  # Example input for your model

# Run optimization
optimized_model = optimize(
 model=model,
 traindataloader=train_loader,
 testdataloader=test_loader,
 example_input=example_input,
 num_classes=1000,
 epochs=120,
 lr=0.01,
 temperature=4,
 alpha=0.9,
 compression_ratio=2,
 bitwidth=16,
 wandb_project="my_optimization_project"
)
```
