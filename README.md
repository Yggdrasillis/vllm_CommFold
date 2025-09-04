
Enhanced MoE Expert Parallelism with Communication Overlap Based on vLLM 0.8.0
===
<br> 

# Installation
## Step 1: Clone the Repository

```bash
git clone https://github.com/Yggdrasillis/vllm_CommFold.git
cd vllm_CommFold
```

## Step 2: Create Python Environment

```bash
# Using conda (recommended)
conda create -n vllm_env python=3.10
conda activate vllm_env
```

## Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Ray for distributed computing
pip install ray[default]

# Install other required packages
pip install transformers
pip install accelerate
pip install sentencepiece
pip install protobuf
pip install huggingface_hub
```

## Step 4: Install Modified vLLM

```bash
# https://github.com/vllm-project/vllm/issues/12577#issuecomment-2757027368
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://files.pythonhosted.org/packages/15/77/7beca2061aadfdfd2d81411102e6445b459bcfedfc46671d4712de6a00fb/vllm-0.8.0-cp38-abi3-manylinux1_x86_64.whl

# Install modified vLLM with precompiled kernels (recommended for faster installation)
VLLM_USE_PRECOMPILED=1 pip install -e third_party/vllm  # This step might take a while
```
<br> 

# Download model
Use the provided script to download and prepare the DeepSeek-V3 model:

```bash
cd model
chmod +x load_partial_model.sh
./load_partial_model.sh
```

This script will **Download first 4 layers** of DeepSeek-V3 model (first 5 model files containing the initial layers)

<br> 

# Configuration

## Single-Node

Before running single node inference, configure the parameters in `workspace/single_node.py` :

```python
llm = LLM(
    model="../scripts/DeepSeek-V3",      # Model path - update to your model location
    trust_remote_code=True,             
    tensor_parallel_size=2,              # Number of GPUs for tensor parallelism (adjust based on available GPUs)
    enable_expert_parallel=True,         # Enable MoE expert parallelism
    enforce_eager=True,                  
    hf_overrides={"moe_pipe_degree": 2}  # The number of split chunks in a batch, used for overlapping communication and computation
)
```

## Multi-Node
Before running distributed inference, configure the parameters in `workspace/multi_node.py`: 

The script will automatically utilize all GPUs for tensor parallelism and expert parallelism.
### 1. Network Configuration

**Head Node Address (Line 11):**
```python
os.environ['RAY_ADDRESS'] = "10.21.48.131:6379"  # Update to your head node IP
```

**Worker Node List (Line 66):**
```python
test_nodes = ['10.21.48.132']  # Add all your worker node IPs
# Example for multiple workers:
# test_nodes = ['10.21.48.132', '10.21.48.133', '10.21.48.134']
```

### 2. Model Configuration

**Model Path (Line 159):**
```python
base_cfg = {
    "model": "../model/DeepSeek-V3",  # Ensure model path is accessible from all nodes
    "trust_remote_code": True,
    "quantization": "fp8",
    "enforce_eager": True,
    "enable_expert_parallel": True,
    "hf_overrides": {"moe_pipe_degree": moe_pipe_degree}
}
```

### 3. MoE Communication Overlap Configuration

**Pipeline Degree (Line 156):**
The number of split chunks in a batch, used for overlapping communication and computation
```python
moe_pipe_degree = 2  
```


<br>

# Execution

## Single Node Inference

For single GPU/node testing:

```bash
cd workspace
python single_node.py
```

## Multi-Node Distributed Inference
### Step 1: Prepare the Cluster

**On Head Node:**
```bash
cd workspace
./start_head.sh
```

**On Worker Nodes:**
```bash
cd workspace
./start_worker.sh <head_node_ip>:6379 --redis-password=<generated_password>

# Example:
./start_worker.sh 10.21.48.131:6379 --redis-password=5fb27b6da87c0b22e7a36d28849a4050
```

### Step 2: Verify Cluster Status

```bash
ray status  # Check cluster nodes and resources
ray dashboard  # Access web dashboard (default: http://head_node_ip:8265)
```

### Step 3: Run Distributed Inference
```bash
cd workspace
python multi_node.py
```

<br>

# Inference Results

```bash
(RayWorkerWrapper pid=308443) Prefill layer 3 forward time: 97.08338928222656 ms
```

**`Prefill layer 3`**: The 4th layer (0-indexed) of the model during prefill phase

**`forward time: 97.08ms`**: Time taken to process this layer (MoE layer performance)


