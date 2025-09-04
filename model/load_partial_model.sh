#!/bin/bash

# Clone model repo without large files
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V3
cd DeepSeek-V3

# Clear params pointers
rm model-00*

# We need use bf16 dtype, download fp8 params in a folder
mkdir fp8 && cd fp8

# Load first three MLP layers + single MoE layer
wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/model-00001-of-000163.safetensors
wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/model-00002-of-000163.safetensors
wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/model-00003-of-000163.safetensors
wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/model-00004-of-000163.safetensors
wget https://huggingface.co/deepseek-ai/DeepSeek-V3/resolve/main/model-00005-of-000163.safetensors

cd ..

# Edit config.json
sed -i 's/"num_hidden_layers": *[0-9]\+/"num_hidden_layers": 4/' config.json

# Convert to bf16
cp model.safetensors.index.json fp8/
cd inference

# FileNotFoundError: No such file or directory: "../fp8/model-00006-of-000163.safetensors"
# It's ok for that
python fp8_cast_bf16.py --input-fp8-hf-path ../fp8 --output-bf16-hf-path ../ || true

# I have edited the loader.py in third_party/vLLM folder
# Pls add setup.py for third_party
