#!/bin/bash

# Ray Worker Node Startup Script
# This script connects a worker node to an existing Ray cluster

# Check if head node address is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <head_node_address> [additional_ray_options]"
    echo ""
    echo "Examples:"
    echo "  $0 10.21.48.131:6379"
    echo "  $0 10.21.48.131:6379 --redis-password=your_password"
    echo ""
    echo "If you have ray_config.env, you can also run:"
    echo "  source ray_config.env && $0 \$RAY_ADDRESS --redis-password=\$RAY_REDIS_PASSWORD"
    exit 1
fi

HEAD_ADDRESS=$1
shift  # Remove first argument, keep the rest as additional options

echo "Starting Ray Worker Node..."
echo "Head node address: $HEAD_ADDRESS"

# Kill any existing Ray processes
ray stop --force 2>/dev/null || true
sleep 2

# Clean up any stale Ray processes
pkill -f ray:: 2>/dev/null || true
sleep 1

# Get the current node's IP address
WORKER_IP=$(hostname -I | awk '{print $1}')
echo "Worker node IP: $WORKER_IP"

# GPU detection
GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l || echo "0")
echo "Detected GPUs: $GPU_COUNT"

# Start Ray worker node
echo "Connecting to Ray cluster at $HEAD_ADDRESS..."

ray start \
    --address=$HEAD_ADDRESS \
    --num-cpus=0 \
    --num-gpus=$GPU_COUNT \
    --object-store-memory=50000000000 \
    --verbose \
    "$@"  # Pass additional arguments

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Ray worker node connected successfully!"
    echo ""
    echo "Worker information:"
    echo "  - Worker IP: $WORKER_IP"
    echo "  - Connected to: $HEAD_ADDRESS"
    echo "  - GPUs available: $GPU_COUNT"
    echo ""
    echo "To disconnect from cluster:"
    echo "  ray stop"
    echo ""
    echo "To check cluster status:"
    echo "  ray status"
    echo ""
    
else
    echo "❌ Failed to connect worker node to Ray cluster"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Check if head node is running: ray status"
    echo "2. Verify network connectivity: ping <head_node_ip>"
    echo "3. Check if ports are open: telnet <head_node_ip> 6379"
    echo "4. Ensure Redis password matches (if using authentication)"
    exit 1
fi
