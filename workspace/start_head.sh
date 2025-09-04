#!/bin/bash

# Ray Head Node Startup Script
# This script initializes the Ray head node for distributed inference

echo "Starting Ray Head Node..."

# Kill any existing Ray processes
ray stop --force 2>/dev/null || true
sleep 2

# Clean up any stale Ray processes
pkill -f ray:: 2>/dev/null || true
sleep 1

# Get the current node's IP address
HEAD_IP=$(hostname -I | awk '{print $1}')
echo "Head node IP: $HEAD_IP"

# Ray configuration
RAY_PORT=6379
DASHBOARD_PORT=8265
REDIS_PASSWORD=$(openssl rand -hex 16)

echo "Ray configuration:"
echo "  - Port: $RAY_PORT"
echo "  - Dashboard: http://$HEAD_IP:$DASHBOARD_PORT"
echo "  - Redis password: $REDIS_PASSWORD"

# Start Ray head node
ray start \
    --head \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$DASHBOARD_PORT \
    --redis-password=$REDIS_PASSWORD \
    --num-cpus=0 \
    --num-gpus=2 \
    --object-store-memory=50000000000 \
    --verbose

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Ray head node started successfully!"
    echo ""
    echo "Connection information:"
    echo "  - Ray address: $HEAD_IP:$RAY_PORT"
    echo "  - Dashboard: http://$HEAD_IP:$DASHBOARD_PORT"
    echo "  - Redis password: $REDIS_PASSWORD"
    echo ""
    echo "To connect worker nodes, run:"
    echo "  ./start_worker.sh $HEAD_IP:$RAY_PORT --redis-password=$REDIS_PASSWORD"
    echo ""
    echo "To stop the cluster:"
    echo "  ray stop"
    echo ""
    
    # Save connection info for easy access
    echo "export RAY_ADDRESS=$HEAD_IP:$RAY_PORT" > ray_config.env
    echo "export RAY_REDIS_PASSWORD=$REDIS_PASSWORD" >> ray_config.env
    echo "Connection info saved to ray_config.env"
    
else
    echo "❌ Failed to start Ray head node"
    exit 1
fi
