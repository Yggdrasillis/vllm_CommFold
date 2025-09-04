import os
import ray
import time
import socket
import torch
from typing import Optional
from vllm import LLM, EngineArgs
from ray.util.placement_group import placement_group, remove_placement_group

# Environment variable reset verification
os.environ['RAY_ADDRESS'] = "10.21.48.131:6379"  # Master node address

# Ensure distributed log output is visible
os.environ['RAY_DEDUP_LOGS'] = "0"  # Disable log deduplication
os.environ['PYTHONUNBUFFERED'] = "1"  # Disable Python output buffering

problematic_envs = [
    'CUDA_VISIBLE_DEVICES', 'VLLM_HOST_IP', 
    'VLLM_USE_RAY_EXCLUSIVE_NODE', 'RAY_NAMESPACE'
]
for var in problematic_envs:
    os.environ.pop(var, None)

# Network connection check utility
def check_network_connection(host: str, port: int) -> bool:
    """Check network connection reachability"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)  # 5 second timeout
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False

def detailed_network_diagnosis(host: str, port: int):
    """Detailed network diagnosis"""
    port_result = check_network_connection(host, port)
    common_ports = [6379, 10001, 8265]  
    for p in common_ports:
        result = check_network_connection(host, p)
    
    return port_result

# Hardware verification enhancement module
def hardware_diagnostics():
    """Comprehensive hardware diagnostics toolkit"""
    print("\nHardware diagnostics report")
    
    # GPU availability
    cuda_ready = torch.cuda.is_available()
    print(f"- CUDA availability: {'PASS' if cuda_ready else 'FAIL'}")
    if cuda_ready:
        print(f"  Detected {torch.cuda.device_count()} GPU devices")
        for i in range(torch.cuda.device_count()):
            dev = torch.cuda.get_device_properties(i)
    
    # Network connection verification
    ray_host, ray_port = os.environ['RAY_ADDRESS'].split(':')
    
    # Detailed node connection test
    head_connection = check_network_connection(ray_host, int(ray_port))

    
    # Test various possible nodes
    test_nodes = ['10.21.48.132']
    # Note: using 48 subnet interface
    node_results = {}
    for node in test_nodes:
        result = check_network_connection(node, 6379)
        node_results[node] = result
    
    # Cross-node interconnection status (success if at least one additional node is connectable)
    cross_node_ok = any(node_results.values()) and head_connection
    
    if not cross_node_ok:
        available_nodes = [node for node, ok in node_results.items() if ok]
    
    # Detailed diagnosis for failed nodes
    failed_nodes = [node for node, ok in node_results.items() if not ok]
    for failed_node in failed_nodes:
        detailed_network_diagnosis(failed_node, 6379)

def advanced_ray_init(max_retries=3):
    """Enhanced Ray initialization protocol"""
    for attempt in range(max_retries):
        try:
            # Clear historical remnants
            if ray.is_initialized():
                ray.shutdown()
            
            # Force cleanup of old placement groups
            try:
                all_pgs = ray.util.list_placement_groups()
                for pg in all_pgs:
                    remove_placement_group(pg)
            except Exception as e:
                print(f"Warning encountered when cleaning placement groups: {str(e)}")
            
            # Hardware-aware initialization
            return ray.init(
                address=os.environ['RAY_ADDRESS'],
                runtime_env={
                    "env_vars": {
                        "RAY_ENABLE_CUSTOM_RESOURCE_SCHEDULING": "1",
                        "RAY_USE_STRICT_PACK": "0",  # Disable strict resource packing
                        "RAY_DEDUP_LOGS": "0",  # Disable log deduplication
                        "PYTHONUNBUFFERED": "1"  # Disable output buffering
                    }
                },
                logging_level="INFO",
                log_to_driver=True  # Ensure worker logs are sent back to driver
            )
        except Exception as e:
            print(f"Initialization failed (attempt {attempt+1}): {str(e)}")
            time.sleep(5)
    raise RuntimeError("Unable to connect to Ray cluster")

def smart_parallel_config():
    """Intelligent parallel parameter planner"""
    if not ray.is_initialized():
        return 1, 1
    
    # Resource discovery with retry
    for _ in range(3):
        try:
            nodes = [n for n in ray.nodes() if n['Alive']]
            break
        except Exception as e:
            print(f"Resource acquisition failed: {str(e)}")
            time.sleep(2)
    
    # Build node topology
    node_map = {}
    for node in nodes:
        res = node.get('Resources', {})
        ip = node.get('NodeManagerAddress')
        gpu = int(res.get('GPU', 0))
        node_map[ip] = {
            'gpus': gpu,
            'labels': res.keys()
        }
    
    print(f"Cluster topology analysis (active nodes: {len(node_map)})")
    cross_node = len(node_map) >= 2 and sum(v['gpus'] for v in node_map.values()) >= 2
    
    # Automatically select parallel strategy
    if cross_node:
        tensor_size = sum(v['gpus'] for v in node_map.values())
        return tensor_size, 1  # Cross-node tensor parallelism
    else:
        return max(v['gpus'] for v in node_map.values()), 1  # Single-node multi-GPU

def robust_llm_initializer():
    """Exception-resistant model initialization engine"""
    moe_pipe_degree = 2
    
    base_cfg = {
        "model": "../model/DeepSeek-V3",
        "trust_remote_code": True,
        "quantization": "fp8", 
        "enforce_eager": True,  # Avoid compilation issues
        "enable_expert_parallel": True,  # Enable expert parallelism
        "hf_overrides": {"moe_pipe_degree": moe_pipe_degree}  # Override moe_pipe_degree
    }
    
    # Mode selection logic
    try:
        ray_ctx = advanced_ray_init()
        print(f"Ray connection successful | Dashboard: {ray_ctx.dashboard_url}")
        
        # Dynamic topology adaptation
        tp_size, pp_size = smart_parallel_config()
        moe_pipe_degree = base_cfg["hf_overrides"]["moe_pipe_degree"]
        print(f"Automatic parallel configuration: TP={tp_size}, PP={pp_size}")
        print(f"MoE configuration: Expert_Parallel=True, MoE_Pipe_Degree={moe_pipe_degree}")
        
        # Topology validation
        if tp_size > 1 and pp_size == 1:
            print("Enabling cross-node tensor parallelism")
            dist_config = {
                **base_cfg,
                "tensor_parallel_size": tp_size,
                "distributed_executor_backend": "ray",
                "disable_custom_all_reduce": False  # Enable optimized communication
            }
            return LLM(**dist_config)
        else:
            print("Fallback to single-machine mode")
            return LLM(**base_cfg)
            
    except Exception as e:
        print(f"Distributed initialization failed: {str(e)} Fallback to local mode")
        return LLM(**base_cfg)

def deployment_health_check(llm: LLM):
    """Deployment health status verification"""
    print("\nDeployment health report:")
    
    # Engine core detection
    try:
        engine = llm.llm_engine
        cfg = engine.model_config
        
        # Parallel status
        tp = cfg.parallel_config.tensor_parallel_size
        pp = cfg.parallel_config.pipeline_parallel_size
        print(f"- Parallel configuration: TP={tp}, PP={pp}")
        
        # Device topology
        devices = engine.parallel_config.worker_devices
        print(f"- Device layout: {devices if devices else 'Local'}")

        # Communication backend
        if hasattr(engine, 'custom_all_reduce'):
            print(f"- Communication optimization: {engine.custom_all_reduce.__class__.__name__}")
            
        # Distributed logging check
        if tp > 1:
            print("Note: When running multi-machine, dispatch time logs may be scattered across worker nodes")
            print("   - Check worker node logs: SSH to each node and view Ray worker process output")
            print("   - Or use: ray logs --follow to view real-time logs")
            
    except Exception as e:
        print(f"- Engine probe exception: {str(e)}")

if __name__ == "__main__":
    # System diagnostics
    hardware_diagnostics()
    
    # Initialize model engine
    print("\nInitializing inference engine...")
    llm = robust_llm_initializer()
    
    # Deployment health check
    deployment_health_check(llm)
    
    # Execute inference
    print("\nExecuting inference task...")
    
    try:
        with open('4k_token.txt', 'r', encoding='utf-8') as file:
            input_text = file.read().strip()
        print(f"Loaded test text, length: {len(input_text)} characters")
    except FileNotFoundError:
        print("4k_token.txt file not found, using default test text")
    except Exception as e:
        print(f"Failed to read file: {str(e)}, using default test text")
    
    outputs = llm.generate(input_text)
    print(f"Generation result: {outputs[0].outputs[0].text[:200]}...")