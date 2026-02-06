import gc
import numpy as np
import random
import torch


def print_gpu_tensors(logger=None, tag="", top_n=20):
    """Print all GPU tensors sorted by size for debugging memory leaks."""
    tensors = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                size_mb = obj.element_size() * obj.nelement() / 1024**2
                tensors.append((size_mb, obj.shape, obj.dtype, obj.device))
        except:
            pass
    
    # Sort by size descending
    tensors.sort(reverse=True, key=lambda x: x[0])
    
    total_size = sum(t[0] for t in tensors)
    print(f"\n{'='*60}", flush=True)
    print(f"[{tag}] GPU Tensors: {len(tensors)} | Total: {total_size:.2f} MB ({total_size/1024:.2f} GB)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Top {top_n} largest tensors:", flush=True)
    for i, (size, shape, dtype, device) in enumerate(tensors[:top_n]):
        print(f"  {i+1:3d}. {size:10.2f} MB | shape: {str(shape):40s} | dtype: {dtype}", flush=True)
    print(f"{'='*60}\n", flush=True)


def print_gpu_memory_summary(logger=None, tag=""):
    """Print a summary of GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[{tag}] GPU Memory: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, peak={max_allocated:.2f}GB", flush=True)


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


def merge_dict_list(dict_list):
    if len(dict_list) == 1:
        return dict_list[0]

    merged_dict = {}
    for k, v in dict_list[0].items():
        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                merged_dict[k] = torch.stack([d[k] for d in dict_list], dim=0)
            else:
                merged_dict[k] = torch.cat([d[k] for d in dict_list], dim=0)
        else:
            # for non-tensor values, we just copy the value from the first item
            merged_dict[k] = v
    return merged_dict
