"""
GPU memory utility helpers.

Shared cleanup and memory checking logic used by both the backend API and
the Gradio UI to keep memory-management behavior consistent.
"""
from __future__ import annotations

import gc

from typing import Any, Dict, Optional

import torch


def get_gpu_memory_info() -> Optional[Dict[str, Any]]:
    """Return a snapshot of current GPU memory usage or None if CUDA not available.

    Keys in returned dict: total_gb, allocated_gb, reserved_gb, free_gb, utilization
    """
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        reserved_memory = torch.cuda.memory_reserved(device)
        free_memory = total_memory - reserved_memory

        return {
            "total_gb": total_memory / 1024 ** 3,
            "allocated_gb": allocated_memory / 1024 ** 3,
            "reserved_gb": reserved_memory / 1024 ** 3,
            "free_gb": free_memory / 1024 ** 3,
            "utilization": (reserved_memory / total_memory) * 100,
        }
    except Exception:
        return None


def cleanup_cuda_memory() -> None:
    """Perform a robust GPU cleanup sequence.

    This includes synchronizing, emptying caches, collecting IPC handles and
    running the Python garbage collector. Use this instead of a raw
    ``torch.cuda.empty_cache()`` where you need reliable freeing of GPU memory
    between model loads or in error handling paths.
    """
    try:
        if torch.cuda.is_available():
            mem_before = get_gpu_memory_info()

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # Collect cross-process cuda resources
            try:
                torch.cuda.ipc_collect()
            except Exception:
                # Older PyTorch versions or non-cuda devices may not support
                # ipc_collect (no-op if not available)
                pass
            gc.collect()

            mem_after = get_gpu_memory_info()
            if mem_before and mem_after:
                freed = mem_before["reserved_gb"] - mem_after["reserved_gb"]
                print(
                    f"CUDA cleanup: freed {freed:.2f}GB, "
                    f"available: {mem_after['free_gb']:.2f}GB/{mem_after['total_gb']:.2f}GB"
                )
            else:
                print("CUDA memory cleanup completed")
    except Exception as e:
        print(f"Warning: CUDA cleanup failed: {e}")


def check_memory_availability(required_gb: float = 2.0) -> tuple[bool, str]:
    """Return whether at least ``required_gb`` seems available on the current GPU.

    The returned tuple is (is_available, message) with a human-friendly message.
    """
    try:
        if not torch.cuda.is_available():
            return False, "CUDA is not available"

        mem_info = get_gpu_memory_info()
        if mem_info is None:
            return True, "Cannot check memory, proceeding anyway"

        if mem_info["free_gb"] < required_gb:
            return (
                False,
                (
                    f"Insufficient GPU memory: {mem_info['free_gb']:.2f}GB available, "
                    f"{required_gb:.2f}GB required. Total: {mem_info['total_gb']:.2f}GB, "
                    f"Used: {mem_info['reserved_gb']:.2f}GB ({mem_info['utilization']:.1f}%)"
                ),
            )

        return (
            True,
            (
                f"Memory check passed: {mem_info['free_gb']:.2f}GB available, "
                f"{required_gb:.2f}GB required"
            ),
        )
    except Exception as e:
        return True, f"Memory check failed: {e}, proceeding anyway"
def estimate_memory_requirement(num_images: int, process_res: int) -> float:
    """Heuristic estimate for memory usage (GB) based on image count and resolution.

    This mirrors the simple policy used by the backend service so other code
    (e.g., Gradio UI) can make consistent decisions when checking available
    memory before loading a model or running inference.

    Args:
        num_images: Number of images to process.
        process_res: Processing resolution.

    Returns:
        Estimated memory requirement in GB.
    """
    base_memory = 2.0
    per_image_memory = (process_res / 504) ** 2 * 0.5
    total_memory = base_memory + (num_images * per_image_memory * 0.1)
    return total_memory
