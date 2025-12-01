import gc
import os
import psutil


def print_mem(stage: str = ""):
    """Prints a small memory snapshot (RSS and overall VM usage).

    Keeps it minimal so it can be called from dataset/train code for diagnostics.
    """
    gc.collect()
    proc = psutil.Process(os.getpid())
    mem = proc.memory_info()
    vmem = psutil.virtual_memory()
    rss_mb = mem.rss / (1024.0 * 1024.0)
    print(
        f"\n{stage} RSS MB: {rss_mb:.1f}, VM%: {vmem.percent}%, Available MB: {vmem.available / (1024 * 1024):.1f}"
    )
